/*
 * Author: nyLiao
 * File Created: 2023-04-19
 * File: prop.cpp
 * Ref: [AGP](https://github.com/wanghzccls/AGP-Approximate_Graph_Propagation)
 */
#include "prop.h"
using namespace std;
using namespace Eigen;
using namespace Spectra;

// ====================
double getCurrentTime() {
    long long time = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
    return static_cast<double>(time) / 1000000.0;
}

float get_proc_memory(){
    struct rusage r_usage;
    getrusage(RUSAGE_SELF,&r_usage);
    return r_usage.ru_maxrss/1000000.0;
}

float get_stat_memory(){
    long rss;
    std::string ignore;
    std::ifstream ifs("/proc/self/stat", std::ios_base::in);
    ifs >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore
        >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore
        >> ignore >> ignore >> ignore >> rss;

    long page_size_kb = sysconf(_SC_PAGE_SIZE) / 1024;
    return rss * page_size_kb / 1000000.0;
}

inline void update_maxr(const float r, float &maxpr, float &maxnr) {
    if (r > maxpr)
        maxpr = r;
    else if (r < maxnr)
        maxnr = r;
}

// ====================
namespace propagation {

void A2prop::load(
        string dataset, uint mm, uint nn, uint nchnn, uint seedd,
        Eigen::Map<Eigen::MatrixXf> &feat) {
    dataset_name = dataset;
    m = mm;
    n = nn;
    seed = seedd;
    nchn = nchnn;

    // Load graph adjacency
    el = vector<uint>(m);   // edge list sorted by source node degree
    pl = vector<uint>(n + 1);
    string dataset_el = dataset + "/adj_el.bin";
    const char *p1 = dataset_el.c_str();
    if (FILE *f1 = fopen(p1, "rb")) {
        size_t rtn = fread(el.data(), sizeof el[0], el.size(), f1);
        if (rtn != m)
            cout << "Error! " << dataset_el << " Incorrect read!" << endl;
        fclose(f1);
    } else {
        cout << dataset_el << " Not Exists." << endl;
        exit(1);
    }
    string dataset_pl = dataset + "/adj_pl.bin";
    const char *p2 = dataset_pl.c_str();
    if (FILE *f2 = fopen(p2, "rb")) {
        size_t rtn = fread(pl.data(), sizeof pl[0], pl.size(), f2);
        if (rtn != n + 1)
            cout << "Error! " << dataset_pl << " Incorrect read!" << endl;
        fclose(f2);
    } else {
        cout << dataset_pl << " Not Exists." << endl;
        exit(1);
    }
    Du   = vector<float>(n, 0);
    for (uint i = 0; i < n; i++) {
        Du[i]   = pl[i + 1] - pl[i];
        if (Du[i] <= 0) {
            Du[i] = 1;
            // cout << i << " ";
        }
    }
}


float A2prop::propagatea(Channel* chnss, Eigen::Map<Eigen::MatrixXf> &feat) {
    L = chnss[0].L;                 // propagation hops
    rmax = chnss[0].rmax;
    alpha = chnss[0].alpha;         // $alpha$ in decaying summation
    rra = chnss[0].rra;               // left normalization
    rrb = chnss[0].rrb;               // right normalization
    Du_a = vector<float>(n, 0);
    Du_b = vector<float>(n, 0);
    for (uint i = 0; i < n; i++) {
        Du_a[i] = pow(Du[i], rra);      // normalized degree
        Du_b[i] = pow(Du[i], rrb);
    }

    // Feat is ColMajor, shape: (n, F)
    // cout << "feat dim: " << feat.cols() << ", nodes: " << feat.rows() << endl;
    fdim = feat.cols() / nchn;
    // cout  << "feat dim: " << feat.cols() << ", nodes: " << feat.cols() << endl;
    rowsum_pos = vector<float>(fdim, 0);
    rowsum_neg = vector<float>(fdim, 0);
    for (uint i = 0; i < fdim; i++) {
        for (uint u = 0; u < n; u++) {
            if (feat(u, i) > 0)
                rowsum_pos[i] += feat(u, i) * Du_b[u];
            else
                rowsum_neg[i] += feat(u, i) * Du_b[u];
        }
        if (rowsum_pos[i] == 0)
            rowsum_pos[i] = 1e-12;
        if (rowsum_neg[i] == 0)
            rowsum_neg[i] = -1e-12;
    }
    feat_map = vector<uint>(fdim);
    for (uint i = 0; i < fdim; i++)
        feat_map[i] = i;
    // random_shuffle(feat_map.begin(),feat_map.end());

    chns = chnss;

    // Begin propagation
    struct timeval ttod_start, ttod_end;
    double ttod, tclk;
    gettimeofday(&ttod_start, NULL);
    tclk = getCurrentTime();
    uint ti;
    int start, ends = 0;

    vector<thread> threads;
    for (ti = 1; ti <= fdim % NUMTHREAD; ti++) {
        start = ends;
        ends += ceil((float)fdim / NUMTHREAD);
        threads.push_back(thread(&A2prop::feat_chn, this, feat, start, ends));
    }
    for (; ti <= NUMTHREAD; ti++) {
        start = ends;
        ends += fdim / NUMTHREAD;
        threads.push_back(thread(&A2prop::feat_chn, this, feat, start, ends));
    }
    for (int t = 0; t < NUMTHREAD; t++)
        threads[t].join();
    vector<thread>().swap(threads);

    tclk = getCurrentTime() - tclk;
    gettimeofday(&ttod_end, NULL);
    ttod = ttod_end.tv_sec - ttod_start.tv_sec + (ttod_end.tv_usec - ttod_start.tv_usec) / 1000000.0;
    cout << "Prop  time: " << ttod << " \ts, \t";
    cout << "Clock time: " << tclk << " \ts" << endl;
    cout << "Max   PRAM: " << get_proc_memory() << " \tGB, \t";
    cout << "End    RAM: " << get_stat_memory() << " \tGB" << endl;
    return ttod;
}


void A2prop::feat_chn(Eigen::Ref<Eigen::MatrixXf> feats, int st, int ed) {
    uint seedt = seed;
    Eigen::VectorXf res0(n), res1(n);
    Eigen::Map<Eigen::VectorXf> rprev(res1.data(), n), rcurr(res0.data(), n);

    // Loop each feature `ift`, index `it`
    for (int it = st; it < ed; it++) {
        const uint ift = feat_map[it];
        const Channel chn = chns[ift / fdim];
        const float rmax_p = rowsum_pos[ift] * chn.rmax;
        const float rmax_n = rowsum_neg[ift] * chn.rmax;
        Eigen::Map<Eigen::VectorXf> feati(feats.col(ift).data(), n);

        // Init residue
        res1.setZero();
        // res0 = feats.col(ift);
        // feati.setZero();
        res0.setZero();
        res0.swap(feats.col(ift));
        rprev = res1;
        rcurr = res0;
        float MaxPR = res0.maxCoeff();  // max positive residue
        float MaxNR = res0.minCoeff();  // max negative residue

        // cout << it << " " << ift << endl;
        // Loop each hop `il`
        int il;
        for (il = 0; il < chn.L; il++) {
            // Early termination
            // TODO: Terminate condition for all positive feat
            if ((MaxPR <= rmax_p) && (MaxNR >= rmax_n))
                break;
            rcurr.swap(rprev);
            rcurr.setZero();

            // Loop each node `u`
            for (uint u = 0; u < n; u++) {
                const float old = rprev[u];
                float thr_p = old / rmax_p;
                float thr_n = old / rmax_n;
                // <<<<< suffix'i' (Identity) p-b i-d
                if (chn.is_i)
                    rcurr[u] += old;
                if (thr_p > 1 || thr_n > 1) {
                    uint im;
                    if ((chn.powl == 1) || (il % chn.powl == 1))
                        feati(u) += old;
                    // Loop each neighbor index `im`, node `v`
                    for (im = pl[u]; im < pl[u+1]; im++) {
                        const uint v = el[im];
                        const float da_v = Du_a[v];
                        if (thr_p > da_v || thr_n > da_v) {
                            rcurr[v] -= old / Du[v];
                            update_maxr(rcurr[v], MaxPR, MaxNR);
                        } else {
                            const float ran = rand_r(&seedt) % RAND_MAX / (float)RAND_MAX;
                            thr_p /= ran;
                            thr_n /= ran;
                            break;
                        }
                    }
                    for (; im < pl[u+1]; im++) {
                        const uint v = el[im];
                        const float da_v = Du_a[v];
                        if (thr_p > da_v) {
                            rcurr[v] -= rmax_p / da_v;
                            update_maxr(rcurr[v], MaxPR, MaxNR);
                        } else if (thr_n > da_v) {
                            rcurr[v] -= rmax_n / da_v;
                            update_maxr(rcurr[v], MaxPR, MaxNR);
                        } else
                            break;
                    }
                } else {
                    if ((chn.powl == 1) || (il % chn.powl == 1))
                        feati(u) += old;
                }
            }
        }

        // feati += rcurr;
        rcurr += feati;
        if (il % 2 == 1) {
            res0 = res1;
        }
        res0.swap(feats.col(ift));
    }
}


// ====================
// ASE(A^2)
void A2prop::aseadj2(Eigen::Ref<Eigen::MatrixXf> feats, int st, int dimension) {
    ApproxAdjProd op(*this);
    assert(st <= feats.rows());
    assert(L > 1);
    int nev = min(st+(int)floor(st*2/L), op.rows());
    SymEigsSolver<ApproxAdjProd> eigs(op, st, nev);

    // Max iter L (max oper L*nev), relative error 1e-2
    SortRule sorting;
    sorting = SortRule::LargestAlge;
    // sorting = SortRule::LargestMagn;
    eigs.init();
    int nconv = eigs.compute(sorting, L, 1e-2);

    Eigen::VectorXf evalues;
    evalues = eigs.eigenvalues();
    feats = eigs.eigenvectors(feats.rows()).transpose();
    for (int i = 0; i < feats.rows(); i++) {
        if (i < nconv) {
            feats.row(i) *= sqrtf(fabs(evalues[i]));
        }
        else {
            feats.row(i).setZero();
        }
    }

    // cout << "Eigenvalues: " << evalues.transpose() << endl;
    cout << " Num iter: " << eigs.num_iterations() << " Num oper: " << eigs.num_operations();
    cout << " Num conv: " << nconv << endl;
}

// ASE(A^2) mul
void A2prop::prodadj2(Eigen::Ref<Eigen::MatrixXf> feats, int st, int ed) {
    uint seedt = seed;
    float **residue = new float *[2];
    for (int i = 0; i < 2; i++)
        residue[i] = new float[n];  // two residue array, residue[j=L%2] for previous layer L-1, residue[k=1-L%2] for current layer L
    // Loop each feature `w`
    for (int it = st; it < ed; it++) {
        int w = it;
        float rowsum_p = 0;
        float rowsum_n = 0;
        for (uint j = 0; j < n; j++) {
            if (feats(w, j) > 0)
                rowsum_p += feats(w, j);
            else
                rowsum_n += feats(w, j);
        }
        float rmax_p = rowsum_p * rmax;
        float rmax_n = rowsum_n * rmax;

        // Init residue
        float MaxPR = 0;  // max positive residue
        float MaxNR = 0;  // max negative residue (absolute)
        for (uint ik = 0; ik < n; ik++) {
            residue[0][ik] = feats(w, ik) / Du_b[ik];
            residue[1][ik] = 0;
            update_maxr(residue[0][ik], MaxPR, MaxNR);
            // feats(w, ik) = 0;
            feats(w, ik) = - feats(w, ik) * Du[ik];
        }

        // Loop each hop `il`
        uint j = 0, k = 0;
        for (int il = 0; il <= 2; il++) {
            j = il % 2;
            k = 1 - j;
            // Output
            if (((MaxPR <= rmax_p) && (MaxNR >= rmax_n)) || (il == 2)) {
                for (uint ik = 0; ik < n; ik++) {
                    feats(w, ik) += residue[j][ik] * Du_b[ik];
                }
                break;
            }

            // Loop each node `ik`
            for (uint ik = 0; ik < n; ik++) {
                float old = residue[j][ik];
                residue[j][ik] = 0;
                if (old > rmax_p || old < rmax_n) {
                    uint im, v;
                    float ran;
                    // >>>>> comment out to set diag to 0
                    // residue[k][ik] -= old;
                    // update_maxr(residue[k][ik], MaxPR, MaxNR);
                    // <<<<<
                    // Loop each neighbor index `im`, node `v`
                    for (im = pl[ik]; im < pl[ik + 1]; im++) {
                        v = el[im];
                        if (old > rmax_p * Du_a[v] || old < rmax_n * Du_a[v]) {
                            residue[k][v] += old;
                            update_maxr(residue[k][v], MaxPR, MaxNR);
                        }
                        else {
                            ran = rand_r(&seedt) % RAND_MAX / (float)RAND_MAX;
                            break;
                        }
                    }
                    for (; im < pl[ik + 1]; im++) {
                        v = el[im];
                        if (ran * rmax_p * Du_a[v] < old) {
                            residue[k][v] += rmax_p / Du_a[v];
                            update_maxr(residue[k][v], MaxPR, MaxNR);
                        } else if (old < ran * rmax_n * Du_a[v]) {
                            residue[k][v] += rmax_n / Du_a[v];
                            update_maxr(residue[k][v], MaxPR, MaxNR);
                        } else
                            break;
                    }
                }
            }
        }
    }
    for (int i = 0; i < 2; i++)
        delete[] residue[i];
    delete[] residue;
}

// sum A^2
void A2prop::featadj2(Eigen::Ref<Eigen::MatrixXf> feats, int st, int ed) {
    uint seedt = seed;
    float **residue = new float *[2];
    for (int i = 0; i < 2; i++)
        residue[i] = new float[n];  // two residue array
    // Loop each feature `w`
    for (int it = st; it < ed; it++) {
        int w = feat_map[it];
        float rowsum_p = rowsum_pos[w];
        float rowsum_n = rowsum_neg[w];
        float rmax_p = rowsum_p * rmax;
        float rmax_n = rowsum_n * rmax;

        // Init residue
        float MaxPR = 0;  // max positive residue
        float MaxNR = 0;  // max negative residue(consider absolute value)
        for (uint ik = 0; ik < n; ik++) {
            residue[0][ik] = feats(w, ik) / Du_b[ik];
            residue[1][ik] = 0;
            update_maxr(residue[0][ik], MaxPR, MaxNR);
            feats(w, ik) = 0;
            // feats(w, ik) = - feats(w, ik) / Du_b[ik];
        }

        // Loop each hop `il`
        uint j = 0, k = 0;
        for (int il = 0; il <= L; il++) {
            j = il % 2;     // residue[j=L%2] for previous layer L-1
            k = 1 - j;      // residue[k=1-j] for current layer L
            // Output
            if (((MaxPR <= rmax_p) && (MaxNR >= rmax_n)) || (il == L)) {
                for (uint ik = 0; ik < n; ik++) {
                    feats(w, ik) += residue[j][ik] * Du_b[ik];
                }
                break;
            }

            // Loop each node `ik`
            for (uint ik = 0; ik < n; ik++) {
                float old = residue[j][ik];
                residue[j][ik] = 0;
                if (old > rmax_p || old < rmax_n) {
                    uint im, v;
                    float ran;
                    if (il % 2 == 1) {
                        feats(w, ik) += old * Du_b[ik];
                    }
                    // Loop each neighbor index `im`, node `v`
                    for (im = pl[ik]; im < pl[ik + 1]; im++) {
                        v = el[im];
                        if (old > rmax_p * Du_a[v] || old < rmax_n * Du_a[v]) {
                            residue[k][v] += old / Du[v];
                            update_maxr(residue[k][v], MaxPR, MaxNR);
                        } else {
                            ran = rand_r(&seedt) % RAND_MAX / (float)RAND_MAX;
                            break;
                        }
                    }
                    for (; im < pl[ik + 1]; im++) {
                        v = el[im];
                        if (ran * rmax_p * Du_a[v] < old) {
                            residue[k][v] += rmax_p / Du_a[v];
                            update_maxr(residue[k][v], MaxPR, MaxNR);
                        } else if (old < ran * rmax_n * Du_a[v]) {
                            residue[k][v] += rmax_n / Du_a[v];
                            update_maxr(residue[k][v], MaxPR, MaxNR);
                        } else
                            break;
                    }
                } else {
                    if (il % 2 == 1) {
                        feats(w, ik) += old * Du_b[ik];
                    }
                }
            }
        }
    }
    for (int i = 0; i < 2; i++)
        delete[] residue[i];
    delete[] residue;
}

// sum L^2
void A2prop::featlap2(Eigen::Ref<Eigen::MatrixXf> feats, int st, int ed) {
    uint seedt = seed;
    float **residue = new float *[2];
    for (int i = 0; i < 2; i++)
        residue[i] = new float[n];  // two residue array
    // Loop each feature `w`
    for (int it = st; it < ed; it++) {
        int w = feat_map[it];
        float rowsum_p = rowsum_pos[w];
        float rowsum_n = rowsum_neg[w];
        float rmax_p = rowsum_p * rmax;
        float rmax_n = rowsum_n * rmax;

        // Init residue
        float MaxPR = 0;  // max positive residue
        float MaxNR = 0;  // max negative residue(consider absolute value)
        for (uint ik = 0; ik < n; ik++) {
            residue[0][ik] = feats(w, ik) / Du_b[ik];
            residue[1][ik] = 0;
            update_maxr(residue[0][ik], MaxPR, MaxNR);
            feats(w, ik) = 0;
            // feats(w, ik) = - feats(w, ik) / Du_b[ik];
        }

        // Loop each hop `il`
        uint j = 0, k = 0;
        for (int il = 0; il <= L; il++) {
            j = il % 2;     // residue[j=L%2] for previous layer L-1
            k = 1 - j;      // residue[k=1-j] for current layer L
            // Output
            if (((MaxPR <= rmax_p) && (MaxNR >= rmax_n)) || (il == L)) {
                for (uint ik = 0; ik < n; ik++) {
                    feats(w, ik) += residue[j][ik] * Du_b[ik];
                }
                break;
            }

            // Loop each node `ik`
            for (uint ik = 0; ik < n; ik++) {
                float old = residue[j][ik];
                residue[j][ik] = 0;
                if (old > rmax_p || old < rmax_n) {
                    uint im, v;
                    float ran;
                    if (il % 2 == 1) {
                        feats(w, ik) += old * Du_b[ik];
                    }
                    // Loop each neighbor index `im`, node `v`
                    for (im = pl[ik]; im < pl[ik + 1]; im++) {
                        v = el[im];
                        if (old > rmax_p * Du_a[v] || old < rmax_n * Du_a[v]) {
                            residue[k][v] -= old / Du[v];
                            update_maxr(residue[k][v], MaxPR, MaxNR);
                        } else {
                            ran = rand_r(&seedt) % RAND_MAX / (float)RAND_MAX;
                            break;
                        }
                    }
                    for (; im < pl[ik + 1]; im++) {
                        v = el[im];
                        if (ran * rmax_p * Du_a[v] < old) {
                            residue[k][v] -= rmax_p / Du_a[v];
                            update_maxr(residue[k][v], MaxPR, MaxNR);
                        } else if (old < ran * rmax_n * Du_a[v]) {
                            residue[k][v] -= rmax_n / Du_a[v];
                            update_maxr(residue[k][v], MaxPR, MaxNR);
                        } else
                            break;
                    }
                } else {
                    if (il % 2 == 1) {
                        feats(w, ik) += old * Du_b[ik];
                    }
                }
            }
        }
    }
    for (int i = 0; i < 2; i++)
        delete[] residue[i];
    delete[] residue;
}

// sum (L+I)
void A2prop::featlapi(Eigen::Ref<Eigen::MatrixXf> feats, int st, int ed) {
    uint seedt = seed;
    float **residue = new float *[2];
    for (int i = 0; i < 2; i++)
        residue[i] = new float[n];  // two residue array
    // Loop each feature `w`
    for (int it = st; it < ed; it++) {
        int w = feat_map[it];
        float rowsum_p = rowsum_pos[w];
        float rowsum_n = rowsum_neg[w];
        float rmax_p = rowsum_p * rmax;
        float rmax_n = rowsum_n * rmax;

        // Init residue
        float MaxPR = 0;  // max positive residue
        float MaxNR = 0;  // max negative residue(consider absolute value)
        for (uint ik = 0; ik < n; ik++) {
            residue[0][ik] = feats(w, ik) / Du_b[ik];
            residue[1][ik] = 0;
            update_maxr(residue[0][ik], MaxPR, MaxNR);
            feats(w, ik) = 0;
            // feats(w, ik) = - feats(w, ik) / Du_b[ik];
        }

        // Loop each hop `il`
        uint j = 0, k = 0;
        for (int il = 0; il <= L; il++) {
            j = il % 2;     // residue[j=L%2] for previous layer L-1
            k = 1 - j;      // residue[k=1-j] for current layer L
            // Output
            if (((MaxPR <= rmax_p) && (MaxNR >= rmax_n)) || (il == L)) {
                for (uint ik = 0; ik < n; ik++) {
                    feats(w, ik) += residue[j][ik] * Du_b[ik];
                }
                break;
            }

            // Loop each node `ik`
            cout << w <<" "<< MaxPR <<" "<< MaxNR << endl;
            for (uint ik = 0; ik < n; ik++) {
                float old = residue[j][ik];
                residue[j][ik] = 0;
                residue[k][ik] += old;
                if (old > rmax_p || old < rmax_n) {
                    uint im, v;
                    float ran;
                    feats(w, ik) += old * Du_b[ik];
                    // Loop each neighbor index `im`, node `v`
                    for (im = pl[ik]; im < pl[ik + 1]; im++) {
                        v = el[im];
                        if (old > rmax_p * Du_a[v] || old < rmax_n * Du_a[v]) {
                            residue[k][v] -= old / Du[v];
                            update_maxr(residue[k][v], MaxPR, MaxNR);
                        } else {
                            ran = rand_r(&seedt) % RAND_MAX / (float)RAND_MAX;
                            break;
                        }
                    }
                    for (; im < pl[ik + 1]; im++) {
                        v = el[im];
                        if (ran * rmax_p * Du_a[v] < old) {
                            residue[k][v] -= rmax_p / Du_a[v];
                            update_maxr(residue[k][v], MaxPR, MaxNR);
                        } else if (old < ran * rmax_n * Du_a[v]) {
                            residue[k][v] -= rmax_n / Du_a[v];
                            update_maxr(residue[k][v], MaxPR, MaxNR);
                        } else
                            break;
                    }
                } else {
                    feats(w, ik) += old * Du_b[ik];
                }
            }
        }
    }
    for (int i = 0; i < 2; i++)
        delete[] residue[i];
    delete[] residue;
}


}  // namespace propagation
