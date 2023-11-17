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
    if (r < maxnr)
        maxnr = r;
}

// ====================
namespace propagation {

float A2prop::propagate(
        string dataset, string prop_alg, uint mm, uint nn, uint seedd,
        int LL, float rmaxx, float alphaa, float ra, float rb,
        Eigen::Map<Eigen::MatrixXf> &feat) {
    m = mm;
    n = nn;
    L = LL;                 // propagation hops
    rmax = rmaxx;
    alpha = alphaa;         // $alpha$ in decaying summation
    rra = ra;               // left normalization
    rrb = rb;               // right normalization
    seed = seedd;
    dataset_name = dataset;

    el = vector<uint>(m);   // edge list sorted by source node degree
    pl = vector<uint>(n + 1);
    string dataset_el = "../data/" + dataset + "/adj_el.bin";
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
    string dataset_pl = "../data/" + dataset + "/adj_pl.bin";
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

    // Feat is ColMajor, shape: (F dimension, n)
    int dimension = feat.rows();
    feat_map = vector<uint>(dimension);
    rowsum_pos = vector<float>(dimension, 0);
    rowsum_neg = vector<float>(dimension, 0);
    for (int i = 0; i < dimension; i++)
        feat_map[i] = i;
    // random_shuffle(feat_map.begin(),feat_map.end());
    // cout << "feat size: " << feat.rows() << " " << feat.cols() << endl;

    Du   = vector<float>(n, 0);
    Du_a = vector<float>(n, 0);
    Du_b = vector<float>(n, 0);
    for (uint i = 0; i < n; i++) {
        Du[i]   = pl[i + 1] - pl[i];
        if (Du[i] <= 0) {
            Du[i] = 1;
            // cout << i << " ";
        }
        Du_a[i] = pow(Du[i], rra);      // normalized degree
        Du_b[i] = pow(Du[i], rrb);
    }
    for (int i = 0; i < dimension; i++) {
        for (uint j = 0; j < n; j++) {
            if (feat(i, j) > 0)
                rowsum_pos[i] += feat(i, j);
            else
                rowsum_neg[i] += feat(i, j);
        }
    }

    // Begin propagation
    struct timeval ttod_start, ttod_end;
    double ttod, tclk;
    gettimeofday(&ttod_start, NULL);
    tclk = getCurrentTime();
    int ti, start;
    int ends = 0;

    if (prop_alg == "aseadj2") {
        aseadj2(feat, dimension, dimension);
    } else {
        vector<thread> threads;
        for (ti = 1; ti <= dimension % NUMTHREAD; ti++) {
            start = ends;
            ends += ceil((float)dimension / NUMTHREAD);
            if (prop_alg == "featadj2")
                threads.push_back(thread(&A2prop::featadj2, this, feat, start, ends));
            else if (prop_alg == "featlap2")
                threads.push_back(thread(&A2prop::featlap2, this, feat, start, ends));
            else if (prop_alg == "featlapi")
                threads.push_back(thread(&A2prop::featlapi, this, feat, start, ends));
        }
        for (; ti <= NUMTHREAD; ti++) {
            start = ends;
            ends += dimension / NUMTHREAD;
            if (prop_alg == "featadj2")
                threads.push_back(thread(&A2prop::featadj2, this, feat, start, ends));
            else if (prop_alg == "featlap2")
                threads.push_back(thread(&A2prop::featlap2, this, feat, start, ends));
            else if (prop_alg == "featlapi")
                threads.push_back(thread(&A2prop::featlapi, this, feat, start, ends));
        }
        for (int t = 0; t < NUMTHREAD; t++)
            threads[t].join();
        vector<thread>().swap(threads);
    }

    tclk = getCurrentTime() - tclk;
    gettimeofday(&ttod_end, NULL);
    ttod = ttod_end.tv_sec - ttod_start.tv_sec + (ttod_end.tv_usec - ttod_start.tv_usec) / 1000000.0;
    cout << "Prop  time: " << ttod << " \ts, \t";
    cout << "Clock time: " << tclk << " \ts" << endl;
    cout << "Max   PRAM: " << get_proc_memory() << " \tGB, \t";
    cout << "End    RAM: " << get_stat_memory() << " \tGB" << endl;

    float dataset_size = (float)(((long long)m + n) * 4 + (long long)n * dimension * 8) / 1024.0 / 1024.0 / 1024.0;
    return dataset_size;
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
