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

inline void update_maxr(const float r, float &maxrp, float &maxrn) {
    if (r > maxrp)
        maxrp = r;
    else if (r < maxrn)
        maxrn = r;
}

// ====================
namespace propagation {

void A2prop::load(string dataset, uint mm, uint nn, uint seedd) {
    dataset_name = dataset;
    m = mm;
    n = nn;
    seed = seedd;

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

    Du = Eigen::ArrayXf::Zero(n);
    for (uint i = 0; i < n; i++) {
        Du(i)   = pl[i + 1] - pl[i];
        if (Du(i) <= 0) {
            Du(i) = 1;
            // cout << i << " ";
        }
    }
}


float A2prop::propagatea(uint nchnn, Channel* chnss, Eigen::Map<Eigen::MatrixXf> &feat) {
    nchn = nchnn;
    chns = chnss;
    assert(nchn <= 4);
    Du_a = Eigen::ArrayX4f::Zero(n, nchn);
    for (uint c = 0; c < nchn; c++) {
        Du_a.col(c) = Du.pow(chns[c].rra);
    }

    // cout << "feat dim: " << feat.cols() << ", nodes: " << feat.rows() << endl;
    const uint fsum = feat.cols();
    if (chns[0].type < 0) {
        // TODO: parallel and clocking
        aseadj2(feat, fsum);
        return 0;
    }

    // Feat is ColMajor, shape: (n, c*F)
    assert(fsum % nchn == 0);
    fdim = fsum / nchn;
    feat_map = vector<uint>(fsum);
    for (uint i = 0; i < fsum; i++)
        feat_map[i] = i;
    // random_shuffle(feat_map.begin(),feat_map.end());

    dlt_p = Eigen::ArrayXf::Zero(fsum);
    dlt_n = Eigen::ArrayXf::Zero(fsum);
    for (uint c = 0; c < nchn; c++) {
        for (uint i = 0; i < fdim; i++) {
            uint it = i + c * fdim;
            for (uint u = 0; u < n; u++) {
                if (feat(u, i) > 0)
                    dlt_p(it) += feat(u, it) * pow(Du(u), chns[c].rrb);
                else
                    dlt_n(it) += feat(u, it) * pow(Du(u), chns[c].rrb);
            }
            if (dlt_p(it) == 0)
                dlt_p(it) = 1e-12;
            if (dlt_n(it) == 0)
                dlt_n(it) = -1e-12;
            dlt_p(it) *= chns[c].rmax;
            dlt_n(it) *= chns[c].rmax;
        }
    }

    // Begin propagation
    struct timeval ttod_start, ttod_end;
    double ttod, tclk;
    gettimeofday(&ttod_start, NULL);
    tclk = getCurrentTime();
    uint ti;
    int start, ends = 0;

    vector<thread> threads;
    for (ti = 1; ti <= fsum % NUMTHREAD; ti++) {
        start = ends;
        ends += ceil((float)fsum / NUMTHREAD);
        threads.push_back(thread(&A2prop::feat_chn, this, feat, start, ends));
    }
    for (; ti <= NUMTHREAD; ti++) {
        start = ends;
        ends += fsum / NUMTHREAD;
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

// ====================
// Feature embs
void A2prop::feat_chn(Eigen::Ref<Eigen::MatrixXf> feats, int st, int ed) {
    uint seedt = seed;
    Eigen::VectorXf res0(n), res1(n);
    Eigen::Map<Eigen::VectorXf> rprev(res1.data(), n), rcurr(res0.data(), n);

    // Loop each feature `ift`, index `it`
    for (int it = st; it < ed; it++) {
        const uint ift = feat_map[it];
        const uint ic = ift / fdim;
        const Channel chn = chns[ic];
        const float dlti_p = dlt_p(ift);
        const float dlti_n = dlt_n(ift);
        Eigen::Map<Eigen::VectorXf> feati(feats.col(ift).data(), n);
        Eigen::Map<Eigen::ArrayXf> Deg_a(Du_a.col(ic).data(), n);

        // Init residue
        res1.setZero();
        res0 = feats.col(ift);
        feati.setZero();
        // res0.setZero();
        // res0.swap(feats.col(ift));
        rprev = res1;
        rcurr = res0;
        float maxr_p = res0.maxCoeff();  // max positive residue
        float maxr_n = res0.minCoeff();  // max negative residue

        // Loop each hop `il`
        int il;
        for (il = 0; il < chn.L; il++) {
            // Early termination
            // TODO: Terminate condition for all positive feat
            if ((maxr_p <= dlti_p) && (maxr_n >= dlti_n))
                break;
            rcurr.swap(rprev);
            rcurr.setZero();

            // Loop each node `u`
            for (uint u = 0; u < n; u++) {
                const float old = rprev[u];
                float thr_p = old / dlti_p;
                float thr_n = old / dlti_n;
                // <<<<< suffix'i' (Identity) p-b i-d
                if (chn.is_i)
                    rcurr[u] += old;
                if (thr_p > 1 || thr_n > 1) {
                    uint im;
                    if ((chn.powl == 1) || (il % chn.powl == 1))
                        feati(u) += old;

                    // Loop each neighbor index `im`, node `v`
                    const float old_t = (chn.is_adj) ? old : (-old);
                    for (im = pl[u]; im < pl[u+1]; im++) {
                        const uint v = el[im];
                        const float da_v = Deg_a(v);
                        if (thr_p > da_v || thr_n > da_v) {
                            rcurr[v] += old_t / Du(v);
                            update_maxr(rcurr[v], maxr_p, maxr_n);
                        } else {
                            const float ran = rand_r(&seedt) % RAND_MAX / (float)RAND_MAX;
                            thr_p /= ran;
                            thr_n /= ran;
                            break;
                        }
                    }

                    const float dlti_pt = (chn.is_adj) ? dlti_p : (-dlti_p);
                    const float dlti_nt = (chn.is_adj) ? dlti_n : (-dlti_n);
                    for (; im < pl[u+1]; im++) {
                        const uint v = el[im];
                        const float da_v = Deg_a(v);
                        if (thr_p > da_v) {
                            rcurr[v] += dlti_pt / da_v;
                            update_maxr(rcurr[v], maxr_p, maxr_n);
                        } else if (thr_n > da_v) {
                            rcurr[v] += dlti_nt / da_v;
                            update_maxr(rcurr[v], maxr_p, maxr_n);
                        } else
                            break;
                    }
                } else {
                    if ((chn.powl == 1) || (il % chn.powl == 1))
                        feati(u) += old;
                }
            }
        }

        feati += rcurr;
        // rcurr += feati;
        // if (il % 2 == 1) {
        //     res0 = res1;
        // }
        // res0.swap(feats.col(ift));
    }
}

// ASE wrapper
void A2prop::aseadj2(Eigen::Ref<Eigen::MatrixXf> feats, int ed) {
    ApproxAdjProd op(*this);
    assert(ed <= feats.cols());
    int nev = min(ed+(int)floor(ed*2/chns[0].L), op.cols());
    SymEigsSolver<ApproxAdjProd> eigs(op, ed, nev);

    // Max iter L (max oper L*nev), relative error 1e-2
    SortRule sorting = SortRule::LargestAlge;
    // SortRule sorting = SortRule::LargestMagn;
    eigs.init();
    int nconv = eigs.compute(sorting, chns[0].L, 1e-2);

    Eigen::VectorXf evalues;
    evalues = eigs.eigenvalues();
    feats = eigs.eigenvectors(feats.cols());
    for (int i = 0; i < feats.cols(); i++) {
        if (i < nconv) {
            feats.col(i) *= sqrtf(fabs(evalues[i]));
        }
        else {
            feats.col(i).setZero();
        }
    }

    // cout << "Eigenvalues: " << evalues.transpose() << endl;
    cout << " Num iter: " << eigs.num_iterations() << " Num oper: " << eigs.num_operations();
    cout << " Num conv: " << nconv << endl;
}

// Graph power iteration
void A2prop::prod_chn(Eigen::Ref<Eigen::ArrayXf> feats) {
    uint seedt = seed;
    Eigen::ArrayXf res0(n), res1(n);
    Eigen::Map<Eigen::ArrayXf> rprev(res1.data(), n), rcurr(res0.data(), n);

    const uint ic = 0;
    const Channel chn = chns[ic];
    const float dlti_p = feats.cwiseMax(0).sum() * chn.rmax;
    const float dlti_n = feats.cwiseMin(0).sum() * chn.rmax;
    Eigen::Map<Eigen::ArrayXf> Deg_a(Du_a.col(ic).data(), n);
    Eigen::ArrayXf Deg_b = Du.pow(chn.rrb);

    // Init residue
    res1.setZero();
    res0 = feats / Deg_b;
    feats *= -Du;
    rprev = res1;
    rcurr = res0;
    float maxr_p = res0.maxCoeff();  // max positive residue
    float maxr_n = res0.minCoeff();  // max negative residue

    // Loop each hop `il`
    int il;
    for (il = 0; il < chn.powl; il++) {
        // Early termination
        // TODO: Terminate condition for all positive feat
        if ((maxr_p <= dlti_p) && (maxr_n >= dlti_n))
            break;
        rcurr.swap(rprev);
        rcurr.setZero();

        // Loop each node `u`
        for (uint u = 0; u < n; u++) {
            const float old = rprev[u];
            float thr_p = old / dlti_p;
            float thr_n = old / dlti_n;

            if (thr_p > 1 || thr_n > 1) {
                uint im;

                // Loop each neighbor index `im`, node `v`
                const float old_t = (chn.is_adj) ? old : (-old);
                for (im = pl[u]; im < pl[u+1]; im++) {
                    const uint v = el[im];
                    const float da_v = Deg_a(v);
                    if (thr_p > da_v || thr_n > da_v) {
                        rcurr[v] += old_t;
                        update_maxr(rcurr[v], maxr_p, maxr_n);
                    } else {
                        const float ran = rand_r(&seedt) % RAND_MAX / (float)RAND_MAX;
                        thr_p /= ran;
                        thr_n /= ran;
                        break;
                    }
                }

                const float dlti_pt = (chn.is_adj) ? dlti_p : (-dlti_p);
                const float dlti_nt = (chn.is_adj) ? dlti_n : (-dlti_n);
                for (; im < pl[u+1]; im++) {
                    const uint v = el[im];
                    const float da_v = Deg_a(v);
                    if (thr_p > da_v) {
                        rcurr[v] += dlti_pt / da_v;
                        update_maxr(rcurr[v], maxr_p, maxr_n);
                    } else if (thr_n > da_v) {
                        rcurr[v] += dlti_nt / da_v;
                        update_maxr(rcurr[v], maxr_p, maxr_n);
                    } else
                        break;
                }
            }
        }
    }

    feats += rcurr;
}


}  // namespace propagation
