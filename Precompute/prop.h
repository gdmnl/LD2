/*
 * Author: nyLiao
 * File Created: 2023-04-19
 * File: prop.h
 */
#ifndef PROP_H
#define PROP_H
#include <iostream>
#include <vector>
#include <iostream>
#include <algorithm>
#include <random>
#include <queue>
#include <unordered_map>
#include <math.h>
#include <cmath>
#include <limits.h>
#include <unistd.h>
#include <math.h>
#include <fstream>
#include <sstream>
#include <assert.h>
#include <thread>
#include <string>
#include <unistd.h>
#include <chrono>
#include <omp.h>
#include <sys/time.h>
#include <sys/resource.h>
#pragma warning(push, 0)
#include <Eigen/Dense>
#include <Spectra/SymEigsSolver.h>
#pragma warning(pop)

using namespace std;
using namespace Eigen;
using namespace Spectra;
typedef unsigned int uint;

namespace propagation{
    const int NUMTHREAD = 32;       // Number of threads

    struct Channel {               // channel scheme
        int type;
			// -1		 -2
			// ASE(ADJ), ASE(ADJ2)
            // 0    1     2     3
            // ADJ, ADJi, ADJ2, ADJi2,
            // 4    5     6     7
            // LAP, LAPi, LAP2, LAPi2
        int powl;       // suffix'2': hop for one prop
        bool is_i;      // suffix'i': add identity
        bool is_adj;    // 'ADJ' or 'LAP'

        int L;          // propagation hop
        float rmax;     // absolute error
        float alpha;    // summation decay
        float rra, rrb; // left & right normalization
    };

    class A2prop{
    public:
    	uint m,n,seed;  // edges and nodes
        uint fdim,nchn; // feature dimension, number of channels
        string dataset_name;
        vector<uint> el;
        vector<uint> pl;
        vector<uint> feat_map;

        Channel* chns;
        Eigen::ArrayXf Du;
        Eigen::ArrayX4f Du_a;
        Eigen::ArrayXf dlt_p, dlt_n;

        void load(string dataset, uint mm, uint nn, uint seedd);
        float propagatea(uint nchnn, Channel* chnss, Eigen::Map<Eigen::MatrixXf> &feat);

        void feat_chn(Eigen::Ref<Eigen::MatrixXf>feats,int st,int ed);
        void aseadj2 (Eigen::Ref<Eigen::MatrixXf>feats,int ed);
        void prod_chn(Eigen::Ref<Eigen::ArrayXf> feats);
    };

    class ApproxAdjProd {
    public:
        using Scalar = float;
        A2prop &a2prop;
    private:

    public:
        ApproxAdjProd(A2prop &a2prop) : a2prop(a2prop) {}

        int rows() const { return a2prop.n; }
        int cols() const { return a2prop.n; }

        void perform_op(const Scalar* x_in, Scalar* y_out) const {
            Eigen::Map<const ArrayXf> x(x_in, cols());
            Eigen::Map<ArrayXf> y(y_out, rows());
            y = x;
            a2prop.prod_chn(y);
        }

        // TODO: https://spectralib.org/doc/classspectra_1_1densesymmatprod#a20efe97ecabc4d809ac10bfd1c1b0d53
        // Eigen::MatrixXf operator*  (const Eigen::Ref<const Eigen::MatrixXf>& mat_in) const { }
        // Scalar opertaor() (Eigen::Index i, Eigen::Index j) const { }
    };
}

#endif // PROP_H
