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
    class A2prop{
    public:
    	uint m,n,seed;  // edges and nodes
        int L;          // propagation levels
    	float rmax,alpha,rra,rrb;
        string dataset_name;
        vector<uint>el;
        vector<uint>pl;
        vector<float>rowsum_pos;
        vector<float>rowsum_neg;
        vector<uint>feat_map;
        vector<float>Du_a;
        vector<float>Du_b;
        vector<float>Du;
        float propagate(string dataset,string prop_alg,uint mm,uint nn,uint seedd,
                        int LL,float rmaxx,float alphaa,float ra,float rb,
                        Eigen::Map<Eigen::MatrixXf> &feat);
        void aseadj2 (Eigen::Ref<Eigen::MatrixXf>feats,int st,int ed);  // ASE(A^2)
        void prodadj2(Eigen::Ref<Eigen::MatrixXf>feats,int st,int ed);  // ASE(A^2) mul
        void featadj2(Eigen::Ref<Eigen::MatrixXf>feats,int st,int ed);  // sum A^2
        void featlap2(Eigen::Ref<Eigen::MatrixXf>feats,int st,int ed);  // sum L^2
        void featlapi(Eigen::Ref<Eigen::MatrixXf>feats,int st,int ed);  // sum (L+I)
    };

    class ApproxAdjProd {
    public:
        using Scalar = float;
        A2prop &a2prop;
    private:
        using sVector = Eigen::Matrix<Scalar, 1, Eigen::Dynamic>;
        using MapConstVec = Eigen::Map<const sVector>;
        using MapVec = Eigen::Map<sVector>;

    public:
        ApproxAdjProd(A2prop &a2prop) : a2prop(a2prop) {}

        int rows() const { return a2prop.n; }
        int cols() const { return a2prop.n; }

        void perform_op(const Scalar* x_in, Scalar* y_out) const {
            MapConstVec x(x_in, cols());
            MapVec y(y_out, rows());
            y = x;
            a2prop.prodadj2(y, 0, 1);
        }
    };
}

#endif // PROP_H
