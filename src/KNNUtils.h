#pragma once
#include "hnswlib/hnswlib.h"    // defines USE_SSE and USE_AVX and includes intrinsics

#if defined(__GNUC__)
#define PORTABLE_ALIGN32hnsw __attribute__((aligned(32)))
#else
#define PORTABLE_ALIGN32hnsw __declspec(align(32))
#endif

#include <QDebug>

#include <cmath>     // std::sqrt, exp
#include <vector>
#include <thread>
#include <atomic>

#include <Eigen/Dense>
#include <Eigen/Core>

/*!
 * 
 * 
 */
enum class knn_library : size_t
{
    KNN_HNSW = 0,       /*!<> */
};

/*!
 * The numerical value corresponds to the order in which each option is added to the GUI in SpidrSettingsWidget
  */
enum class knn_distance_metric : size_t
{
    KNN_METRIC_QF = 0,      /*!< Quadratic form distance */
    KNN_METRIC_EMD = 1,     /*!< Earth mover distance*/
    KNN_METRIC_HEL = 2,     /*!< Hellinger distance */
    KNN_METRIC_EUC = 3,     /*!< Euclidean distance - not suitable for histogram features */
    KNN_METRIC_PCOL = 4,     /*!< Collection distance between the neighborhoods around two items*/
};

/*!
 * 
 * 
 */
enum class ground_dist : size_t
{
    SIM_EUC = 0,    /*!<> */
    SIM_EXP = 1,    /*!<> */
};

namespace hnswlib {


    /*
     * The method is borrowed from nmslib
     */
    template<class Function>
    inline void ParallelFor(size_t start, size_t end, size_t numThreads, Function fn) {
        if (numThreads <= 0) {
            numThreads = std::thread::hardware_concurrency();
        }

        if (numThreads == 1) {
            for (size_t id = start; id < end; id++) {
                fn(id, 0);
            }
        }
        else {
            std::vector<std::thread> threads;
            std::atomic<size_t> current(start);

            // keep track of exceptions in threads
            // https://stackoverflow.com/a/32428427/1713196
            std::exception_ptr lastException = nullptr;
            std::mutex lastExceptMutex;

            for (size_t threadId = 0; threadId < numThreads; ++threadId) {
                threads.push_back(std::thread([&, threadId] {
                    while (true) {
                        size_t id = current.fetch_add(1);

                        if ((id >= end)) {
                            break;
                        }

                        try {
                            fn(id, threadId);
                        }
                        catch (...) {
                            std::unique_lock<std::mutex> lastExcepLock(lastExceptMutex);
                            lastException = std::current_exception();
                            /*
                             * This will work even when current is the largest value that
                             * size_t can fit, because fetch_add returns the previous value
                             * before the increment (what will result in overflow
                             * and produce 0 instead of current + 1).
                             */
                            current = end;
                            break;
                        }
                    }
                }));
            }
            for (auto &thread : threads) {
                thread.join();
            }
            if (lastException) {
                std::rethrow_exception(lastException);
            }
        }

    }


    // ---------------
    // Quadratic form for 1D Histograms
    // ---------------

    // data struct for distance calculation in QFSpace
    struct space_params_QF {
        size_t dim;
        size_t bin;
        ::std::vector<float> A;
    };

    static float
        QFSqr(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
        float* pVect1 = (float*)pVect1v;
        float* pVect2 = (float*)pVect2v;

        const space_params_QF* sparam = (space_params_QF*)qty_ptr;
        const size_t ndim = sparam->dim;
        const size_t nbin = sparam->bin;
        const float* pWeight = sparam->A.data();

        float res = 0;
        // add the histogram distance for each dimension
        for (size_t d = 0; d < ndim; d++) {
            for (size_t i = 0; i < nbin; i++) {
                float t1 = *(pVect1 + i) - *(pVect2 + i);
                for (size_t j = 0; j < nbin; j++) {
                    float t2 = *(pVect1 + j) - *(pVect2 + j);
                    res += *(pWeight + i * nbin + j) * t1 * t2;
                }
            }
            // point to next dimension
            pVect1 += nbin;
            pVect2 += nbin;
        }

        return res;
    }

    static float
        QFSqrSSE(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
        float* pVect1 = (float*)pVect1v;
        float* pVect2 = (float*)pVect2v;

        space_params_QF* sparam = (space_params_QF*)qty_ptr;
        const size_t ndim = sparam->dim;
        const size_t nbin = sparam->bin;

        size_t nbin4 = nbin >> 2 << 2;		// right shift by 2, left-shift by 2: create a multiple of 4

        float res = 0;
        float PORTABLE_ALIGN32hnsw TmpRes[8];			// memory aligned float array
        __m128 v1, v2, TmpSum, wRow, diff;			// write in registers of 128 bit size
        float *pA, *pEnd1, *pW, *pWend, *pwR;
        unsigned int wloc;

        // add the histogram distance for each dimension
        for (size_t d = 0; d < ndim; d++) {
            pA = sparam->A.data();					// reset to first weight for every dimension

           // calculate the QF distance for each dimension

           // 1. calculate w = (pVect1-pVect2)
            std::vector<float> w(nbin);
            wloc = 0;
            pEnd1 = pVect1 + nbin4;			// point to the first dimension not to be vectorized
            while (pVect1 < pEnd1) {
                v1 = _mm_loadu_ps(pVect1);					// Load the next four float values
                v2 = _mm_loadu_ps(pVect2);
                diff = _mm_sub_ps(v1, v2);					// substract all float values
                _mm_store_ps(&w[wloc], diff);				// store diff values in memory
                pVect1 += 4;								// advance pointer to position after loaded values
                pVect2 += 4;
                wloc += 4;
            }

            // manually calc the rest dims
            for (wloc; wloc < nbin; wloc++) {
                w[wloc] = *pVect1 - *pVect2;
                pVect1++;
                pVect2++;
            }

            // 2. calculate d = w'Aw
            for (unsigned int row = 0; row < nbin; row++) {
                TmpSum = _mm_set1_ps(0);
                pW = w.data();					// pointer to first float in w
                pWend = pW + nbin4;			// point to the first dimension not to be vectorized
                pwR = pW + row;
                wRow = _mm_load1_ps(pwR);					// load one float into all elements fo wRow

                while (pW < pWend) {
                    v1 = _mm_loadu_ps(pW);
                    v2 = _mm_loadu_ps(pA);
                    TmpSum = _mm_add_ps(TmpSum, _mm_mul_ps(wRow, _mm_mul_ps(v1, v2)));	// multiply all values and add them to temp sum values
                    pW += 4;
                    pA += 4;
                }
                _mm_store_ps(TmpRes, TmpSum);
                res += TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3];

                // manually calc the rest dims
                for (unsigned int uloc = nbin4; uloc < nbin; uloc++) {
                    res += *pwR * *pW * *pA;
                    pW++;
                    pA++;
                }
            }

            // point to next dimension is done in the last iteration
            // of the for loop in the rest calc under point 1. (no pVect1++ necessary here)
        }

        return res;
    }


    class QFSpace : public SpaceInterface<float> {

        DISTFUNC<float> fstdistfunc_;
        size_t data_size_;
        space_params_QF params_;

    public:
        // ground_weight might be set to (0.5 * sd of all data * ground_dist_max^2) as im doi:10.1006/cviu.2001.0934
        QFSpace(size_t dim, size_t bin, ground_dist ground_type = ground_dist::SIM_EUC, float ground_weight = 1) {
            qDebug() << "Distance Calculation: Prepare QFSpace";

            fstdistfunc_ = QFSqr;
            // Not entirely sure why this only shows positive effects for high bin counts...
            if (bin >= 12)
            {
                fstdistfunc_ = QFSqrSSE;
            }

            data_size_ = dim * bin * sizeof(float);

            ::std::vector<float> A;
            A.resize(bin*bin);
            size_t ground_dist_max = bin - 1;

            int bin_diff = 0;

            size_t ground_dist_max_2 = ground_dist_max * ground_dist_max;
            size_t bin_diff_2 = 0;

            if (ground_type == ground_dist::SIM_EUC) {
                for (int i = 0; i < bin; i++) {
                    for (int j = 0; j < bin; j++) {
                        bin_diff = (i - j);
                        bin_diff_2 = bin_diff * bin_diff;
                        A[i * bin + j] = 1 - std::sqrt(float(bin_diff_2) / float(ground_dist_max_2));
                    }
                }
            }
            else if (ground_type == ground_dist::SIM_EXP) {
                for (int i = 0; i < bin; i++) {
                    for (int j = 0; j < bin; j++) {
                        bin_diff = (i - j);
                        bin_diff_2 = bin_diff * bin_diff;
                        A[i * bin + j] = ::std::exp(-1 * ground_weight * (float(bin_diff_2) / float(ground_dist_max_2)));
                    }
                }
            }
        
            params_ = { dim, bin, A};
        }

        size_t get_data_size() {
            return data_size_;
        }

        DISTFUNC<float> get_dist_func() {
            return fstdistfunc_;
        }

        void *get_dist_func_param() {
            return (void *) &params_;
        }

        ~QFSpace() {}
    };



    // ---------------
    //    Hellinger
    // ---------------

    // data struct for distance calculation in HellingerSpace
    struct space_params_Hel {
        size_t dim;
        size_t bin;
    };

    static float
        HelSqr(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
        float *pVect1 = (float *)pVect1v;
        float *pVect2 = (float *)pVect2v;
       
        const space_params_Hel* sparam = (space_params_Hel*)qty_ptr;
        const size_t ndim = sparam->dim;
        const size_t nbin = sparam->bin;

        float res = 0;
        // add the histogram distance for each dimension
        for (size_t d = 0; d < ndim; d++) {
            for (size_t i = 0; i < nbin; i++) {
                float t = ::std::sqrt(*pVect1) - ::std::sqrt(*pVect2);
                pVect1++;
                pVect2++;
                res += t * t;
            }
        }
        return (res);
    }


    class HellingerSpace : public SpaceInterface<float> {

        DISTFUNC<float> fstdistfunc_;
        size_t data_size_;

        space_params_Hel params_;

    public:
        HellingerSpace(size_t dim, size_t bin) {
            qDebug() << "Distance Calculation: Prepare HellingerSpace";

            fstdistfunc_ = HelSqr;
            params_ = { dim, bin };
            data_size_ = dim * bin * sizeof(float);
        }

        size_t get_data_size() {
            return data_size_;
        }

        DISTFUNC<float> get_dist_func() {
            return fstdistfunc_;
        }

        void *get_dist_func_param() {
            return (void *) &params_;
        }

        ~HellingerSpace() {}
    };


    // ---------------
    //    Point collection distance
    // ---------------

    // data struct for distance calculation in PointCollectionSpace
    struct space_params_Col {
        size_t dim;
        size_t neighborhoodSize;        //  (2 * (params._numLocNeighbors) + 1) * (2 * (params._numLocNeighbors) + 1)
        DISTFUNC<float> L2distfunc_;
    };

    static float
        ColDist(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
        float *pVect1 = (float *)pVect1v;   // points to data item
        float *pVect2 = (float *)pVect2v;   // points to data item

        const space_params_Col* sparam = (space_params_Col*)qty_ptr;
        const size_t ndim = sparam->dim;
        const size_t neighborhoodSize = sparam->neighborhoodSize;
        DISTFUNC<float> L2distfunc_ = sparam->L2distfunc_;

        float res = 0;
        float minDist = 0;
        float tmpDist = 0;

        // Euclidean dist between all neighbor pairs
        // Take the min of all dists from a item in neigh1 to all items in Neigh2
        // Sum over all the min dists
        for (size_t n1 = 0; n1 < neighborhoodSize; n1++) {
            minDist = FLT_MAX;
            for (size_t n2 = 0; n2 < neighborhoodSize; n2++) {
                tmpDist = L2distfunc_( (pVect1 +(n1*ndim)), (pVect2 + (n2*ndim)), &ndim);

                if (tmpDist < minDist)
                    minDist = tmpDist;
            }
            res += minDist;
        }
        return (res);
    }

    class PointCollectionSpace : public SpaceInterface<float> {

        DISTFUNC<float> fstdistfunc_;
        size_t data_size_;

        space_params_Col params_;

    public:
        PointCollectionSpace(size_t dim, size_t neighborhoodSize) {
            fstdistfunc_ = ColDist;
            data_size_ = dim * sizeof(float);

            params_ = { dim, neighborhoodSize, L2Sqr };

#if defined(USE_SSE) || defined(USE_AVX)
            if (dim % 16 == 0)
                params_.L2distfunc_ = L2SqrSIMD16Ext;
            else if (dim % 4 == 0)
                params_.L2distfunc_ = L2SqrSIMD4Ext;
            else if (dim > 16)
                params_.L2distfunc_ = L2SqrSIMD16ExtResiduals;
            else if (dim > 4)
                params_.L2distfunc_ = L2SqrSIMD4ExtResiduals;
#endif
        }

        size_t get_data_size() {
            return data_size_;
        }

        DISTFUNC<float> get_dist_func() {
            return fstdistfunc_;
        }

        void *get_dist_func_param() {
            return &params_;
        }

        ~PointCollectionSpace() {}
    };

    // ---------------
    //    Wasserstein distance (EMD - Earth mover distance)
    // ---------------

    // data struct for distance calculation in QFSpace
    struct space_params_EMD {
        size_t dim;
        size_t bin;
        ::std::vector<float> A;
        float eps;
        float gamma;
    };

    static float
        EMD(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
        float* pVect1 = (float*)pVect1v;
        float* pVect2 = (float*)pVect2v;

        const space_params_EMD* sparam = (space_params_EMD*)qty_ptr;
        const size_t ndim = sparam->dim;
        const size_t nbin = sparam->bin;
        const float eps = sparam->eps;
        const float gamma = sparam->gamma;
        const float* pWeight = sparam->A.data();

        float res = 0;

        Eigen::VectorXf a = Eigen::Map<Eigen::VectorXf>(pVect1, nbin);
        Eigen::VectorXf b = Eigen::Map<Eigen::VectorXf>(pVect2, nbin);

        Eigen::VectorXf u = Eigen::VectorXf::Ones(a.size());
        Eigen::VectorXf v = Eigen::VectorXf::Ones(b.size());

        // for comparing differences between each sinkhorn iteration
        Eigen::VectorXf u_temp = u;
        Eigen::VectorXf v_temp = v;

        // ground distances and kernel
        Eigen::MatrixXf M = Eigen::Map<Eigen::MatrixXf>(pWeight, nbin, nbin);
        Eigen::MatrixXf K = (-1 * M / gamma).array().exp();
        Eigen::MatrixXf K_t = K.transpose();

        // sinkhorn iterations (fixpoint iteration)
        float iter_diff = 0;
        do {
            // update u, then v
            u = a.cwiseQuotient(K * v);
            v = b.cwiseQuotient(K_t * u);

            iter_diff = ((u - u_temp).squaredNorm() + (v - v_temp).squaredNorm()) / 2;
            u_temp = u;
            v_temp = v;

        } while (iter_diff > eps);

        // calculate divergence (inner product of ground distance and transportation matrix)
        Eigen::MatrixXf P = u.asDiagonal() * K * v.asDiagonal();
        float res = (M.cwiseProduct(P)).sum();	// implicit conversion to scalar only works for MatrixXd


        return res;
    }

    class EMDSpace : public SpaceInterface<float> {

        DISTFUNC<float> fstdistfunc_;
        size_t data_size_;
        space_params_QF params_;

    public:
        // ground_weight might be set to (0.5 * sd of all data * ground_dist_max^2) as im doi:10.1006/cviu.2001.0934
        EMDSpace(size_t dim, size_t bin) {
            qDebug() << "Distance Calculation: Prepare QFSpace";

            fstdistfunc_ = EMD;

            data_size_ = dim * bin * sizeof(float);

            ::std::vector<float> A;
            A.resize(bin * bin);

            for (int i = 0; i < (int)bin; i++)
                for (int j = 0; j < (int)bin; j++)
                    A[i * bin + j] = std::abs(i - j);// +1;

            params_ = { dim, bin, A };
        }

        size_t get_data_size() {
            return data_size_;
        }

        DISTFUNC<float> get_dist_func() {
            return fstdistfunc_;
        }

        void *get_dist_func_param() {
            return (void *)&params_;
        }

        ~EMDSpace() {}
    };

}
