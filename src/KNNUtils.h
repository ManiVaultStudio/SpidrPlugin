#pragma once
#include "hnswlib/hnswlib.h"    // defines USE_SSE and USE_AVX and includes intrinsics

#if defined(__GNUC__)
#define PORTABLE_ALIGN32hnsw __attribute__((aligned(32)))
#else
#define PORTABLE_ALIGN32hnsw __declspec(align(32))
#endif

#include <QDebug>

#include <omp.h>

#include <cmath>     // std::sqrt, exp, floor
#include <numeric>   // std::inner_product
#include <algorithm> // std::find, fill, sort
#include <utility>   // std:: pair
#include <vector>
#include <thread>
#include <atomic>

#include "hdi/data/map_mem_eff.h" // hdi::data::MapMemEff

#include <chrono>

#include <Eigen/Dense>
#include <Eigen/Core>

#include "FeatureUtils.h"

/*! kNN library that is used kNN computations
 * The librarires are extended in order to work with different feature types
 */
enum class knn_library : size_t
{
    EVAL = 99,          /*!< No knn library in use, full dist matrix and save it to disk */ 
    NONE = 0,           /*!< No knn library in use, no approximation i.e. exact kNN computation */ 
    KNN_HNSW = 1,       /*!< HNSWLib */
};

/*! Defines the distance metric
  */
enum class distance_metric : size_t
{
    METRIC_QF = 0,       /*!< Quadratic form distance */
    METRIC_EMD = 1,      /*!< Earth mover distance*/
    METRIC_HEL = 2,      /*!< Hellinger distance */
    METRIC_EUC = 3,      /*!< Euclidean distance - not suitable for histogram features */
    METRIC_CHA = 4,     /*!< Chamfer distance (points collection)*/
};

/*!
 * Types of ground distance calculation that are used as the basis for bin similarities
 */
enum class bin_sim : size_t
{
    SIM_EUC = 0,    /*!< 1 - sqrt(Euclidean distance between bins)/(Max dist) */
    SIM_EXP = 1,    /*!< exp(-(Euclidean distance between bins)^2/(Max dist)) */
    SIM_UNI = 2,    /*!< 1 (uniform) */
};


/*!
 * Computes the similarities of bins of a 1D histogram.
 *
 * Entry A_ij refers to the sim between bin i and bin j. The diag entries should be 1, all others <= 1.
 *
 * \param num_bins    
 * \param sim_type type of ground distance calculation
 * \param sim_weight Only comes into play for ground_type = SIM_EXP, might be set to (0.5 * sd of all data * ground_dist_max^2) as im doi:10.1006/cviu.2001.0934
 * \return Matrix of neighborhood_width*neighborhood_width (stored in a vector) 
 */
static std::vector<float> BinSimilarities(size_t num_bins, bin_sim sim_type = bin_sim::SIM_EUC, float sim_weight = 1) {
    ::std::vector<float> A(num_bins*num_bins, -1);
    size_t ground_dist_max = num_bins - 1;

    if (sim_type == bin_sim::SIM_EUC) {
        for (int i = 0; i < (int)num_bins; i++) {
            for (int j = 0; j < (int)num_bins; j++) {
                A[i * num_bins + j] = 1 - (float(std::abs(i - j)) / float(ground_dist_max));
            }
        }
    }
    else if (sim_type == bin_sim::SIM_EXP) {
        for (int i = 0; i < (int)num_bins; i++) {
            for (int j = 0; j < (int)num_bins; j++) {
                A[i * num_bins + j] = ::std::exp(-1 * sim_weight * float(std::abs(i - j)));
            }
        }
    }
    else if (sim_type == bin_sim::SIM_UNI) {
        std::fill(A.begin(), A.end(), 1);
    }

    // if there is a -1 in A, this value was not set (invalid ground_type option selected)
    assert(std::find(A.begin(), A.end(), -1) == A.end());

    return A;
}

/*! Compute approximated kNN with a custom metric using HNSWLib
 * \param dataFeatures Features used for distance calculation, dataFeatures->size() == (numPoints * indMultiplier)
 * \param space HNSWLib metric space
 * \param featureSize Size of one data item features
 * \param numPoints Number of points in the data
 * \param nn Number of kNN to compute
 * \return Tuple of knn Indices and respective squared distances
*/
template<typename T>
std::tuple<std::vector<int>, std::vector<float>> ComputeHNSWkNN(const std::vector<T> dataFeatures, hnswlib::SpaceInterface<float> *space, size_t featureSize, size_t numPoints, unsigned int nn);

/*! Compute exact kNNs 
 * Calculate the distances between all point pairs and find closest neighbors
 * \param dataFeatures Features used for distance calculation, dataFeatures->size() == (numPoints * indMultiplier)
 * \param space HNSWLib metric space
 * \param featureSize Size of one data item features
 * \param numPoints Number of points in the data
 * \param nn Number of nearest neighbors
 * \param sort Whether to sort the nearest neighbor distances. Default is true. Set to false if nn == numPoints and you want to calculate the full distance matrix
 * \return Tuple of indices and respective squared distances
*/
template<typename T>
std::tuple<std::vector<int>, std::vector<float>> ComputeExactKNN(const std::vector<T> dataFeatures, hnswlib::SpaceInterface<float> *space, size_t featureSize, size_t numPoints, unsigned int nn, bool sort = true) {
    std::vector<std::pair<int, float>> indices_distances;
    std::vector<int> knn_indices;
    std::vector<float> knn_distances_squared;

    indices_distances.resize(numPoints);
    knn_indices.resize(numPoints*nn, -1);
    knn_distances_squared.resize(numPoints*nn, -1.0f);

    hnswlib::DISTFUNC<float> distfunc = space->get_dist_func();
    void* params = space->get_dist_func_param();

    // For each point, calc distances to all other
    // and take the nn smallest as kNN
    for (int i = 0; i < (int)numPoints; i++) {
        // Calculate distance to all points  using the respective metric
#ifdef NDEBUG
#pragma omp parallel for
#endif
        for (int j = 0; j < (int)numPoints; j++) {
            indices_distances[j] = std::make_pair(j, distfunc(dataFeatures.data() + i * featureSize, dataFeatures.data() + j * featureSize, params));
        }

        if (sort)
        {
            // sort all distances to point i
            std::sort(indices_distances.begin(), indices_distances.end(), [](std::pair<int, float> a, std::pair<int, float> b) {return a.second < b.second; });
        }

        // Take the first nn indices 
        std::transform(indices_distances.begin(), indices_distances.begin() + nn, knn_indices.begin() + i * nn, [](const std::pair<int, float>& p) { return p.first; });
        // Take the first nn distances 
        std::transform(indices_distances.begin(), indices_distances.begin() + nn, knn_distances_squared.begin() + i * nn, [](const std::pair<int, float>& p) { return p.second; });
    }

    return std::make_tuple(knn_indices, knn_distances_squared);
}

/*! Compute the full distance matrix between all data points
 * Calls ComputeExactKNN with the correct parameters, basically syntactic sugar
 * \param dataFeatures Features used for distance calculation, dataFeatures->size() == (numPoints * indMultiplier)
 * \param space HNSWLib metric space
 * \param featureSize Size of one data item features
 * \param numPoints Number of points in the data
 * \return Tuple of indices and respective squared distances
*/
template<typename T>
std::tuple<std::vector<int>, std::vector<float>> ComputeFullDistMat(const std::vector<T> dataFeatures, hnswlib::SpaceInterface<float> *space, size_t featureSize, size_t numPoints) {
    // set nn = numPoints and sort = false
    return ComputeExactKNN(dataFeatures, space, featureSize, numPoints, numPoints, false);
}

/*! Creates a metric space used by HNSWLib to build a kNN index
 * 
 * \param knn_metric distance metric to compare two points with
 * \param numDims Number of data channels
 * \param neighborhoodSize Size of neighborhood, must be a perfect square
 * \param neighborhoodWeighting Featureless distances use the weighting
 * \param numPoints Number of points in the data
 * \param attribute_data For use in some DEPRECATED distance metrics
 * \param numHistBins Number of histogram bins of feature type is a vector i.e. histogram
 * \return A HNSWLib compatible SpaceInterface, which is used as the basis to compare two points
 */
hnswlib::SpaceInterface<float>* CreateHNSWSpace(const distance_metric knn_metric, const size_t numDims, const size_t neighborhoodSize, const loc_Neigh_Weighting neighborhoodWeighting, const size_t numHistBins=0, const float* dataVecBegin = NULL);


/*! Calculates the size of an feature wrt to the feature type
 * Used as a step size for adding points to an HNSWlib index
 *
 * \param featureType type of feature (e.g. scalar LISA or vector Texture Histogram)
 * \param numDims Number of data channels
 * \param numHistBins Number of histogram bins of feature type is a vector i.e. histogram
 * \param neighborhoodSize Size of neighborhood, must be a perfect square
 * \return 
 */
const size_t NumFeatureValsPerPoint(const feature_type featureType, const size_t numDims, const size_t numHistBins, const size_t neighborhoodSize);

namespace hnswlib {


    /* !
     * The method is borrowed from nmslib, https://github.com/nmslib/nmslib/blob/master/similarity_search/include/thread_pool.h
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
        ::std::vector<float> A;     // bin similarity matrix for 1D histograms: entry A_ij refers to the sim between bin i and bin j 
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
        float t1 = 0;
        float t2 = 0;

        // add the histogram distance for each dimension
        for (size_t d = 0; d < ndim; d++) {
            // QF distance = sum_ij ( a_ij * (x_i-y_i) * (x_j-y_j) )
            for (size_t i = 0; i < nbin; i++) {
                t1 = *(pVect1 + i) - *(pVect2 + i);
                for (size_t j = 0; j < nbin; j++) {
                    t2 = *(pVect1 + j) - *(pVect2 + j);
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
        QFSpace(size_t dim, size_t bin, bin_sim ground_type = bin_sim::SIM_EUC) {
            qDebug() << "Distance Calculation: Prepare QFSpace";

            fstdistfunc_ = QFSqr;
            // Not entirely sure why this only shows positive effects for high bin counts...
            if (bin >= 12)
            {
                fstdistfunc_ = QFSqrSSE;
            }

            data_size_ = dim * bin * sizeof(float);

            ::std::vector<float> A = BinSimilarities(bin, ground_type);
            
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

        // Calculate Hellinger distance based on Bhattacharyya coefficient 
        float binSim = 0;
        float histDiff = 1;
        // add the histogram distance for each dimension
        for (size_t d = 0; d < ndim; d++) {
            histDiff = 1;
            for (size_t i = 0; i < nbin; i++) {
                binSim = (*pVect1) * (*pVect2);
                histDiff -= ::std::sqrt(binSim);
                pVect1++;
                pVect2++;
            }
            res += (histDiff>=0) ? ::std::sqrt(histDiff) : 0; // sometimes histDiff is slightly below 0 due to rounding errors
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
    //    Point cloud distance (Chamfer)
    // ---------------

    // data struct for distance calculation in PointCloudSpace
    struct space_params_Col {
        const float* dataVectorBegin;
        size_t dim;
        ::std::vector<float> A;         // neighborhood similarity matrix
        size_t neighborhoodSize;        //  (2 * (params._numLocNeighbors) + 1) * (2 * (params._numLocNeighbors) + 1)
        DISTFUNC<float> L2distfunc_;
    };

    static float
        ChamferDist(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
        float *pVect1 = (float *)pVect1v;   // points to first ID in neighborhood 1
        float *pVect2 = (float *)pVect2v;   // points to first ID in neighborhood 2

        const space_params_Col* sparam = (space_params_Col*)qty_ptr;
        const size_t ndim = sparam->dim;
        const size_t neighborhoodSize = sparam->neighborhoodSize;
        const float* dataVectorBegin = sparam->dataVectorBegin; 
        const float* pWeight = sparam->A.data();
        DISTFUNC<float> L2distfunc_ = sparam->L2distfunc_;

        const std::vector<int> idsN1(pVect1, pVect1 + neighborhoodSize);    // implicitly converts float to int
        const std::vector<int> idsN2(pVect2, pVect2 + neighborhoodSize);

        float res = 0;
        float colDist = FLT_MAX;
        std::vector<float> rowDist(neighborhoodSize, FLT_MAX);
        float tmpDist = 0;

        // Euclidean dist between all neighbor pairs
        // Take the min of all dists from a item in neigh1 to all items in Neigh2
        // (the above can be written in a distance matrix)
        // Sum over all the column-wise and row-wise minima
        for (size_t n1 = 0; n1 < neighborhoodSize; n1++) {
            if (idsN1[n1] == -2.0f)    // -1 is used for unprocessed locations during feature extraction, thus -2 indicated values outside image
                continue;
            colDist = FLT_MAX;
            for (size_t n2 = 0; n2 < neighborhoodSize; n2++) {
                if (idsN2[n2] == -2.0f)
                    continue;
                tmpDist = L2distfunc_(dataVectorBegin + (idsN1[n1] * ndim), dataVectorBegin + (idsN2[n2] * ndim), &ndim);

                if (tmpDist < colDist)
                    colDist = tmpDist;

                if (tmpDist < rowDist[n2]) 
                    rowDist[n2] = tmpDist;

            }
            // add (weighted) min of col
            res += colDist * *(pWeight + n1);
        }
        // add (weighted) min of all rows
        res += std::inner_product(rowDist.begin(), rowDist.end(), pWeight, 0.0f);

        return (res);
    }

    class PointCloudSpace : public SpaceInterface<float> {

        DISTFUNC<float> fstdistfunc_;
        size_t data_size_;

        space_params_Col params_;

    public:
        PointCloudSpace(size_t dim, size_t neighborhoodSize, loc_Neigh_Weighting weighting, const float* dataVectorBegin) {
            fstdistfunc_ = ChamferDist;
            data_size_ = neighborhoodSize * sizeof(float);

            assert((::std::sqrt(neighborhoodSize) - std::floor(::std::sqrt(neighborhoodSize))) == 0);  // neighborhoodSize must be perfect square
            unsigned int _kernelWidth = (int)::std::sqrt(neighborhoodSize);

            ::std::vector<float> A (neighborhoodSize);
            switch (weighting)
            {
            case loc_Neigh_Weighting::WEIGHT_UNIF: std::fill(A.begin(), A.end(), 1); break;
            case loc_Neigh_Weighting::WEIGHT_BINO: A = BinomialKernel2D(_kernelWidth, norm_vec::NORM_MAX); break;        // weight the center with 1
            case loc_Neigh_Weighting::WEIGHT_GAUS: A = GaussianKernel2D(_kernelWidth, 1.0, norm_vec::NORM_NONE); break;
            default:  std::fill(A.begin(), A.end(), -1);  break;  // no implemented weighting type given. 
            }

            params_ = { dataVectorBegin, dim, A, neighborhoodSize, L2Sqr };

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

        ~PointCloudSpace() {}
    };


    // ---------------
    //    Wasserstein distance (EMD - Earth mover distance)
    // ---------------

    // data struct for distance calculation in QFSpace
    struct space_params_EMD {
        size_t dim;
        size_t bin;
        ::std::vector<float> D;     // ground distance matrix
        float eps;                  // sinkhorn iteration update threshold
        unsigned int itMax;         // max sinkhorn iterations
        float gamma;                // entropic regularization multiplier
    };

    static float
        EMD_sinkhorn(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
        float* pVect1 = (float*)pVect1v;
        float* pVect2 = (float*)pVect2v;

        space_params_EMD* sparam = (space_params_EMD*)qty_ptr;      // no const because of pWeight
        const size_t ndim = sparam->dim;
        const size_t nbin = sparam->bin;
        const float eps = sparam->eps;
        unsigned int itMax = sparam->itMax;
        const float gamma = sparam->gamma;
        float* pGroundDist = sparam->D.data();                          // no const because of Eigen::Map

        float res = 0;

        // ground distances and kernel
        // the ground distance diag is 0 such that the kernel (here acting as a sim measure) has a diag of 1
        Eigen::MatrixXf M = Eigen::Map<Eigen::MatrixXf>(pGroundDist, nbin, nbin);
        Eigen::MatrixXf K = (-1 * M / gamma).array().exp();
        Eigen::MatrixXf K_t = K.transpose();

        Eigen::VectorXf a;  // histogram A, to which pVect1 points
        Eigen::VectorXf b;  // histogram B, to which pVect2 points

        Eigen::VectorXf u;  // sinkhorn update variable
        Eigen::VectorXf v;  // sinkhorn update variable
        Eigen::VectorXf u_old;  // sinkhorn update variable
        Eigen::VectorXf v_old;  // sinkhorn update variable

        Eigen::MatrixXf P;  // Optimal transport matrix

        for (size_t d = 0; d < ndim; d++) {

            a = Eigen::Map<Eigen::VectorXf>(pVect1 + (d*nbin), nbin);
            b = Eigen::Map<Eigen::VectorXf>(pVect2 + (d*nbin), nbin);

            assert(a.sum() == b.sum());     // the current implementation only works for histograms that contain the same number of entries (balanced form of Wasserstein distance)

            u = Eigen::VectorXf::Ones(a.size());
            v = Eigen::VectorXf::Ones(b.size());

            // for comparing differences between each sinkhorn iteration
            u_old = u;
            v_old = v;

            // sinkhorn iterations (fixpoint iteration)
            // introduce an additional break contidion (itCount) in case iter_diff does not converge 
            float iter_diff;
            unsigned int itCount;
            for(iter_diff=2*eps, itCount=0; iter_diff>eps && itCount < itMax; itCount++){
                // update u, then v
                u = a.cwiseQuotient(K * v);
                v = b.cwiseQuotient(K_t * u);

                iter_diff = ((u - u_old).squaredNorm() + (v - v_old).squaredNorm()) / 2;        // this might better be a percentage value
                u_old = u;
                v_old = v;

            } 

            // calculate divergence (inner product of ground distance and transportation matrix)
            P = u.asDiagonal() * K * v.asDiagonal();
            res += (M.cwiseProduct(P)).sum();
        }

        return res;
    }

    class EMDSpace : public SpaceInterface<float> {

        DISTFUNC<float> fstdistfunc_;
        size_t data_size_;
        space_params_EMD params_;

    public:
        // ground_weight might be set to (0.5 * sd of all data * ground_dist_max^2) as im doi:10.1006/cviu.2001.0934
        EMDSpace(size_t dim, size_t bin) {
            qDebug() << "Distance Calculation: Prepare EMDSpace";

            fstdistfunc_ = EMD_sinkhorn;

            data_size_ = dim * bin * sizeof(float);

            ::std::vector<float> D;
            D.resize(bin * bin);

            // ground distance between bin entries
            for (int i = 0; i < (int)bin; i++)
                for (int j = 0; j < (int)bin; j++)
                    D[i * bin + j] = std::abs(i - j);

            // these are fast parameters, but not the most accurate
            float eps = 0.1;
            unsigned int maxSinkhonIt = 10000;
            float gamma = 0.5;

            params_ = { dim, bin, D, eps, maxSinkhonIt, gamma };
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
