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

namespace Counter {
    static unsigned int co = 1;
    static unsigned int ci = 0;
}

typedef std::vector<hdi::data::MapMemEff<int, float> > sparse_scalar_matrix;

/*! 
 * kNN library that is used (extended) for kNN computations
 */
enum class knn_library : size_t
{
    NONE = 0,           /*!< No knn library in use, no approximation */ 
    KNN_HNSW = 1,       /*!< HNSWLib */
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
    KNN_METRIC_PCOLappr = 5,     /*!< (approx) Collection distance between the neighborhoods around two items*/
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

    int bin_diff = 0;

    size_t ground_dist_max_2 = ground_dist_max * ground_dist_max;
    size_t bin_diff_2 = 0;

    if (sim_type == bin_sim::SIM_EUC) {
        for (int i = 0; i < (int)num_bins; i++) {
            for (int j = 0; j < (int)num_bins; j++) {
                bin_diff = (i - j);
                bin_diff_2 = bin_diff * bin_diff;
                A[i * num_bins + j] = 1 - std::sqrt(float(bin_diff_2) / float(ground_dist_max_2));
            }
        }
    }
    else if (sim_type == bin_sim::SIM_EXP) {
        for (int i = 0; i < (int)num_bins; i++) {
            for (int j = 0; j < (int)num_bins; j++) {
                bin_diff = (i - j);
                bin_diff_2 = bin_diff * bin_diff;
                A[i * num_bins + j] = ::std::exp(-1 * sim_weight * (float(bin_diff_2) / float(ground_dist_max_2)));
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

/*! Compute kNN for a custom metric on features of a data set
 * \param dataFeatures Features used for distance calculation, dataFeatures->size() == (numPoints * indMultiplier)
 * \param space HNSWLib metric space
 * \param featureSize Size of one data item features
 * \param numPoints Number of points in the data
 * \param nn Number of kNN to compute
 * \return Tuple of knn Indices and respective squared distances
*/
template<typename T>
std::tuple<std::vector<int>, std::vector<float>> ComputekNN(const std::vector<T>* dataFeatures, hnswlib::SpaceInterface<float> *space, size_t featureSize, size_t numPoints, unsigned int nn);

/*! Compute all distance using a custom metric on features of a data set
 * \param dataFeatures Features used for distance calculation, dataFeatures->size() == (numPoints * indMultiplier)
 * \param space HNSWLib metric space
 * \param featureSize Size of one data item features
 * \param numPoints Number of points in the data
 * \return Tuple of indices and respective squared distances
*/
template<typename T>
std::tuple<std::vector<int>, std::vector<float>> ComputeExactKNN(const std::vector<T>* dataFeatures, hnswlib::SpaceInterface<float> *space, size_t featureSize, size_t numPoints, unsigned int nn) {
    std::vector<std::pair<int, float>> indices_distances;
    std::vector<int> knn_indices;
    std::vector<float> knn_distances_squared;

    assert(numPoints*numPoints < indices_distances.max_size());

    indices_distances.resize(numPoints*numPoints);
    knn_indices.resize(numPoints*nn, -1);
    knn_distances_squared.resize(numPoints*nn, -1.0f);

    hnswlib::DISTFUNC<float> distfunc = space->get_dist_func();
    void* params = space->get_dist_func_param();

    // Calculate a distance between all point combinations using the respective metric
//#pragma omp parallel for
    for (int i = 0; i < (int)numPoints; i++) {
        for (int j = 0; j < (int)numPoints; j++) {
            //indices[i * numPoints + j] = j;
            indices_distances[i * numPoints + j] = std::make_pair(j, distfunc(dataFeatures + i * featureSize, dataFeatures + j * featureSize, params));
        }
    }

    // find k nearest neighbors
    // get k smallest distances for each item
    for (int i = 0; i < (int)numPoints; i++) {
        // sort all distances to point i
        std::sort(indices_distances.begin() + i* numPoints, indices_distances.begin() + i* numPoints + numPoints, [](std::pair<int, float> a, std::pair<int, float> b) {return a.second < b.second;});

        // Take the first nn distances and indices 
        std::transform(indices_distances.begin() + i * nn, indices_distances.begin() + i * nn + nn, knn_indices.begin() + i * nn, [](const std::pair<int, float>& p) { return p.first; });
        std::transform(indices_distances.begin() + i * nn, indices_distances.begin() + i * nn + nn, knn_distances_squared.begin() + i * nn, [](const std::pair<int, float>& p) { return p.second; });

    }

    return std::make_tuple(knn_indices, knn_distances_squared);
}


hnswlib::SpaceInterface<float>* CreateHNSWSpace(knn_distance_metric knn_metric, size_t numDims, size_t numHistBins, size_t neighborhoodSize, loc_Neigh_Weighting neighborhoodWeighting, size_t numPoints, std::vector<float>* attribute_data);


namespace hnswlib {


    /* !
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

        float t = 0;
        float res = 0;
        // add the histogram distance for each dimension
        for (size_t d = 0; d < ndim; d++) {
            for (size_t i = 0; i < nbin; i++) {
                t = ::std::sqrt(*pVect1) - ::std::sqrt(*pVect2);
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
        ::std::vector<float> A;         // neighborhood similarity matrix
        size_t neighborhoodSize;        //  (2 * (params._numLocNeighbors) + 1) * (2 * (params._numLocNeighbors) + 1)
        DISTFUNC<float> L2distfunc_;
    };

    static float
        ColDist(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
        float *pVect1 = (float *)pVect1v;   // points to data item: values of neighbors
        float *pVect2 = (float *)pVect2v;   // points to data item: values of neighbors

        const space_params_Col* sparam = (space_params_Col*)qty_ptr;
        const size_t ndim = sparam->dim;
        const size_t neighborhoodSize = sparam->neighborhoodSize;
        const float* pWeight = sparam->A.data();
        DISTFUNC<float> L2distfunc_ = sparam->L2distfunc_;

        float res = 0;
        float colDist = FLT_MAX;
        std::vector<float> rowDist(neighborhoodSize, FLT_MAX);
        float tmpDist = 0;

        // Euclidean dist between all neighbor pairs
        // Take the min of all dists from a item in neigh1 to all items in Neigh2
        // (the above can be written in a distance matrix)
        // Sum over all the column-wise and row-wise minima
        for (size_t n1 = 0; n1 < neighborhoodSize; n1++) {
            colDist = FLT_MAX;
            for (size_t n2 = 0; n2 < neighborhoodSize; n2++) {
                tmpDist = L2distfunc_( (pVect1 +(n1*ndim)), (pVect2 + (n2*ndim)), &ndim);

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

    class PointCollectionSpace : public SpaceInterface<float> {

        DISTFUNC<float> fstdistfunc_;
        size_t data_size_;

        space_params_Col params_;

    public:
        PointCollectionSpace(size_t dim, size_t neighborhoodSize, loc_Neigh_Weighting weighting) {
            fstdistfunc_ = ColDist;
            data_size_ = dim * neighborhoodSize * sizeof(float);

            assert((::std::sqrt(neighborhoodSize) - std::floor(::std::sqrt(neighborhoodSize))) == 0);  // neighborhoodSize must be perfect square
            unsigned int _kernelWidth = (int)::std::sqrt(neighborhoodSize);

            ::std::vector<float> A (neighborhoodSize);
            switch (weighting)
            {
            case loc_Neigh_Weighting::WEIGHT_UNIF: std::fill(A.begin(), A.end(), 1); break;
            case loc_Neigh_Weighting::WEIGHT_BINO: A = BinomialKernel2D(_kernelWidth, norm_vec::NORM_MAX); break;        // weight the center with 1
            case loc_Neigh_Weighting::WEIGHT_GAUS: A = GaussianKernel2D(_kernelWidth, 1.0, norm_vec::NORM_NOT); break;
            default:  std::fill(A.begin(), A.end(), -1);  break;  // no implemented weighting type given. 
            }

            params_ = { dim, A, neighborhoodSize, L2Sqr };

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
    //    Point collection distance approx
    // ---------------

    // data struct for distance calculation in PointCollectionSpaceApprox
    struct space_params_Col_Appr {
        size_t dim;
        ::std::vector<float> A;         // neighborhood similarity matrix
        size_t neighborhoodSize;        //  (2 * (params._numLocNeighbors) + 1) * (2 * (params._numLocNeighbors) + 1)
        DISTFUNC<float> L2distfunc_;
        sparse_scalar_matrix kNN_lookup_table;
    };

    static float
        ColDistAppr(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
        float *pVect1 = (float *)pVect1v;   // points to data item: knn_indices of neighbors
        float *pVect2 = (float *)pVect2v;   // points to data item: knn_indices of neighbors

        space_params_Col_Appr* sparam = (space_params_Col_Appr*)qty_ptr;
        const size_t ndim = sparam->dim;
        const size_t neighborhoodSize = sparam->neighborhoodSize;
        const float* pWeight = sparam->A.data();
        DISTFUNC<float> L2distfunc_ = sparam->L2distfunc_;
        sparse_scalar_matrix* kNN_lookup_table = &(sparam->kNN_lookup_table);

        std::vector<float> rowDist(neighborhoodSize, FLT_MAX);
        std::vector<float> colDist(neighborhoodSize, FLT_MAX);
        float tmpDist = FLT_MAX;

        int An = 0; // index of point An
        int Bn = 0; // index of point Bn

        // Euclidean dist between all neighbor pairs
        // Take the min of all dists from a item in neigh1 to all items in Neigh2
        // (the above can be written in a distance matrix)
        // Sum over all the column-wise and row-wise minima

        // loop over all items in neighborhood A
        for (size_t n1 = 0; n1 < neighborhoodSize; n1++) {
            An = (int)*(pVect1 + n1);

            // loop over all items in neighborhood B
            for (size_t n2 = 0; n2 < neighborhoodSize; n2++) {
                Bn = (int)*(pVect2 + n2);

                // If An and Bn are inside the image/selection, look up the distance
                if ((An | Bn) >= 0)
                {
                    auto iter = (*kNN_lookup_table)[An].find(Bn);
                    if (iter != (*kNN_lookup_table)[An].end())
                        tmpDist = iter->second;
                    else
                        tmpDist = FLT_MAX;  // This will too often be the case for large images
                }
                else
                    tmpDist = 0;

                if (tmpDist < colDist[n1])
                    colDist[n1] = tmpDist;

                if (tmpDist < rowDist[n2])
                    rowDist[n2] = tmpDist;
            } // loop over all items in neighborhood B

        } //loop over all items in neighborhood A

        // add (weighted) min of all cols and rows
        return std::inner_product(colDist.begin(), colDist.end(), pWeight, 0.0f) + std::inner_product(rowDist.begin(), rowDist.end(), pWeight, 0.0f);
    }

    class PointCollectionSpaceApprox : public SpaceInterface<float> {

        DISTFUNC<float> fstdistfunc_;
        size_t data_size_;

        space_params_Col_Appr params_;

    public:
        PointCollectionSpaceApprox(size_t numDims, size_t neighborhoodSize, size_t numPoints, std::vector<float>* _attribute_data, loc_Neigh_Weighting weighting) {
            fstdistfunc_ = ColDistAppr;
            data_size_ = neighborhoodSize * sizeof(float); 

            assert((::std::sqrt(neighborhoodSize) - std::floor(::std::sqrt(neighborhoodSize))) == 0);  // neighborhoodSize must be perfect square
            unsigned int _kernelWidth = (int)::std::sqrt(neighborhoodSize);

            ::std::vector<float> A(neighborhoodSize);
            switch (weighting)
            {
            case loc_Neigh_Weighting::WEIGHT_UNIF: std::fill(A.begin(), A.end(), 1); break;
            case loc_Neigh_Weighting::WEIGHT_BINO: A = BinomialKernel2D(_kernelWidth, norm_vec::NORM_MAX); break;        // weight the center with 1
            case loc_Neigh_Weighting::WEIGHT_GAUS: A = GaussianKernel2D(_kernelWidth, 1.0, norm_vec::NORM_NOT); break;
            default:  std::fill(A.begin(), A.end(), -1);  break;  // no implemented weighting type given. 
            }

            qDebug() << "PointCollectionSpaceApprox: Calculate kNN for distance approximation";

            // precalculate kNN with attribute data for PC distance approximation
            unsigned int nn = 91;  // std::sqrt(numPoints)
            std::vector<int> knn_indices;
            std::vector<float> knn_distances_squared;
            SpaceInterface<float> *space = new L2Space(numDims);
            std::tie(knn_indices, knn_distances_squared) = ComputekNN(_attribute_data, space, numDims, numPoints, nn);

            // Debug check if all knn_indices and distances are set
            assert(knn_indices.size() == (numPoints*nn));
            assert(knn_distances_squared.size() == (numPoints*nn));
            assert(std::none_of(knn_indices.begin(), knn_indices.end(), [](int i) {return i == -1; }));
            assert(std::none_of(knn_distances_squared.begin(), knn_distances_squared.end(), [](float i) {return i == -1.0f; }));

            qDebug() << "PointCollectionSpaceApprox: Fill distance lookup table with kNN";

            // Fill distance lookup table
            sparse_scalar_matrix kNN_lookup_table;
            kNN_lookup_table.clear();
            kNN_lookup_table.resize(numPoints);
            for (unsigned int pointID = 0; pointID < numPoints; pointID++) {
                for (unsigned int neighID = 0; neighID < nn; neighID++) {
                    kNN_lookup_table[pointID][knn_indices[pointID * nn + neighID]] = knn_distances_squared[pointID * nn + neighID];
                }
            }

            // Choose distance function
            DISTFUNC<float> L2distfunc_ = L2Sqr;

#if defined(USE_SSE) || defined(USE_AVX)
            if (numDims % 16 == 0)
                L2distfunc_ = L2SqrSIMD16Ext;
            else if (numDims % 4 == 0)
                L2distfunc_ = L2SqrSIMD4Ext;
            else if (numDims > 16)
                L2distfunc_ = L2SqrSIMD16ExtResiduals;
            else if (numDims > 4)
                L2distfunc_ = L2SqrSIMD4ExtResiduals;
#endif

            params_ = { numDims, A, neighborhoodSize, L2distfunc_, kNN_lookup_table };

            qDebug() << "PointCollectionSpaceApprox: Finished space construction";
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

        ~PointCollectionSpaceApprox() {}
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
            float iter_diff = 0;
            do {
                // update u, then v
                u = a.cwiseQuotient(K * v);
                v = b.cwiseQuotient(K_t * u);

                iter_diff = ((u - u_old).squaredNorm() + (v - v_old).squaredNorm()) / 2;        // this might better be a percentage value
                u_old = u;
                v_old = v;

            } while (iter_diff > eps);

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
            float gamma = 0.5;

            params_ = { dim, bin, D, eps, gamma };
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
