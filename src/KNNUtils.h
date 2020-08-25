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
#include <algorithm> // std::find, fill
#include <vector>
#include <thread>
#include <atomic>

#include <Eigen/Dense>
#include <Eigen/Core>

#include "FeatureUtils.h"

/*! 
 * kNN library that is used (extended) for kNN computations
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
 * \param indMultiplier Size of one data item features
 * \param numPoints Number of points in the data
 * \param nn Number of kNN to compute
 * \return Tuple of knn Indices and respective squared distances
*/
template<typename T>
std::tuple<std::vector<int>, std::vector<float>> ComputekNN(const std::vector<T>* dataFeatures, hnswlib::SpaceInterface<float> *space, size_t indMultiplier, size_t numPoints, unsigned int nn);

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
        // add the histogram distance for each dimension
        for (size_t d = 0; d < ndim; d++) {
            // QF distance = sum_ij ( a_ij * (x_i-y_i) * (x_j-y_j) )
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
        ::std::vector<float> A;         // neighborhood similarity matrix
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
            data_size_ = dim * sizeof(float);

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
        space_params_Col_Appr() {};

        space_params_Col_Appr(size_t dim, ::std::vector<float> A, size_t neighborhoodSize, DISTFUNC<float> L2distfunc_, const std::vector<float>* dataFeatures, const std::vector<float>* attribute_data, unsigned int nn, std::vector<int> kNN_indices, std::vector<float> kNN_distances_squared) :
            dim(dim), A(A), neighborhoodSize(neighborhoodSize), L2distfunc_(L2distfunc_), dataFeatures(dataFeatures), attribute_data(attribute_data), nn(nn), kNN_indices(kNN_indices), kNN_distances_squared(kNN_distances_squared)
        {}

        size_t dim;
        ::std::vector<float> A;         // neighborhood similarity matrix
        size_t neighborhoodSize;        //  (2 * (params._numLocNeighbors) + 1) * (2 * (params._numLocNeighbors) + 1)
        DISTFUNC<float> L2distfunc_;
        const std::vector<float>* dataFeatures;         // pointer to data features: neighborhood indices
        const std::vector<float>* attribute_data;       // pointer to attributes of size dim
        unsigned int nn;                                // number of nearest neighbors
        std::vector<int> kNN_indices;                   // indices of kNN in datafeatures
        std::vector<float> kNN_distances_squared;       // distances corresponding to the kNN
    };

    /*! Estimate distance between itemID and centralNeighID
    * First, check whether any knn of itemID are the in neighborhood of centralNeighID. If not, calculate the min distance between itemID and all values in neighborhood of centralNeighID
    */
    static float
        ColDistNeighCalc(unsigned int itemID, unsigned int *centralNeighID, const unsigned int nn, const std::vector<int>* kNN_indices, const std::vector<float>* kNN_distances_squared, const std::vector<float>* dataFeatures, const std::vector<float>* attribute_data, const size_t ndim, const size_t neighborhoodSize, DISTFUNC<float> L2distfunc_) {

        float dist = FLT_MAX;

        for (unsigned int knn = 0; knn < nn; knn++) {

            // look up kNN indices for pVect1
            const int kNN_ID = (itemID * nn) + knn + 1;            // +1 because we don't count points themselves as nearest neighbors
            const int kNN_index = *(kNN_indices->data() + (kNN_ID));

            // check whether kNN index is in dataFeatures (neighbors)
            const float* neigh_start = dataFeatures->data() + (*centralNeighID * neighborhoodSize);
            const float* neigh_end = dataFeatures->data() + ((*centralNeighID + 1) * neighborhoodSize);
            const float* p_n1_knn_index = std::find(neigh_start, neigh_end, (float)kNN_index);

            if (p_n1_knn_index != neigh_end) {
                // -2 marks points outside the selection/image
                if (*p_n1_knn_index == -2.0f)
                    continue;

                // take approximated closest value
                dist = *(kNN_distances_squared->data() + (kNN_ID));
                break;
            }
        }

        // if no knn are in neighborhood, manually calc neighborhood distances and use the smallest
        if (dist == FLT_MAX) {
            float tmpDist = 0;
            const float* p_itemVal = attribute_data->data() + (itemID * ndim);    // pointer to first itemIds attribute values

            std::vector<float> nullAttr(ndim, 0);   // Replace values outside with zeros

            for (size_t neigh = 0; neigh < neighborhoodSize; neigh++) {

                float neighID = dataFeatures->at((*centralNeighID * neighborhoodSize) + neigh); // pointer to values in second points neighborhood

                const float* p_neighVal = (neighID != -2.0f) ? (attribute_data->data() + (unsigned int)neighID) : nullAttr.data();  // if neighID is outside selection, compare with 0
                tmpDist = L2distfunc_(p_itemVal, p_neighVal, &ndim);

                if (tmpDist < dist)
                    dist = tmpDist;
            }
        }

        assert(dist >= 0);
        assert(dist < FLT_MAX);

        return dist;
    }

    static float
        ColDistAppr(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
        unsigned int *pVect1 = (unsigned int *)pVect1v;   // points to item in _pointIds
        unsigned int *pVect2 = (unsigned int *)pVect2v;   // points to item in _pointIds

        // Easy access to parameters
        const space_params_Col_Appr* sparam = (space_params_Col_Appr*)qty_ptr;
        const size_t ndim = sparam->dim;
        const size_t neighborhoodSize = sparam->neighborhoodSize;
        const float* pWeight = sparam->A.data();
        DISTFUNC<float> L2distfunc_ = sparam->L2distfunc_;
        const std::vector<float>* dataFeatures = sparam->dataFeatures;                    // pointer to data features: neighborhood indices
        const std::vector<float>* attribute_data = sparam->attribute_data;
        const unsigned int nn = sparam->nn;
        const std::vector<int>* kNN_indices = &(sparam->kNN_indices);
        const std::vector<float>* kNN_distances_squared = &(sparam->kNN_distances_squared);

        float res = 0;
        float colDist = FLT_MAX;
        float rowDist = FLT_MAX;

        // Euclidean dist between all neighbor pairs
        // Take the min of all dists from a item in neigh1 to all items in Neigh2
        // This is aproximated using precalculated kNN
        // (the above can be written in a distance matrix, hence the column and row terms)
        // Sum over all the column-wise and row-wise minima
        for (size_t neigh = 0; neigh < neighborhoodSize; neigh++) {

            float An = dataFeatures->at((*pVect1 * neighborhoodSize) + neigh);
            float Bn = dataFeatures->at((*pVect2 * neighborhoodSize) + neigh);

            assert(An != -1.0f);    // This would mark an unprocessed value and possible error in FeatureExtraction
            assert(Bn != -1.0f);

            // for each col and row: look up if a knn if that item is in the neighborhood, else calc all distances
            // if item is outside the selection (ID == -2), simply assign no distance
            colDist = (An != -2.0f) ? ColDistNeighCalc((unsigned int)An, pVect2, nn, kNN_indices, kNN_distances_squared, dataFeatures, attribute_data, ndim, neighborhoodSize, L2distfunc_) : 0;
            rowDist = (Bn != -2.0f) ? ColDistNeighCalc((unsigned int)Bn, pVect1, nn, kNN_indices, kNN_distances_squared, dataFeatures, attribute_data, ndim, neighborhoodSize, L2distfunc_) : 0;

            // add (weighted) min of col and row
            res += colDist * *(pWeight + neigh) + rowDist * *(pWeight + neigh);
        }

        return (res);
    }

    class PointCollectionSpaceApprox : public SpaceInterface<float> {

        DISTFUNC<float> fstdistfunc_;
        size_t data_size_;

        space_params_Col_Appr params_;

    public:
        PointCollectionSpaceApprox(size_t numDims, size_t neighborhoodSize, size_t numPoints, const std::vector<float>* dataFeatures, const std::vector<float>* _attribute_data, loc_Neigh_Weighting weighting) {
            fstdistfunc_ = ColDistAppr;
            data_size_ = neighborhoodSize * sizeof(float);  // numDims

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

            // precalculate kNN with attribute data for PC distance approximation
            unsigned int nn = 91;
            std::vector<int> indices;
            std::vector<float> distances_squared;

            qDebug() << "PointCollectionSpaceApprox: Calculate kNN for distance approximation";

            SpaceInterface<float> *space = new L2Space(numDims);
            std::tie(indices, distances_squared) = ComputekNN(_attribute_data, space, numDims, numPoints, nn);

            // Debug check if all indices and distances are set
            assert(indices.size() == (numPoints*nn));
            assert(distances_squared.size() == (numPoints*nn));
            assert(std::find(indices.begin(), indices.end(), -1) == indices.end());
            assert(std::find(distances_squared.begin(), distances_squared.end(), -1) == distances_squared.end());

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

            params_ = space_params_Col_Appr(numDims, A, neighborhoodSize, L2distfunc_, dataFeatures, _attribute_data, nn, indices, distances_squared);

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

#pragma omp parallel
        for (size_t d = 0; d < ndim; d++) {

            Eigen::VectorXf a = Eigen::Map<Eigen::VectorXf>(pVect1 + (d*nbin), nbin);
            Eigen::VectorXf b = Eigen::Map<Eigen::VectorXf>(pVect2 + (d*nbin), nbin);

            assert(a.sum() == b.sum());     // the current implementation only works for histograms that contain the same number of entries (balanced form of Wasserstein distance)

            Eigen::VectorXf u = Eigen::VectorXf::Ones(a.size());
            Eigen::VectorXf v = Eigen::VectorXf::Ones(b.size());

            // for comparing differences between each sinkhorn iteration
            Eigen::VectorXf u_old = u;
            Eigen::VectorXf v_old = v;

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
            Eigen::MatrixXf P = u.asDiagonal() * K * v.asDiagonal();
#pragma omp atomic
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
