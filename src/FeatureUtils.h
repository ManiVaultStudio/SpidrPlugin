#pragma once

#include <QSize>

#include <cmath>
#include <vector>
#include <stdexcept>
#include <algorithm>    // for_each_n
#include <execution>    // par_unseq
#include <numeric>      // iota

/*! Types of neighborhood features
 *
 */
enum class feature_type : unsigned int
{
    TEXTURE_HIST_1D = 0,    /*!< Histograms of data point neighborhood, vector feature */
    LISA = 1,               /*!< Local Indicator of Spatial Associations, scalar feaure */
    GEARYC = 2,             /*!< Geary's, scalar feature */
    PCOL = 3,               /*!< Collection of points (neighborhood), i.e. no feature but a view of the data*/
};


/*!
 * 
 * 
 */
enum class loc_Neigh_Weighting : unsigned int
{
    WEIGHT_UNIF = 0,    /*!<> */
    WEIGHT_BINO = 1,    /*!<> */
    WEIGHT_GAUS = 2,    /*!<> */
};

/*!
 * 
 * 
 */
enum class norm_vec : unsigned int
{
    NORM_NOT = 0,   /*!<No normalization */
    NORM_MAX = 1,   /*!<> */
    NORM_SUM = 2,   /*!<> */
};

/*!
 * 
 * 
 */
enum class bin_size : unsigned int
{
    MANUAL = 0,     /*!<> */
    SQRT = 1,       /*!<> */
    STURGES = 2,    /*!<> */
    RICE = 3,       /*!<> */
};

/*!
 *
 * \tparam  T
 * \arg  cev
 * \arg  nornmVal
 * \return
 */
template<typename T>
std::vector<float> NormVector(std::vector<T> vec, float normVal);

/*!
 *
 * \param n
 * \return
 */
std::vector<unsigned int> PascalsTriangleRow(const unsigned int n);

/*!
 *
 * \param width
 * \param norm
 * \return
 */
std::vector<float> BinomialKernel2D(const unsigned int width, norm_vec norm = norm_vec::NORM_NOT);

/*!
 *
 * \param width
 * \param sd
 * \return
 */
std::vector<float> GaussianKernel1D(const unsigned int width, const float sd = 1);

// @param norm: 1 indicates max, 2 indicates sum, 0 indicates no normalization
/*!
 *
 * \param width
 * \param sd
 * \param norm
 * \return
 */
std::vector<float> GaussianKernel2D(const unsigned int width, const float sd = 1, norm_vec norm = norm_vec::NORM_NOT);

/*!
 *
 * \param numItems
 * \return
 */
unsigned int SqrtBinSize(unsigned int numItems);

/*!
 *
 * \param numItems
 * \return
 */
unsigned int SturgesBinSize(unsigned int numItems);

/*!
 *
 * \param numItems
 * \return
 */
unsigned int RiceBinSize(unsigned int numItems);

/*! Get neighborhood point ids for one data item
 * For now, expect a rectangle selection (lasso selection might cause edge cases that were not thought of)
 * Padding: assign -1 to points outside the selection. Later assign 0 vector to all of them.
 * \param pointInd
 * \param locNeighbors
 * \param imgSize
 * \param pointIds
 * \return 
 */
std::vector<int> neighborhoodIndices(const unsigned int pointInd, const size_t locNeighbors, const QSize imgSize, const std::vector<unsigned int>& pointIds);

/*! Get data for all neighborhood point ids
 * Padding: if neighbor is outside selection, assign 0 to all dimension values
 * 
 * \param neighborIDs
 * \param _attribute_data
 * \param _neighborhoodSize
 * \param _numDims
 * \return 
 */
std::vector<float> getNeighborhoodValues(const std::vector<int>& neighborIDs, const std::vector<float>& attribute_data, const size_t neighborhoodSize, const size_t numDims);

/*! Calculate the minimum and maximum value for each channel
 *
 * \param numPoints
 * \param numDims
 * \param attribute_data
 * \return vector with [min_Ch0, max_Ch0, min_Ch1, max_Ch1, ...]
 */
template<typename T>
std::vector<float> CalcMinMaxPerChannel(size_t numPoints, size_t numDims, const std::vector<T>& attribute_data) {
    std::vector<float> minMaxVals(2 * numDims, 0);

    // for each dimension iterate over all values
    // remember data stucture (point1 d0, point1 d1,... point1 dn, point2 d0, point2 d1, ...)
    for (unsigned int dimCount = 0; dimCount < numDims; dimCount++) {
        // init min and max
        float currentVal = attribute_data[dimCount];
        minMaxVals[2 * dimCount] = currentVal;
        minMaxVals[2 * dimCount + 1] = currentVal;

        for (unsigned int pointCount = 0; pointCount < numPoints; pointCount++) {
            currentVal = attribute_data[pointCount * numDims + dimCount];
            // min
            if (currentVal < minMaxVals[2 * dimCount])
                minMaxVals[2 * dimCount] = currentVal;
            // max
            else if (currentVal > minMaxVals[2 * dimCount + 1])
                minMaxVals[2 * dimCount + 1] = currentVal;
        }
    }

    return minMaxVals;
}

/*! Calculate the mean value for each channel
 *
 * \param numPoints
 * \param numDims
 * \param attribute_data
 * \return vector with [mean_Ch0, mean_Ch1, ...]
 */
template<typename T>
std::vector<float> CalcMeanPerChannel(size_t numPoints, size_t numDims, const std::vector<T>& attribute_data) {
    std::vector<float> meanVals(numDims, 0);

#pragma omp parallel for 
    for (int dimCount = 0; dimCount < (int)numDims; dimCount++) {
        float sum = 0;
        for (unsigned int pointCount = 0; pointCount < numPoints; pointCount++) {
            sum += attribute_data[pointCount * numDims + dimCount];
        }

        meanVals[dimCount] = sum / numPoints;
    }

    return meanVals;
}

/*! Calculate estimate of the variance
 *  Assuming equally likely values, a (biased) estimated of the variance is computed for each dimension
 *
 * \param numPoints
 * \param numDims
 * \param attribute_data
 * \return vector with [var_Ch0, var_Ch1, ...]
 */
template<typename T>
std::vector<float> CalcVarEstimate(size_t numPoints, size_t numDims, const std::vector<T>& attribute_data, const std::vector<float> &meanVals) {
    std::vector<float> varVals(numDims, 0);

#pragma omp parallel for 
    for (int dimCount = 0; dimCount < (int)numDims; dimCount++) {
        float sum = 0;
        float temp_diff = 0;
        for (unsigned int pointCount = 0; pointCount < numPoints; pointCount++) {
            temp_diff = attribute_data[pointCount * numDims + dimCount] - meanVals[dimCount];
            sum += (temp_diff * temp_diff);
        }

        varVals[dimCount] = sum / numPoints;

    }

    return varVals;
}