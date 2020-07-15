#pragma once

#include <QSize>

#include <cmath>
#include <vector>
#include <stdexcept>

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
    NORM_NOT = 0,   /*!<> */ // No normalization
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

// @param norm: 1 indicates max, 2 indicates sum, 0 indicates no normalization
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

