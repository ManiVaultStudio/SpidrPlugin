#pragma once

#include <QSize>

#include <cmath>
#include <vector>
#include <exception>

enum loc_Neigh_Weighting
{
    WEIGHT_UNIF = 0,
    WEIGHT_BINO = 1,
    WEIGHT_GAUS = 2
};

enum norm_vec
{
    NORM_NOT = 0, // No normalization
    NORM_MAX = 1,
    NORM_SUM = 2
};


enum bin_size
{
    MANUAL = 0,
    SQRT = 1,
    STURGES = 2,
    RICE = 3
};

template<typename T>
std::vector<float> NormVector(std::vector<T> vec, float normVal);

std::vector<unsigned int> PascalsTriangleRow(const unsigned int n);

// @param norm: 1 indicates max, 2 indicates sum, 0 indicates no normalization
std::vector<float> BinomialKernel2D(const unsigned int width, norm_vec norm = norm_vec::NORM_NOT);

std::vector<float> GaussianKernel1D(const unsigned int width, const float sd = 1);

// @param norm: 1 indicates max, 2 indicates sum, 0 indicates no normalization
std::vector<float> GaussianKernel2D(const unsigned int width, const float sd = 1, norm_vec norm = norm_vec::NORM_NOT);

unsigned int SqrtBinSize(unsigned int numItems);

unsigned int SturgesBinSize(unsigned int numItems);

unsigned int RiceBinSize(unsigned int numItems);

