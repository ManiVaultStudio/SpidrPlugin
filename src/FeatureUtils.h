#pragma once

#include "KNNUtils.h"

#include <QSize>

#include <vector>

enum loc_Neigh_Weighting
{
    WEIGHT_UNIF = 0,
    WEIGHT_BINO = 1,
    WEIGHT_GAUS = 2
};


std::vector<float> PascalsTriangleRow(const unsigned int n);
std::vector<float> BinomialKernel2D(const unsigned int n);

std::vector<float> GaussianKernel1D(const unsigned int n, const unsigned int sd=1);
std::vector<float> GaussianKernel2D(const unsigned int n, const unsigned int sd = 1);

class Parameters {
public:
    Parameters() :
        _perplexity(30), _perplexity_multiplier(3),
        _aknn_algorithm(knn_library::KNN_HNSW), _aknn_metric(knn_distance_metric::KNN_METRIC_QF), _neighWeighting(loc_Neigh_Weighting::WEIGHT_UNIF),
        _numHistBins(-1), _nn(-1), _numPoints(-1), _numDims(-1), _imgSize(-1, -1)
    {}

public:
    float               _perplexity;            //! Perplexity value in evert distribution.
    int                 _perplexity_multiplier; //! Multiplied by the perplexity gives the number of nearest neighbors used
    knn_library         _aknn_algorithm;
    knn_distance_metric _aknn_metric;
    loc_Neigh_Weighting _neighWeighting;
    unsigned int        _numHistBins;           // to be set in FeatureExtraction
    unsigned int        _nn;                    // number of nearest neighbors, determined by _perplexity*_perplexity_multiplier + 1; to be set in DistanceCalculation
    unsigned int        _numPoints;             // to be set in SpidrAnalysis
    unsigned int        _numDims;               // to be set in SpidrAnalysis
    QSize               _imgSize;               // to be set in SpidrAnalysis
};


