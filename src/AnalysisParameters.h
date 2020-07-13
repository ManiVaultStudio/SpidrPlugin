#pragma once

#include "KNNUtils.h"
#include "FeatureUtils.h"
#include <QSize>


class Parameters {
public:
    Parameters() :
        _nn(-1), _numPoints(-1), _numDims(-1), _imgSize(-1, -1),
        _neighWeighting(loc_Neigh_Weighting::WEIGHT_UNIF), _numLocNeighbors(-1), _numHistBins(-1),
        _aknn_algorithm(knn_library::KNN_HNSW), _aknn_metric(knn_distance_metric::KNN_METRIC_QF),
        _perplexity(30), _perplexity_multiplier(3)
    {}

public:
    // data
    unsigned int        _nn;                    // number of nearest neighbors, determined by _perplexity*_perplexity_multiplier + 1; to be set in DistanceCalculation
    unsigned int        _numPoints;             // to be set in SpidrAnalysis
    unsigned int        _numDims;               // to be set in SpidrAnalysis
    QSize               _imgSize;               // to be set in SpidrAnalysis
    // features
    loc_Neigh_Weighting _neighWeighting;        //
    unsigned int        _numLocNeighbors;       //
    unsigned int        _numHistBins;           // to be set in FeatureExtraction
    // distance
    knn_library         _aknn_algorithm;
    knn_distance_metric _aknn_metric;
    // embeddings
    float               _perplexity;            //! Perplexity value in evert distribution.
    int                 _perplexity_multiplier; //! Multiplied by the perplexity gives the number of nearest neighbors used
};
