#pragma once

#include "KNNUtils.h"
#include "FeatureUtils.h"
#include <QSize>


/*!
 * Stores all parameters used in the Spatial Analysis.
 * 
 * Used to set parameters for FeatureExtraction, DistanceCalculation and TsneComputatio 
 */
class Parameters {
public:
    Parameters() :
        _nn(-1), _numPoints(-1), _numDims(-1), _imgSize(-1, -1),
        _featureType(feature_type::TEXTURE_HIST_1D), _neighWeighting(loc_Neigh_Weighting::WEIGHT_UNIF), _numLocNeighbors(-1), _numHistBins(-1),
        _aknn_algorithm(knn_library::KNN_HNSW), _aknn_metric(knn_distance_metric::KNN_METRIC_QF),
        _perplexity(30), _perplexity_multiplier(3)
    {}

public:
    // data
    size_t              _numPoints;             /*!<> */
    size_t              _numDims;               /*!<> */
    QSize               _imgSize;               /*!<> */
    // features
    feature_type        _featureType;           /*!< texture histogram, LISA */
    loc_Neigh_Weighting _neighWeighting;        /*!<> */
    size_t              _numLocNeighbors;       /*!<> */
    size_t              _numHistBins;           /*!<> */
    // distance
    size_t              _nn;                    // number of nearest neighbors, determined by _perplexity*_perplexity_multiplier + 1
    knn_library         _aknn_algorithm;        /*!<> */
    knn_distance_metric _aknn_metric;           /*!<> */
    // embeddings
    float               _perplexity;            //! Perplexity value in evert distribution.
    int                 _perplexity_multiplier; //! Multiplied by the perplexity gives the number of nearest neighbors used
    int                 _numIterations;         /*!<> */
    int                 _exaggeration;          /*!<> */
};