#pragma once

#include "KNNUtils.h"
#include "FeatureUtils.h"

#include <string>

#include <QSize>

/*!
 * Stores all parameters used in the Spatial Analysis.
 * 
 * Used to set parameters for FeatureExtraction, DistanceCalculation and TsneComputatio 
 */
class Parameters {
public:
    Parameters() :
        _nn(-1), _numPoints(-1), _numDims(-1), _imgSize(-1, -1), _embeddingName(""), _dataVecBegin(NULL),
        _featureType(feature_type::TEXTURE_HIST_1D), _neighWeighting(loc_Neigh_Weighting::WEIGHT_UNIF), _numLocNeighbors(-1), _numHistBins(-1),
        _kernelWidth(0), _neighborhoodSize(0), _numFeatureValsPerPoint(0),
        _aknn_algorithm(knn_library::KNN_HNSW), _aknn_metric(distance_metric::METRIC_QF), _MVNweight(0),
        _perplexity(30), _perplexity_multiplier(3), _numIterations(1000), _exaggeration(250)
    {}

public:
    // data
    size_t              _numPoints;             /*!<> */
    size_t              _numDims;               /*!<> */
    QSize               _imgSize;               /*!<> */
    std::string         _embeddingName;         /*!< Name of the embedding */
    const float*        _dataVecBegin;          /*!< Points to the first element in the data vector */
    // features
    feature_type        _featureType;           /*!< texture histogram, LISA */
    size_t              _numFeatureValsPerPoint;/*!< depending on the feature type, the features vector has a different length (scalar features vs vector features per dimension)> */
    loc_Neigh_Weighting _neighWeighting;        /*!<> */
    size_t              _kernelWidth;           /*!<> */
    size_t              _neighborhoodSize;      /*!<> */
    size_t              _numLocNeighbors;       /*!<> */
    size_t              _numHistBins;           /*!<> */
    // distance
    size_t              _nn;                    // number of nearest neighbors, determined by _perplexity*_perplexity_multiplier + 1
    knn_library         _aknn_algorithm;        /*!<> */
    distance_metric     _aknn_metric;           /*!<> */
    float               _MVNweight;             /*!<> */
    // embeddings
    float               _perplexity;            //! Perplexity value in evert distribution.
    int                 _perplexity_multiplier; //! Multiplied by the perplexity gives the number of nearest neighbors used
    int                 _numIterations;         /*!< Number of gradient descent iterations> */
    int                 _exaggeration;          /*!< Number of iterations for early exxageration> */
};
