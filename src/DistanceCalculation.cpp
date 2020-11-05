#include "DistanceCalculation.h"

#include "AnalysisParameters.h"
#include "KNNUtils.h"
#include "EvalUtils.h"

#include "hnswlib/hnswlib.h"

#include <QDebug>

#include <chrono>
#include <algorithm>    // std::none_of

DistanceCalculation::DistanceCalculation() :
    _knn_lib(knn_library::KNN_HNSW),
    _knn_metric(distance_metric::METRIC_QF)
{
}

DistanceCalculation::~DistanceCalculation()
{
}

void DistanceCalculation::setup(std::vector<unsigned int>& pointIds, std::vector<float>& attribute_data, std::vector<float>* dataFeatures, Parameters& params) {
    _featureType = params._featureType;

    // Parameters
    _knn_lib = params._aknn_algorithm;
    _knn_metric = params._aknn_metric;
    _nn = params._nn;
    _neighborhoodSize = (2 * (params._numLocNeighbors) + 1) * (2 * (params._numLocNeighbors) + 1); // square neighborhood with _numLocNeighbors to each side from the center
    _neighborhoodWeighting = params._neighWeighting;

    // Data
    // Input
    _numPoints = params._numPoints;
    _numDims = params._numDims;
    _numHistBins = params._numHistBins;
    _dataFeatures = dataFeatures;
    _pointIds = &pointIds;
    _attribute_data = &attribute_data;
    _embeddingName = params._embeddingName;

    // Output
    //_knn_indices.resize(_numPoints*_nn, -1);              // unnecessary, done in ComputeHNSWkNN
    //_knn_distances_squared.resize(_numPoints*_nn, -1);    // unnecessary, done in ComputeHNSWkNN

    assert(params._nn == (size_t)(params._perplexity * params._perplexity_multiplier + 1));     // should be set in SpidrAnalysis::initializeAnalysisSettings

    if (_featureType == feature_type::TEXTURE_HIST_1D)
    {
        assert(_dataFeatures->size() == (_numPoints * _numDims * _numHistBins));
        qDebug() << "Distance calculation: Feature values per point: " << _numDims * _numHistBins << "Number of NN to calculate" << _nn << ". Metric: " << (size_t)_knn_metric;
    }
    else if ((_featureType == feature_type::LISA) | (_featureType == feature_type::GEARYC)) {
        assert(_dataFeatures->size() == (_numPoints * _numDims));
        qDebug() << "Distance calculation: Feature values per point: " << _numDims << "Number of NN to calculate" << _nn << ". Metric: " << (size_t)_knn_metric;
    }
    else if (_featureType == feature_type::PCOL) {
        assert(_dataFeatures->size() == (_numPoints * _numDims * _neighborhoodSize));
        qDebug() << "Distance calculation: Feature values per point: " << _numDims * _neighborhoodSize << "Number of NN to calculate" << _nn << ". Metric: " << (size_t)_knn_metric;
    }

    // -1 would mark an unset feature
    assert(std::none_of(_dataFeatures->begin(), _dataFeatures->end(), [](float i) {return i == -1.0f; }));
}

void DistanceCalculation::compute() {
    qDebug() << "Distance calculation: started";

    computekNN();

    qDebug() << "Distance calculation: finished";

}

void DistanceCalculation::computekNN() {
    
    qDebug() << "Distance calculation: Setting up metric space";
    auto t_start_CreateHNSWSpace = std::chrono::steady_clock::now();

    // setup hsnw index
    hnswlib::SpaceInterface<float> *space = CreateHNSWSpace(_knn_metric, _numDims, _neighborhoodSize, _neighborhoodWeighting, _numPoints, _attribute_data, _numHistBins);
    assert(space != NULL);

    auto t_end_CreateHNSWSpace = std::chrono::steady_clock::now();
    qDebug() << "Distance calculation: Build time metric space (sec): " << ((float)std::chrono::duration_cast<std::chrono::milliseconds> (t_end_CreateHNSWSpace - t_start_CreateHNSWSpace).count()) / 1000;
    qDebug() << "Distance calculation: Compute kNN";

    // depending on the feature type, the features vector has a different length (scalar features vs vector features per dimension)
    size_t featureSize = SetFeatureSize(_featureType, _numDims, _numHistBins, _neighborhoodSize);

    auto t_start_ComputeDist = std::chrono::steady_clock::now();

    if (_knn_lib == knn_library::KNN_HNSW) {
        qDebug() << "Distance calculation: HNSWLib for knn computation";

        std::tie(_knn_indices, _knn_distances_squared) = ComputeHNSWkNN(_dataFeatures, space, featureSize, _numPoints, _nn);

    }
    else if (_knn_lib == knn_library::NONE) {
        qDebug() << "Distance calculation: Exact kNN computation";

        std::tie(_knn_indices, _knn_distances_squared) = ComputeExactKNN(_dataFeatures, space, featureSize, _numPoints, _nn);

    }
    else if (_knn_lib == knn_library::EVAL) {
        // Save the entire distance matrix to disk. Then calc the exact kNN and perform the embedding
        // Note: You could also sort the distance matrix instead of recalculating it - but I'm lazy and will only use this for small data set where the performance is not an issue.

        qDebug() << "Distance calculation: Evaluation mode - Full distance matrix";
        std::vector<int> knn_indices_to_Disk;
        std::vector<float> knn_distances_squared_to_Disk;
        std::tie(knn_indices_to_Disk, knn_distances_squared_to_Disk) = ComputeFullDistMat(_dataFeatures, space, featureSize, _numPoints);

        qDebug() << "Distance calculation: Evaluation mode - Write distance matrix to disk";

        // Write distance matrices to disk
        std::string savePath = "D:/Documents/Project 2020a/Spidr/Paper/SpidrEvaluation/Data/";
        savePath += _embeddingName;
        std::string infoStr = "_nD_" + std::to_string(_numDims) + "_nP_" + std::to_string(_numPoints) + "_nN_" + std::to_string(_numPoints);
        writeVecToBinary(knn_indices_to_Disk, savePath + "_knnInd" + infoStr + ".bin");
        writeVecToBinary(knn_distances_squared_to_Disk, savePath + "_knnDist" + infoStr + ".bin");
        
        qDebug() << "Distance calculation: Evaluation mode - Full distance matrix";
        std::tie(_knn_indices, _knn_distances_squared) = ComputeExactKNN(_dataFeatures, space, featureSize, _numPoints, _nn);

    }


    auto t_end_ComputeDist = std::chrono::steady_clock::now();
    qDebug() << "Distance calculation: Computation duration (sec): " << ((float)std::chrono::duration_cast<std::chrono::milliseconds> (t_end_ComputeDist - t_start_ComputeDist).count()) / 1000;

    // -1 would mark unset values
    assert(_knn_indices.size() == _numPoints * _nn);
    assert(_knn_distances_squared.size() == _numPoints * _nn);
    assert(std::none_of(_knn_indices.begin(), _knn_indices.end(), [](int i) {return i == -1; }));
    assert(std::none_of(_knn_distances_squared.begin(), _knn_distances_squared.end(), [](float i) {return i == -1.0f; }));

}

const std::tuple< std::vector<int>, std::vector<float>> DistanceCalculation::output() {
    return { _knn_indices, _knn_distances_squared };
}

std::vector<int>* DistanceCalculation::get_knn_indices() {
    return &_knn_indices;
}

std::vector<float>* DistanceCalculation::get_knn_distances_squared() {
    return &_knn_distances_squared;
}


void DistanceCalculation::setKnnAlgorithm(knn_library knn)
{
    _knn_lib = knn;
}

void DistanceCalculation::setDistanceMetric(distance_metric metric)
{
    _knn_metric = metric;
}