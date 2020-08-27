#include "DistanceCalculation.h"

#include "AnalysisParameters.h"
#include "KNNUtils.h"

#include "hnswlib/hnswlib.h"

#include <QDebug>

#include <chrono>
#include <algorithm>    // std::none_of

DistanceCalculation::DistanceCalculation() :
    _knn_lib(knn_library::KNN_HNSW),
    _knn_metric(knn_distance_metric::KNN_METRIC_QF)
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

    // Output
    //_indices.resize(_numPoints*_nn, -1);              // unnecessary, done in ComputekNN
    //_distances_squared.resize(_numPoints*_nn, -1);    // unnecessary, done in ComputekNN

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
    else if (_featureType == feature_type::PCOLappr) {
        assert(_dataFeatures->size() == (_numPoints * _neighborhoodSize));
        qDebug() << "Distance calculation: Feature values per point: " << _neighborhoodSize << "Number of NN to calculate" << _nn << ". Metric: " << (size_t)_knn_metric;
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
    
    if (_knn_lib == knn_library::KNN_HNSW) {
        qDebug() << "Distance calculation: HNSWLib for knn computation";

        auto start = std::chrono::steady_clock::now();

        qDebug() << "Distance calculation: Setting up metric space";

        // setup hsnw index
        hnswlib::SpaceInterface<float> *space = NULL;
        if (_knn_metric == knn_distance_metric::KNN_METRIC_QF)
        {
            qDebug() << "Distance calculation: QFSpace as vector feature";
            space = new hnswlib::QFSpace(_numDims, _numHistBins);
        }
        else if (_knn_metric == knn_distance_metric::KNN_METRIC_EMD)
        {
            qDebug() << "Distance calculation: EMDSpace as vector feature";
            space = new hnswlib::EMDSpace(_numDims, _numHistBins);
        }
        else if (_knn_metric == knn_distance_metric::KNN_METRIC_HEL)
        {
            qDebug() << "Distance calculation: HellingerSpace as vector feature metric";
            space = new hnswlib::HellingerSpace(_numDims, _numHistBins);
        }
        else if (_knn_metric == knn_distance_metric::KNN_METRIC_EUC)
        {
            qDebug() << "Distance calculation: EuclidenSpace (L2Space) as scalar feature metric";
            space = new hnswlib::L2Space(_numDims);
        }
        else if (_knn_metric == knn_distance_metric::KNN_METRIC_PCOL)
        {
            qDebug() << "Distance calculation: EuclidenSpace (PointCollectionSpace) as scalar feature metric";
            space = new hnswlib::PointCollectionSpace(_numDims, _neighborhoodSize, _neighborhoodWeighting);
        }
        else if (_knn_metric == knn_distance_metric::KNN_METRIC_PCOLappr)
        {
            qDebug() << "Distance calculation: EuclidenSpace (PointCollectionSpaceApprox) as scalar feature metric";
            space = new hnswlib::PointCollectionSpaceApprox(_numDims, _neighborhoodSize, _numPoints, _attribute_data, _neighborhoodWeighting);
        }
        else
        {
            qDebug() << "Distance calculation: ERROR: Distance metric unknown.";
            return;
        }

        // depending on the feature type, the features vector has a different length (scalar features vs vector features per dimension)
        size_t featureSize = 0;
        switch (_featureType) {
        case feature_type::TEXTURE_HIST_1D: featureSize = _numDims * _numHistBins; break;
        case feature_type::LISA:            // same as Geary's C
        case feature_type::GEARYC:          featureSize = _numDims; break;
        case feature_type::PCOL:            featureSize = _numDims * _neighborhoodSize; break;
        case feature_type::PCOLappr:        featureSize = _neighborhoodSize; break;
        }
        
        qDebug() << "Distance calculation: Compute kNN";
        std::tie(_indices, _distances_squared) = ComputekNN(_dataFeatures, space, featureSize, _numPoints, _nn);

        auto t = std::find(_indices.begin(), _indices.end(), -1);

        //qDebug() << *t;

        // -1 would mark unset values
        assert(std::none_of(_indices.begin(), _indices.end(), [](int i) {return i == -1; }));
        assert(std::none_of(_distances_squared.begin(), _distances_squared.end(), [](float i) {return i == -1.0f; }));

        auto end = std::chrono::steady_clock::now();
        qDebug() << "Distance calculation: Knn build and search index duration (sec): " << ((float)std::chrono::duration_cast<std::chrono::milliseconds> (end - start).count()) / 1000;

    }
    // else if exact solution is wanted: calculate full distance table and hand that over to the t-sne calculation
}

const std::tuple< std::vector<int>, std::vector<float>> DistanceCalculation::output() {
    return { _indices, _distances_squared };
}

std::vector<int>* DistanceCalculation::get_knn_indices() {
    return &_indices;
}

std::vector<float>* DistanceCalculation::get_knn_distances_squared() {
    return &_distances_squared;
}


void DistanceCalculation::setKnnAlgorithm(knn_library knn)
{
    _knn_lib = knn;
}

void DistanceCalculation::setDistanceMetric(knn_distance_metric metric)
{
    _knn_metric = metric;
}