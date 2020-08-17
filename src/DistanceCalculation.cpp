#include "DistanceCalculation.h"

#include "AnalysisParameters.h"
#include "KNNUtils.h"

#include "hnswlib/hnswlib.h"

#include <QDebug>

#include <chrono>

DistanceCalculation::DistanceCalculation() :
    _knn_lib(knn_library::KNN_HNSW),
    _knn_metric(knn_distance_metric::KNN_METRIC_QF)
{
}


DistanceCalculation::~DistanceCalculation()
{
}

void DistanceCalculation::setup(std::vector<float>* histogramFeatures, Parameters& params) {
    _featureType = params._featureType;

    // Parameters
    _knn_lib = params._aknn_algorithm;
    _knn_metric = params._aknn_metric;
    _nn = params._nn;
    _neighborhoodSize = (2 * (params._numLocNeighbors) + 1) * (2 * (params._numLocNeighbors) + 1); // square neighborhood with _numLocNeighbors to each side from the center

    // Data
    // Input
    _numPoints = params._numPoints;
    _numDims = params._numDims;
    _numHistBins = params._numHistBins;
    _dataFeatures = histogramFeatures;

    // Output
    _indices.resize(_numPoints*_nn);
    _distances_squared.resize(_numPoints*_nn);

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

}

void DistanceCalculation::compute() {
    qDebug() << "Distance calculation: started";

    computekNN();

    qDebug() << "Distance calculation: finished";

}

void DistanceCalculation::computekNN() {
    
    if (_knn_lib == knn_library::KNN_HNSW) {
        qDebug() << "Distance calculation: HNSWLib for knn computation";

        // setup hsnw index
        hnswlib::SpaceInterface<float> *space = NULL;
        if (_knn_metric == knn_distance_metric::KNN_METRIC_QF)
        {
            qDebug() << "Distance calculation: QFSpace as vector feature";
            space = new hnswlib::QFSpace(_numDims, _numHistBins);
        }
        else if (_knn_metric == knn_distance_metric::KNN_METRIC_HEL)
        {
            qDebug() << "Distance calculation: HellingerSpace as vector feature metric";
            space = new hnswlib::HellingerSpace(_numDims, _numHistBins);
        }
        else if (_knn_metric == knn_distance_metric::KNN_METRIC_EUC)
        {
            qDebug() << "Distance calculation: EuclidenSpace as scalar feature metric";
            space = new hnswlib::L2Space(_numDims);
        }
        else if (_knn_metric == knn_distance_metric::KNN_METRIC_PCOL)
        {
            qDebug() << "Distance calculation: EuclidenSpace as scalar feature metric";
            space = new hnswlib::PointCollectionSpace(_numDims, _neighborhoodSize);
        }
        else
        {
            qDebug() << "Distance calculation: ERROR: Distance metric unknown.";
            return;
        }

        qDebug() << "Distance calculation: Build akNN Index";

        hnswlib::HierarchicalNSW<float> appr_alg(space, _numPoints);   // use default HNSW values for M, ef_construction random_seed

        int num_threads = std::thread::hardware_concurrency();

        auto start = std::chrono::steady_clock::now();

        // depending on the feature type, the features vector has a different length (scalar features vs vector features per dimension)
        size_t indMultiplier = 0;
        switch (_featureType) {
        case feature_type::TEXTURE_HIST_1D: indMultiplier = _numDims * _numHistBins; break;
        case feature_type::LISA:            // same as Geary's C
        case feature_type::GEARYC:          indMultiplier = _numDims; break;
        case feature_type::PCOL:            indMultiplier = _numDims * _neighborhoodSize; break;
        }
        
        // add data points: each data point holds _numDims*_numHistBins values
        appr_alg.addPoint((void*)_dataFeatures->data(), (std::size_t) 0);

        // This loop is for debugging, when you want to sequentially add points
        //for (int i = 1; i < _numPoints; ++i)
        //{
        //    appr_alg.addPoint((void*)(_dataFeatures->data() + (i*indMultiplier)), (hnswlib::labeltype) i);
        //}

        hnswlib::ParallelFor(0, _numPoints, num_threads, [&](size_t i, size_t threadId) {
            appr_alg.addPoint((void*)(_dataFeatures->data() + (i*indMultiplier)), (hnswlib::labeltype) i);
        });
        //appr_alg.checkIntegrity();

        auto end = std::chrono::steady_clock::now();
        qDebug() << "Distance calculation: Build duration (sec): " << ((float) std::chrono::duration_cast<std::chrono::milliseconds> (end - start).count()) / 1000;

        qDebug() << "Distance calculation: Search akNN Index";

        // query dataset
#pragma omp parallel for
        for (int i = 0; i < _numPoints; ++i)
        {
            // find nearest neighbors
            auto top_candidates = appr_alg.searchKnn((void*)(_dataFeatures->data() + (i*indMultiplier)), (hnswlib::labeltype)_nn);
            while (top_candidates.size() > _nn) {
                top_candidates.pop();
            }
            // save nn in _indices and _distances_squared 
            auto *distances_offset = _distances_squared.data() + (i*_nn);
            auto indices_offset = _indices.data() + (i*_nn);
            int j = 0;
            while (top_candidates.size() > 0) {
                auto rez = top_candidates.top();
                distances_offset[_nn - j - 1] = rez.first;
                indices_offset[_nn - j - 1] = appr_alg.getExternalLabel(rez.second);
                top_candidates.pop();
                ++j;
            }
        }
    }

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