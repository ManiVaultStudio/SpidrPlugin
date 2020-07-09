#include "DistanceCalculation.h"

#include "SpidrPlugin.h"        // class Parameters
#include "hnswlib/hnswlib.h"

DistanceCalculation::DistanceCalculation() :
    _knn_lib(knn_library::KNN_HNSW),
    _knn_metric(knn_distance_metric::KNN_METRIC_QF)
{
}


DistanceCalculation::~DistanceCalculation()
{
}

void DistanceCalculation::setupData(std::vector<float>* histogramFeatures, Parameters& params) {
    // (Most) Options are set outside this funciton
    _nn = params._perplexity*params._perplexity_multiplier + 1;
    params._nn = _nn;

    // Data
    // Input
    _numPoints = params._numPoints;
    _numDims = params._numDims;
    _numHistBins = params._numHistBins;
    _histogramFeatures = histogramFeatures;

    // Output
    _indices.resize(_numPoints*_nn);
    _distances_squared.resize(_numPoints*_nn);

    assert(_histogramFeatures->size() == _numPoints * _numDims * _numHistBins);

    qDebug() << "Distance calculation: Num data points: " << _numPoints << " Feature values per point: " << _numDims * _numHistBins << "Number of NN to calculate" << _nn;

}

void DistanceCalculation::start() {
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
            qDebug() << "Distance calculation: QFSpace for metric definition";
            space = new hnswlib::QFSpace(_numDims, _numHistBins);
        }
        else if (_knn_metric == knn_distance_metric::KNN_METRIC_HEL)
        {
            qDebug() << "Distance calculation: HellingerSpace for metric definition";
            space = new hnswlib::HellingerSpace(_numDims, _numHistBins);
        }
        else
        {
            qDebug() << "Distance calculation: ERROR: Distance metric unknown. Using default metric: QFSpace.";
            space = new hnswlib::QFSpace(_numDims, _numHistBins);
        }

        hnswlib::HierarchicalNSW<float> appr_alg(space, _numPoints);   // use default HNSW values for M, ef_construction random_seed

        // add data points: each data point holds _numDims*_numHistBins values
        appr_alg.addPoint((void*)_histogramFeatures->data(), (std::size_t) 0);
//#pragma omp parallel for
        for (int i = 1; i < _numPoints; ++i)
        {
            appr_alg.addPoint((void*)(_histogramFeatures->data() + (i*_numDims*_numHistBins)), (hnswlib::labeltype) i);
        }

        // query dataset
#pragma omp parallel for
        for (int i = 0; i < _numPoints; ++i)
        {
            // find nearest neighbors
            auto top_candidates = appr_alg.searchKnn((void*)(_histogramFeatures->data() + (i*_numDims*_numHistBins)), (hnswlib::labeltype)_nn);
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


void DistanceCalculation::setKnnAlgorithm(int index)
{
    // index corresponds to order in which algorithm were added to widget
    switch (index)
    {
    case 0: _knn_lib = knn_library::KNN_HNSW; break;
    default: _knn_lib = knn_library::KNN_HNSW;
    }
}

void DistanceCalculation::setDistanceMetric(int index)
{
    // index corresponds to order in which algorithm were added to widget
    switch (index)
    {
    case 0: _knn_metric = knn_distance_metric::KNN_METRIC_QF; break;
    //case 1: _knn_metric = knn_distance_metric::KNN_METRIC_EMD; break;
    case 1: _knn_metric = knn_distance_metric::KNN_METRIC_HEL; break;
    default: _knn_metric = knn_distance_metric::KNN_METRIC_QF;
    }
}