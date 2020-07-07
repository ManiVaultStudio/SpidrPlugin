#include "DistanceCalculation.h"

#include "hnswlib/hnswlib.h"

DistanceCalculation::DistanceCalculation() 
{
}


DistanceCalculation::~DistanceCalculation()
{
}

void DistanceCalculation::setupData(std::vector<float>* histogramFeatures, Parameters& params) {
    
    _knn_lib = params._aknn_algorithm;
    _knn_metric = params._aknn_metric;
    _histogramFeatures = histogramFeatures;

    // resize
    _nn = params._perplexity*params._perplexity_multiplier + 1;
    _indices.resize(_numPoints*_nn);
    _distances_squared.resize(_numPoints*_nn);

    //TODO 
//     _numHistBins = 
}

void DistanceCalculation::run() {
    //computekNN();
}

void DistanceCalculation::computekNN() {
    

    if (_knn_lib == knn_library::KNN_HSNW) {

        // setup hsnw index
        hnswlib::SpaceInterface<float> *space = NULL;
        space = new hnswlib::QFSpace(_numDims, _numHistBins);
        hnswlib::HierarchicalNSW<float> appr_alg(space, _numPoints);   // use default values for M, ef_construction random_seed

        // add data points: each data point holds _numDims*_numHistBins values
        appr_alg.addPoint((void*)_histogramFeatures, (std::size_t) 0);
#pragma omp parallel for
        for (int i = 1; i < _numPoints; ++i)
        {
            appr_alg.addPoint((void*)(_histogramFeatures + (i*(_numDims*_numHistBins))), (hnswlib::labeltype) i);
        }

        // query dataset
#pragma omp parallel for
        for (int i = 0; i < _numPoints; ++i)
        {
            // find nearest neighbors
            auto top_candidates = appr_alg.searchKnn(_histogramFeatures + (i*(_numDims*_numHistBins)), (hnswlib::labeltype)_nn);
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

const std::tuple< std::vector<int>, std::vector<float>>& DistanceCalculation::output() {
    return { _indices, _distances_squared };
}

