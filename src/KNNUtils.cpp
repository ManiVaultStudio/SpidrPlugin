#pragma once
#include "KNNUtils.h"

template<typename T>
std::tuple<std::vector<int>, std::vector<float>> ComputeHNSWkNN(const std::vector<T>* dataFeatures, hnswlib::SpaceInterface<float> *space, size_t indMultiplier, size_t numPoints, unsigned int nn) {

    std::vector<int> indices(numPoints * nn, -1);
    std::vector<float> distances_squared(numPoints * nn, -1);

    qDebug() << "ComputeHNSWkNN: Build akNN Index";

    hnswlib::HierarchicalNSW<float> appr_alg(space, numPoints);   // use default HNSW values for M, ef_construction random_seed

    // add data points: each data point holds _numDims*_numHistBins values
    appr_alg.addPoint((void*)dataFeatures->data(), (std::size_t) 0);

#ifdef NDEBUG
    // This loop is for release mode, it's parallel loop implementation from hnswlib
    int num_threads = std::thread::hardware_concurrency();
    hnswlib::ParallelFor(1, numPoints, num_threads, [&](size_t i, size_t threadId) {
        appr_alg.addPoint((void*)(dataFeatures->data() + (i*indMultiplier)), (hnswlib::labeltype) i);
    });
#else
// This loop is for debugging, when you want to sequentially add points
    for (int i = 1; i < numPoints; ++i)
    {
        appr_alg.addPoint((void*)(dataFeatures->data() + (i*indMultiplier)), (hnswlib::labeltype) i);
    }
#endif
    qDebug() << "ComputeHNSWkNN: Search akNN Index";

    // query dataset
#pragma omp parallel for
    for (int i = 0; i < numPoints; ++i)
    {
        // find nearest neighbors
        auto top_candidates = appr_alg.searchKnn((void*)(dataFeatures->data() + (i*indMultiplier)), (hnswlib::labeltype)nn);
        while (top_candidates.size() > nn) {
            top_candidates.pop();
        }
        // save nn in _knn_indices and _knn_distances_squared 
        auto *distances_offset = distances_squared.data() + (i*nn);
        auto indices_offset = indices.data() + (i*nn);
        int j = 0;
        while (top_candidates.size() > 0) {
            auto rez = top_candidates.top();
            distances_offset[nn - j - 1] = rez.first;
            indices_offset[nn - j - 1] = appr_alg.getExternalLabel(rez.second);
            top_candidates.pop();
            ++j;
        }
    }

    return std::make_tuple(indices, distances_squared);
}
// Resolve linker errors with explicit instantiation, https://isocpp.org/wiki/faq/templates#separate-template-fn-defn-from-decl
template std::tuple<std::vector<int>, std::vector<float>> ComputeHNSWkNN<float>(const std::vector<float>* dataFeatures, hnswlib::SpaceInterface<float> *space, size_t indMultiplier, size_t numPoints, unsigned int nn);
template std::tuple<std::vector<int>, std::vector<float>> ComputeHNSWkNN<unsigned int>(const std::vector<unsigned int>* dataFeatures, hnswlib::SpaceInterface<float> *space, size_t indMultiplier, size_t numPoints, unsigned int nn);


hnswlib::SpaceInterface<float>* CreateHNSWSpace(distance_metric knn_metric, size_t numDims, size_t neighborhoodSize, loc_Neigh_Weighting neighborhoodWeighting, size_t numPoints, std::vector<float>* attribute_data, size_t numHistBins) {
    // chose distance metric
    hnswlib::SpaceInterface<float> *space = NULL;
    if (knn_metric == distance_metric::METRIC_QF)
    {
        assert(numHistBins > 0);
        qDebug() << "Distance calculation: QFSpace as vector feature";
        space = new hnswlib::QFSpace(numDims, numHistBins);
    }
    else if (knn_metric == distance_metric::METRIC_EMD)
    {
        assert(numHistBins > 0);
        qDebug() << "Distance calculation: EMDSpace as vector feature";
        space = new hnswlib::EMDSpace(numDims, numHistBins);
    }
    else if (knn_metric == distance_metric::METRIC_HEL)
    {
        assert(numHistBins > 0);
        qDebug() << "Distance calculation: HellingerSpace as vector feature metric";
        space = new hnswlib::HellingerSpace(numDims, numHistBins);
    }
    else if (knn_metric == distance_metric::METRIC_EUC)
    {
        qDebug() << "Distance calculation: EuclidenSpace (L2Space) as scalar feature metric";
        space = new hnswlib::L2Space(numDims);
    }
    else if (knn_metric == distance_metric::METRIC_PCOL)
    {
        qDebug() << "Distance calculation: EuclidenSpace (PointCollectionSpace) as scalar feature metric";
        space = new hnswlib::PointCollectionSpace(numDims, neighborhoodSize, neighborhoodWeighting);
    }
    else
    {
        qDebug() << "Distance calculation: ERROR: Distance metric unknown.";
        return NULL;
    }

    return space;
}