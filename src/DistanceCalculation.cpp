#include "DistanceCalculation.h"

#include <flann/flann.hpp>


DistanceCalculation::DistanceCalculation() 
{
}


DistanceCalculation::~DistanceCalculation()
{
}

void DistanceCalculation::setupData(std::vector<float>* histogramFeatures, knn_library knn_lib = knn_library::KNN_FLANN, knn_distance_metric knn_metric = knn_distance_metric::KNN_METRIC_QF) {

}

void DistanceCalculation::run() {
    //computekNN();
}

void DistanceCalculation::computekNN() {

    if (_knn_lib == knn_library::KNN_FLANN) {
        flann::Matrix<float> dataset(high_dimensional_data, _numPoints, _numDims);
        flann::Matrix<float> query(high_dimensional_data, _numPoints, _numDims);

        flann::Index<flann::L2<float> > index(dataset, flann::KDTreeIndexParams(params._num_trees), distance);
        const unsigned int nn = params._perplexity*params._perplexity_multiplier + 1;
        distances_squared.resize(_numPoints*nn);
        indices.resize(_numPoints*nn);

        index.buildIndex();

        flann::Matrix<int> indices_mat(indices.data(), query.rows, nn);
        flann::Matrix<float> dists_mat(distances_squared.data(), query.rows, nn);
        flann::SearchParams flann_params(params._num_checks);
        flann_params.cores = 0; //all cores
        index.knnSearch(query, indices_mat, dists_mat, nn, flann_params);

    }
}

const std::vector<float>& DistanceCalculation::output() {

}

