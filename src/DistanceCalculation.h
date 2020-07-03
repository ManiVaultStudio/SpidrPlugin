#pragma once

#include "Utils.h"

#include <vector>
#include <QThread>
#include <QSize>

/**
* Calculate Spatial Features
* .start() will execute run() in a new thread
*/
class DistanceCalculation : public QThread
{
    Q_OBJECT
public:
    DistanceCalculation();
    ~DistanceCalculation(void) override;

    const std::vector<float>& output(); // tuple of indices and dists

    void setupData(std::vector<float>* histogramFeatures, knn_library knn_lib = knn_library::KNN_FLANN, knn_distance_metric knn_metric= knn_distance_metric::KNN_METRIC_QF);

private:
    void run() override;

    void computekNN();

    // functor for metrics

signals:

private:
    // Options
    knn_library _knn_lib;
    knn_distance_metric _knn_metric;

    // Data
    // Input
    std::vector<float>* _histogramFeatures;

    // Output
    std::vector<int> indices;
    std::vector<float> distances_squared;
};