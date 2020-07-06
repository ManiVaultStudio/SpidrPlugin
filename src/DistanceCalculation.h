#pragma once

#include "KNNUtils.h"

#include <tuple>
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

    const std::tuple< std::vector<int>, std::vector<float>>& output(); // tuple of indices and dists

    void setupData(std::vector<float>* histogramFeatures, Parameters& params);

private:
    void run() override;

    void computekNN();

    // functor for metrics

signals:

private:
    // Options
    knn_library _knn_lib;
    knn_distance_metric _knn_metric;
    unsigned int _nn;

    // Data
    // Input
    unsigned int _numDims;
    unsigned int _numPoints;
    unsigned int _numHistBins;
    std::vector<float>* _histogramFeatures;

    // Output
    std::vector<int> _indices;
    std::vector<float> _distances_squared;
};