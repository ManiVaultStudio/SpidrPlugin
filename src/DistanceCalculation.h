#pragma once

#include "KNNUtils.h"

#include <tuple>
#include <vector>

#include <QThread>
#include <QSize>

class Parameters;

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

    const std::tuple< std::vector<int>, std::vector<float>> output(); // tuple of indices and dists
    std::vector<int>* get_knn_indices();
    std::vector<float>* get_knn_distances_squared();

    void setKnnAlgorithm(int index);
    void setDistanceMetric(int index);

    void setupData(std::vector<float>* histogramFeatures, Parameters& params);

private:
    void run() override;

    void computekNN();

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