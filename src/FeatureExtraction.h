#pragma once

#include "TsneData.h"

#include <vector>
#include <QThread>
#include <QSize>

/**
* Calculate Spatial Features
* .start() will execute run() in a new thread
*/
class FeatureExtraction : public QThread
{
    Q_OBJECT
public:
    FeatureExtraction();
    ~FeatureExtraction(void) override;

    const std::vector<float>& output();

    void setNeighborhoodSize(unsigned int size);    // TODO sets _numNeighbors
    void setNumHistBins(unsigned int size);
//    void setNumHistBins(heuristic heu);
    void setNeighborhoodWeighting(int weighting);   // TODO introduce enum for options

    /**
    * Setup feature extraction by introducing the data
    */
    void setupData(const std::vector<float>& data, const std::vector<unsigned int>& pointIds, const int numDimensions, QSize imgSize);

private:
    void run() override;

    /**
    * Calculates histgram features
    */
    void computeHistogramFeatures();

    /**
    *  Init, i.e. identify min and max per dimension for histogramming
    *  Sets _minMaxVals according to _inputData
    */
    void initExtraction();

    void extractFeatures();

    std::vector<unsigned int> neighborhoodIndices(unsigned int pointInd);

    void calculateHistogram(unsigned int pointInd, std::vector<float> neighborValues);

signals:

private:
    // Options 

    // Square neighborhood centered around an item with _neighborhoodSize left, right, top, buttom
    unsigned int _neighborhoodSize;
    // Number of neighbors including center
    unsigned int _numNeighbors;
    // Weightings of neighborhood kernel
    std::vector<float> _neighborhoodWeights;
    // Number of bins in each histogram
    unsigned int _numHistBins;
    // Extrema for each dimension/channel, i.e. [min_Ch0, max_Ch0, min_Ch1, max_Ch1, ...]
    std::vector<float> _minMaxVals;

    // Data

    // high dimensional data
    TsneData _inputData;
    // Histogram features for each item, i.e. in case of 1D histograms for each data point there are _inputData.getNumDimensions() histograms with _numHistBins values
    std::vector<float> _histogramFeatures;
    // Image Size
    QSize _imgSize;
    // Global IDs of points in data
    std::vector<unsigned int> _pointIds;

};