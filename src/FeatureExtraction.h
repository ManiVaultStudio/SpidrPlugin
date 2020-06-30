#pragma once


#include <vector>
#include <QThread>

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

private:
    void run() override;

    /**
    * Calculates histgram features
    */
    void computeHistogramFeatures();

signals:

private:
    std::vector<float> _histogramFeatures;
    // Square neighborhood centered around an item with _neighborhoodSize left, right, top, buttom
    unsigned int _neighborhoodSize;
    // Number of neighbors including center
    unsigned int _numNeighbors;
    std::vector<float> _neighborhoodWeights;
    unsigned int _numHistBins;
};