#include "FeatureExtraction.h"

#include <algorithm>    // std::fill

FeatureExtraction::FeatureExtraction() :
    _neighborhoodSize(1),
    _numHistBins(5)
{
    // square neighborhood
    _numNeighbors = ((_neighborhoodSize * 2) + 1) * ((_neighborhoodSize * 2) + 1);
    // uniform weighting
    std::fill(_neighborhoodWeights.begin(), _neighborhoodWeights.end(), 1);
}

FeatureExtraction::~FeatureExtraction() 
{
}

void FeatureExtraction::run() {

}

void FeatureExtraction::computeHistogramFeatures() {
    
}

const std::vector<float>& FeatureExtraction::output()
{
    return _histogramFeatures;
}
