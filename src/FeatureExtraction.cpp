#include "FeatureExtraction.h"

#include <iterator>     // std::advance
#include <QDebug>       // qDebug
#include <algorithm>    // std::for_each, std::fill
#include <execution>    // std::execution::par_unseq
#include <vector>       // std::vector, std::begin, std::end

// Boost might be more useful for higher dimensional histograms
// but it's convinient for now
#include <boost/histogram.hpp>

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
    //computeHistogramFeatures();
}

void FeatureExtraction::setupData(const std::vector<float>& data, const std::vector<unsigned int>& pointIds, const int numDimensions, QSize imgSize) {
    unsigned int numPoints = pointIds.size();
    _pointIds = pointIds;
    _imgSize = imgSize;
    _inputData.assign(numPoints, numDimensions, data);
    qDebug() << "Variables set. Num dims: " << numDimensions << " Num data points: " << numPoints << " Image size (width, height): " << imgSize.width() << ", " << imgSize.height();
    qDebug() << "feature extraction dataassigned.";
}

void FeatureExtraction::computeHistogramFeatures() {
    // init, i.e. identify min and max per dimension for histogramming
    initExtraction();

    // convolution over all points to create histograms
    extractFeatures();
}

void FeatureExtraction::initExtraction() {
    // Init, a.o. find min and max for each channel

    unsigned int numDims = _inputData.getNumDimensions();
    unsigned int numPoints = _inputData.getNumPoints();

    _minMaxVals.resize(2 * numDims, 0);

    // for each dimension iterate over all values
    for (unsigned int dimCount = 0; dimCount < numDims; dimCount++) {
        // set data iterator to dimension
        std::vector<float>::iterator dataIt = _inputData.getDataNonConst().begin();
        std::advance(dataIt, dimCount);
        // init min and max
        _minMaxVals.at(2 * dimCount) = *dataIt;
        _minMaxVals.at(2 * dimCount + 1) = *dataIt;

        for (unsigned int pointCount = 0; pointCount < numPoints; pointCount++) {
            // min
            if (*dataIt < _minMaxVals.at(2 * dimCount))
                _minMaxVals.at(2 * dimCount) = *dataIt;
            // max
            if (*dataIt > _minMaxVals.at(2 * dimCount + 1))
                _minMaxVals.at(2 * dimCount + 1) = *dataIt;
            // step forward to next point
            std::advance(dataIt, numDims);
        }
    }

}

void FeatureExtraction::extractFeatures() {
    _histogramFeatures.resize(_inputData.getNumPoints() * _inputData.getNumDimensions() * _numHistBins);

    unsigned int numDims = _inputData.getNumDimensions();
    unsigned int numPoints = _inputData.getNumPoints();
    std::vector<float> data = _inputData.getDataNonConst(); // there must be a better way to hand this to the par for_each
    std::vector<float>::iterator dataIt = _inputData.getDataNonConst().begin();

    // convolve over all selected data points
    std::for_each(std::execution::par_unseq, std::begin(_pointIds), std::end(_pointIds), [this](int pointID) {
        // get neighborhood of the current point
        std::vector<unsigned int> neighborIDs = neighborhoodIndices(pointID);

        // get data for neighborhood points
        std::vector<float> neighborValues;

        // calculate histograms
        calculateHistogram(pointID, neighborValues);
    });


}

// For now, expect a rectangle selection (lasso selection might cause edge cases that were not thought of)
// Use edge padding, i.e. repeat the value of the closest item covered by the kernel
std::vector<unsigned int> FeatureExtraction::neighborhoodIndices(unsigned int pointInd) {
    // TODO: implements this
    std::vector<unsigned int> neighborsIDs(_numNeighbors, 0);
    return neighborsIDs;
}

void FeatureExtraction::calculateHistogram(unsigned int pointInd, std::vector<float> neighborValues) {
    // TODO: set _histogramFeatures
}

const std::vector<float>& FeatureExtraction::output()
{
    return _histogramFeatures;
}
