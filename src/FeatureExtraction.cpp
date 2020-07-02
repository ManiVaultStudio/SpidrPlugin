#include "FeatureExtraction.h"

#include <QDebug>       // qDebug
#include <iterator>     // std::advance
#include <algorithm>    // std::for_each, std::fill, std::find
#include <execution>    // std::execution::par_unseq
#include <vector>       // std::vector, std::begin, std::end
#include <array>        // std::array
#include <numeric>      // std::iota
#include <utility>      // std::forward

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

void FeatureExtraction::setupData(QSize imgSize, const std::vector<unsigned int>& pointIds, const int numDimensions, const std::vector<float>& data) {
    _imgSize = imgSize;
    _pointIds = pointIds;
    _numPoints = pointIds.size();
    _numDims = numDimensions;
    _data = data;

    assert(_data.size() == _numPoints * _numDims);

    qDebug() << "Variables set. Num dims: " << numDimensions << " Num data points: " << pointIds.size() << " Image size (width, height): " << imgSize.width() << ", " << imgSize.height();
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

    _minMaxVals.resize(2 * _numDims, 0);

    // for each dimension iterate over all values
    for (unsigned int dimCount = 0; dimCount < _numDims; dimCount++) {
        // set data iterator to dimension
        std::vector<float>::iterator dataIt = _data.begin();
        std::advance(dataIt, dimCount);
        // init min and max
        _minMaxVals.at(2 * dimCount) = *dataIt;
        _minMaxVals.at(2 * dimCount + 1) = *dataIt;

        for (unsigned int pointCount = 0; pointCount < _numPoints; pointCount++) {
            // min
            if (*dataIt < _minMaxVals.at(2 * dimCount))
                _minMaxVals.at(2 * dimCount) = *dataIt;
            // max
            if (*dataIt > _minMaxVals.at(2 * dimCount + 1))
                _minMaxVals.at(2 * dimCount + 1) = *dataIt;
            // step forward to next point
            std::advance(dataIt, _numDims);
        }
    }

}

void FeatureExtraction::extractFeatures() {
    _histogramFeatures.resize(_numPoints * _numDims * _numHistBins);

    // convolve over all selected data points
    std::for_each(std::execution::par_unseq, std::begin(_pointIds), std::end(_pointIds), [this](int pointID) {
        // get neighborhood of the current point
        std::vector<int> neighborIDs = neighborhoodIndices(pointID);

        // get data for neighborhood points, TODO
        std::vector<float> neighborValues;
        for (unsigned int neighborID : neighborIDs) {

        }

        // calculate histograms, TODO
        calculateHistogram(pointID, neighborValues);
    });

}

// For now, expect a rectangle selection (lasso selection might cause edge cases that were not thought of)
// Padding: assign -1 to points outside the selection. Later assign 0 vector to all of them.
std::vector<int> FeatureExtraction::neighborhoodIndices(unsigned int pointInd) {
    std::vector<int> neighborsIDs(_numNeighbors, 0);

    // left and right neighbors
    std::vector<int> lrNeighIDs(2*_neighborhoodSize + 1, 0);
    std::iota(lrNeighIDs.begin(), lrNeighIDs.end(), pointInd-_neighborhoodSize);

    // above and below neighbors
    unsigned int localNeighCount = 0;
    for (int i = -1 * _neighborhoodSize; i <= _neighborhoodSize; i++) {
        for (int ID : lrNeighIDs) {
            neighborsIDs[localNeighCount] = ID + i * _imgSize.width();
            localNeighCount++;
        }
    }

    // Check if neighborhood IDs are in selected points
    for (int& ID : neighborsIDs) {
        if (std::find(_pointIds.begin(), _pointIds.end(), ID) == _pointIds.end()) {
            ID = -1;
        }
    }

    return neighborsIDs;
}

void FeatureExtraction::calculateHistogram(unsigned int pointInd, std::vector<float> neighborValues) {
    // TODO: set _histogramFeatures
    using namespace boost::histogram; // strip the boost::histogram prefix
    auto h = make_histogram(axis::regular<>(6, -1.0, 2.0, "x"));

    auto axis = boost::histogram::axis::regular<>(_numHistBins, _minMaxVals[0], _minMaxVals[1]);
    auto h0 = boost::histogram::make_histogram_with(std::vector<int>(), axis);
    auto h1 = boost::histogram::make_histogram_with(std::vector<int>(), std::forward<boost::histogram::axis::regular<>>(axis), std::forward<boost::histogram::axis::regular<>>(axis));

    // hist(..., weight(w))
}

const std::vector<float>& FeatureExtraction::output()
{
    return _histogramFeatures;
}
