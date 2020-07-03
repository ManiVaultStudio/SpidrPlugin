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

        assert(neighborIDs.size() == _numNeighbors);

        // get data for all neighborhood points
        // Padding: if neighbor is outside selection, assign 0 to all dimension values
        std::vector<float> neighborValues;
        neighborValues.resize(_numNeighbors * _numDims);
        for (unsigned int neighbor = 0; neighbor < _numNeighbors; neighbor++) {
            if (neighborIDs[neighbor] == -1)
            for (unsigned int dim = 0; dim < _numDims; dim++) {
                neighborValues[neighbor * _numDims + dim] = (neighborIDs[neighbor] != -1) ? _data[neighborIDs[neighbor] * _numDims + dim] : 0;
            }
        }

        // calculate histograms, save histos in _histogramFeatures TODO
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
    // Attantion: _minMaxVals[1] value will be placed in an overflow bin. Either change the last bin or collaps the overflow bin into the last bin
    // once this works, check if the following is faster (VS Studio will complain but compile)
    //const auto axi = axis::regular(10, 0.0, 1.0);
    //auto h1 = make_histogram_with(std::vector<int>(), axi);

    // 1D histograms for each dimension
    // save the histogram in _histogramFeatures
    for (unsigned int dim = 0; dim < _numDims; dim++) {
        auto h = boost::histogram::make_histogram(boost::histogram::axis::regular(_numHistBins, _minMaxVals[dim], _minMaxVals[dim + 1]));
        for (unsigned int neighbor = 0; neighbor < _numNeighbors; neighbor++) {
            h(neighborValues[neighbor * _numDims + dim], _neighborhoodWeights[neighbor]);
        }

        assert(h.rank() == 1); // 1D hist
        assert(h.axis().size() == _numHistBins);

        for (unsigned int bin = 0; bin < _numHistBins; bin++) {
            _histogramFeatures[pointInd * _numDims * _numHistBins + dim * _numDims + bin] = h.at(bin);
        }
        _histogramFeatures[pointInd * _numDims * _numHistBins + dim * _numDims + _numHistBins] = h.at(_numHistBins + 1); // _minMaxVals[dim + 1] is saved in overflow bin
    }

}

const std::vector<float>& FeatureExtraction::output()
{
    return _histogramFeatures;
}
