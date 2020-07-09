#include "FeatureExtraction.h"

#include "KNNUtils.h"
#include "SpidrPlugin.h"    // class Parameters

#include "omp.h"

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
    _neighborhoodWeights.resize(_numNeighbors);
    std::fill(_neighborhoodWeights.begin(), _neighborhoodWeights.end(), 1);
}


FeatureExtraction::~FeatureExtraction()
{
}

void FeatureExtraction::start() {
    qDebug() << "Feature extraction started.";

    computeHistogramFeatures();

    qDebug() << "Feature extraction finished.";

}

void FeatureExtraction::setupData(QSize imgSize, const std::vector<unsigned int>& pointIds, const std::vector<float>& data, Parameters& params) {
    // Options are set outside this function
    params._numHistBins = _numHistBins;

    // Data
    // Input
    _imgSize = imgSize;
    _pointIds = pointIds;
    _numPoints = pointIds.size();
    _numDims = params._numDims;
    _attribute_data = data;

    // Output
    _histogramFeatures.resize(_numPoints * _numDims * _numHistBins);
    std::fill(_histogramFeatures.begin(), _histogramFeatures.end(), -1);

    assert(_attribute_data.size() == _numPoints * _numDims);

    qDebug() << "Feature Extraction. Num Bins: " << _numHistBins;
}

void FeatureExtraction::computeHistogramFeatures() {
    // init, i.e. identify min and max per dimension for histogramming
    initExtraction();

    // convolution over all points to create histograms
    extractFeatures();

    // if there is a -1 in the features, this value was not set at all
    assert(std::find(_histogramFeatures.begin(), _histogramFeatures.end(), -1) == _histogramFeatures.end());
}

void FeatureExtraction::initExtraction() {
    // Init
    // a.o.: find min and max for each channel
    _minMaxVals.resize(2 * _numDims, 0);

    // for each dimension iterate over all values
    // remember data stucture (point1 d0, point1 d1,... point1 dn, point2 d0, point2 d1, ...)
    for (unsigned int dimCount = 0; dimCount < _numDims; dimCount++) {
        // init min and max
        float currentVal = _attribute_data.at(dimCount);
        _minMaxVals.at(2 * dimCount) = currentVal;
        _minMaxVals.at(2 * dimCount + 1) = currentVal;

        for (unsigned int pointCount = 0; pointCount < _numPoints; pointCount++) {
            currentVal = _attribute_data.at(pointCount * _numDims + dimCount);
            // min
            if (currentVal < _minMaxVals.at(2 * dimCount))
                _minMaxVals.at(2 * dimCount) = currentVal;
            // max
            else if (currentVal > _minMaxVals.at(2 * dimCount + 1))
                _minMaxVals.at(2 * dimCount + 1) = currentVal;
        }
    }

}

void FeatureExtraction::extractFeatures() {
    
    // convolve over all selected data points
//    #pragma omp parallel for 
    for (int pointID = 0; pointID < _pointIds.size(); pointID++) {
        // get neighborhood of the current point
        std::vector<int> neighborIDs = neighborhoodIndices(_pointIds.at(pointID));

        assert(neighborIDs.size() == _numNeighbors);

        // get data for all neighborhood points
        // Padding: if neighbor is outside selection, assign 0 to all dimension values
        std::vector<float> neighborValues;
        neighborValues.resize(_numNeighbors * _numDims);
        for (unsigned int neighbor = 0; neighbor < _numNeighbors; neighbor++) {
            for (unsigned int dim = 0; dim < _numDims; dim++) {
                neighborValues[neighbor * _numDims + dim] = (neighborIDs[neighbor] != -1) ? _attribute_data[neighborIDs[neighbor] * _numDims + dim] : 0;
            }
        }

        // calculate histograms, save histos in _histogramFeatures
        calculateHistogram(_pointIds.at(pointID), neighborValues);
    }

}

// For now, expect a rectangle selection (lasso selection might cause edge cases that were not thought of)
// Padding: assign -1 to points outside the selection. Later assign 0 vector to all of them.
std::vector<int> FeatureExtraction::neighborhoodIndices(unsigned int pointInd) {
    std::vector<int> neighborsIDs(_numNeighbors, -1);
    int imWidth = _imgSize.width();
    int rowID = int(pointInd / imWidth);

    // left and right neighbors
    std::vector<int> lrNeighIDs(2 * _neighborhoodSize + 1, 0);
    std::iota(lrNeighIDs.begin(), lrNeighIDs.end(), pointInd - _neighborhoodSize);

    // are left and right out of the picture?
    for (int& n : lrNeighIDs) {
        if (n < rowID * imWidth)
            n = -1;
        else if (n >= (rowID + 1) * imWidth)
            n = -1;
    }

    // above and below neighbors
    unsigned int localNeighCount = 0;
    for (int i = -1 * _neighborhoodSize; i <= (int)_neighborhoodSize; i++) {
        for (int ID : lrNeighIDs) {
            neighborsIDs[localNeighCount] = ID != -1 ? ID + i * _imgSize.width() : -1;  // if left or right is already out of image, above and below will be as well
            localNeighCount++;
        }
    }

    // Check if neighborhood IDs are in selected points
    for (int& ID : neighborsIDs) {
        // if neighbor is not in neighborhood, assign -1
        if (std::find(_pointIds.begin(), _pointIds.end(), ID) == _pointIds.end()) {
            ID = -1;
        }
    }

    return neighborsIDs;
}

void FeatureExtraction::calculateHistogram(unsigned int pointInd, std::vector<float> neighborValues) {

    // 1D histograms for each dimension
    // save the histogram in _histogramFeatures
    for (unsigned int dim = 0; dim < _numDims; dim++) {
        auto h = boost::histogram::make_histogram(boost::histogram::axis::regular(_numHistBins, _minMaxVals[2 * dim], _minMaxVals[2 * dim + 1]));
        // once this works, check if the following is faster (VS Studio will complain but compile)
        //auto h = boost::histogram::make_histogram_with(std::vector<float>(), boost::histogram::axis::regular(_numHistBins, _minMaxVals[dim], _minMaxVals[dim + 1]));
        for (unsigned int neighbor = 0; neighbor < _numNeighbors; neighbor++) {
            h(neighborValues[neighbor * _numDims + dim], boost::histogram::weight(_neighborhoodWeights[neighbor]));
        }

        assert(h.rank() == 1); // 1D hist
        assert(h.axis().size() == _numHistBins);

        for (unsigned int bin = 0; bin < _numHistBins; bin++) {
            _histogramFeatures[pointInd * _numDims * _numHistBins + dim * _numHistBins + bin] = h.at(bin);
        }
        // the max value is stored in the overflow bin
        if (h.at(_numHistBins) != 0)
            _histogramFeatures[pointInd * _numDims * _numHistBins + dim * _numHistBins + _numHistBins - 1] += h.at(_numHistBins);

    }

}

std::vector<float>* FeatureExtraction::output()
{
    return &_histogramFeatures;
}
