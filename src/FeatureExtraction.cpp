#include "FeatureExtraction.h"

#include "KNNUtils.h"
#include "FeatureUtils.h"
#include "AnalysisParameters.h"     // class Parameters

#include "omp.h"

#include <QDebug>       // qDebug
#include <iterator>     // std::advance
#include <algorithm>    // std::fill, std::find, std::swap_ranges
#include <execution>    // std::execution::par_unseq
#include <vector>       // std::vector, std::begin, std::end
#include <array>        // std::array
#include <numeric>      // std::iota
#include <utility>      // std::forward
#include <chrono>       // std::chrono

// Boost might be more useful for higher dimensional histograms
// but it's convinient for now
#include <boost/histogram.hpp>

FeatureExtraction::FeatureExtraction() :
    _neighborhoodSize(1),
    _numHistBins(5),
    _stopFeatureComputation(false)
{
    // square neighborhood
    _locNeighbors = ((_neighborhoodSize * 2) + 1) * ((_neighborhoodSize * 2) + 1);
    // uniform weighting
    _neighborhoodWeighting = loc_Neigh_Weighting::WEIGHT_UNIF;
    _neighborhoodWeights.resize(_locNeighbors);
    std::fill(_neighborhoodWeights.begin(), _neighborhoodWeights.end(), 1);
}


FeatureExtraction::~FeatureExtraction()
{
}

void FeatureExtraction::compute() {
    qDebug() << "Feature extraction: started";

    computeHistogramFeatures();

    qDebug() << "Feature extraction: finished";

}

void FeatureExtraction::setup(const std::vector<unsigned int>& pointIds, const std::vector<float>& attribute_data, const Parameters& params) {
    _featType = params._featureType;

    // Parameters
    _numHistBins = params._numHistBins;
    _locNeighbors = params._numLocNeighbors;
    _neighborhoodWeighting = params._neighWeighting;

    // Set neighborhood
    _kernelWidth = (2 * _locNeighbors) + 1;
    _neighborhoodSize = _kernelWidth * _kernelWidth;
    weightNeighborhood(params._neighWeighting);     // sets _neighborhoodWeights

    // Data
    // Input
    _imgSize = params._imgSize;
    _pointIds = pointIds;
    _numPoints = pointIds.size();
    _numDims = params._numDims;
    _attribute_data = attribute_data;

    assert(_attribute_data.size() == _numPoints * _numDims);

    qDebug() << "Feature extraction: Num neighbors (in each direction): " << _locNeighbors << "(total neighbors: " << _neighborhoodSize << ") Neighbor weighting: " << (unsigned int)_neighborhoodWeighting;
    if (_featType == feature_type::TEXTURE_HIST_1D)
        qDebug() << "Feature extraction: Type 1d texture histogram, Num Bins: " << _numHistBins;
    else if(_featType == feature_type::LISA)
        qDebug() << "Feature extraction: LISA";
    else if (_featType == feature_type::GEARYC)
        qDebug() << "Feature extraction: local Geary's C";
    else if (_featType == feature_type::PCOL)
        qDebug() << "Feature extraction: Collection of points (neighborhood)";
    else
        qDebug() << "Feature extraction: unknown feature type";
}

void FeatureExtraction::computeHistogramFeatures() {
    // init, i.e. identify min and max per dimension for histogramming
    initExtraction();

    auto start = std::chrono::steady_clock::now();
    // convolution over all points to create histograms
    extractFeatures();
    auto end = std::chrono::steady_clock::now();
    qDebug() << "Feature extraction: extraction duration (sec): " << ((float)std::chrono::duration_cast<std::chrono::milliseconds> (end - start).count()) / 1000;

    // if there is a -1 in the features, this value was not set at all
    assert(std::find(_outFeatures.begin(), _outFeatures.end(), -1) == _outFeatures.end());
}

void FeatureExtraction::initExtraction() {
    qDebug() << "Feature extraction: init feature extraction";

    if (_featType == feature_type::TEXTURE_HIST_1D) {
        // find min and max for each channel, resize the output larger due to vector features
        _minMaxVals = CalcMinMaxPerChannel(_numPoints, _numDims, _attribute_data);
        _outFeatures.resize(_numPoints * _numDims * _numHistBins);
    }
    else if ((_featType == feature_type::LISA) | (_featType == feature_type::GEARYC)) {
        // find mean and varaince for each channel
        _meanVals = CalcMeanPerChannel(_numPoints, _numDims, _attribute_data);
        _varVals = CalcVarEstimate(_numPoints, _numDims, _attribute_data, _meanVals);
        _outFeatures.resize(_numPoints * _numDims);
    }
    else if (_featType == feature_type::PCOL)
        _outFeatures.resize(_numPoints * _numDims * _neighborhoodSize);

    // this is basically for easier debugging to see if all features are assigned a valid value
    // (currently only positive feature values are valid)
    std::fill(_outFeatures.begin(), _outFeatures.end(), -1);
}

void FeatureExtraction::extractFeatures() {
    qDebug() << "Feature extraction: extract features";

    // select feature extraction methood
    if (_featType == feature_type::TEXTURE_HIST_1D)
        featFunct = &FeatureExtraction::calculateHistogram;  // will be called as calculateHistogram(_pointIds[pointID], neighborValues);
    else if (_featType == feature_type::LISA)
        featFunct = &FeatureExtraction::calculateLISA;
    else if (_featType == feature_type::GEARYC)
        featFunct = &FeatureExtraction::calculateGearysC;
    else if (_featType == feature_type::PCOL)
        featFunct = &FeatureExtraction::calculateAllNeighborhoods;
    else
        qDebug() << "Feature extraction: unknown feature Type";

    // convolve over all selected data points
    #pragma omp parallel for
    for (int pointID = 0; pointID < (int)_numPoints; pointID++) {
        // get neighborhood ids of the current point
        std::vector<int> neighborIDs = neighborhoodIndices(_pointIds[pointID], _locNeighbors, _imgSize, _pointIds);
        assert(neighborIDs.size() == _neighborhoodSize);

        // get neighborhood values of the current point
        std::vector<float> neighborValues = getNeighborhoodValues(neighborIDs, _attribute_data, _neighborhoodSize, _numDims);

        // calculate feature(s) for neighborhood
        (this->*featFunct)(_pointIds[pointID], neighborValues);  // function pointer defined above
    }
}

void FeatureExtraction::calculateHistogram(size_t pointInd, std::vector<float> neighborValues) {
    assert(_outFeatures.size() == _numPoints * _numDims * _numHistBins);
    assert(_minMaxVals.size() == 2*_numDims);
    assert(_neighborhoodWeights.size() == _neighborhoodSize);

    // 1D histograms for each dimension
    // save the histogram in _histogramFeatures
    for (size_t dim = 0; dim < _numDims; dim++) {
        auto h = boost::histogram::make_histogram(boost::histogram::axis::regular(_numHistBins, _minMaxVals[2 * dim], _minMaxVals[2 * dim + 1]));
        // once this works, check if the following is faster (VS Studio will complain but compile)
        //auto h = boost::histogram::make_histogram_with(std::vector<float>(), boost::histogram::axis::regular(_numHistBins, _minMaxVals[dim], _minMaxVals[dim + 1]));
        for (size_t neighbor = 0; neighbor < _neighborhoodSize; neighbor++) {
            h(neighborValues[neighbor * _numDims + dim], boost::histogram::weight(_neighborhoodWeights[neighbor]));
        }

        assert(h.rank() == 1); // 1D hist
        assert(h.axis().size() == _numHistBins);

        for (size_t bin = 0; bin < _numHistBins; bin++) {
            _outFeatures[pointInd * _numDims * _numHistBins + dim * _numHistBins + bin] = h[bin];
        }
        // the max value is stored in the overflow bin
        if (h[_numHistBins] != 0) {
            _outFeatures[pointInd * _numDims * _numHistBins + dim * _numHistBins + _numHistBins - 1] += h[_numHistBins];
        }
    }

}

void FeatureExtraction::calculateLISA(size_t pointInd, std::vector<float> neighborValues) {
    assert(_outFeatures.size() == _numPoints * _numDims);
    assert(_varVals.size() == _numDims);
    assert(_neighborhoodWeights.size() == _neighborhoodSize);

    for (size_t dim = 0; dim < _numDims; dim++) {
        float neigh_diff_from_mean_sum = 0;
        for (size_t neighbor = 0; neighbor < _neighborhoodSize; neighbor++) {
            neigh_diff_from_mean_sum += _neighborhoodWeights[neighbor] * (neighborValues[neighbor * _numDims + dim] - _meanVals[dim]);
        }
        float diff_from_mean = (_attribute_data[pointInd * _numDims + dim] - _meanVals[dim]);
        _outFeatures[pointInd * _numDims + dim] = diff_from_mean * neigh_diff_from_mean_sum / _varVals[dim];
    }
}

void FeatureExtraction::calculateGearysC(size_t pointInd, std::vector<float> neighborValues) {
    assert(_outFeatures.size() == _numPoints * _numDims);
    assert(_meanVals.size() == _numDims);
    assert(_varVals.size() == _numDims);
    assert(_neighborhoodWeights.size() == _neighborhoodSize);

    for (size_t dim = 0; dim < _numDims; dim++) {
        float diff_from_neigh_sum = 0;
        float diff_from_neigh = 0;
        for (size_t neighbor = 0; neighbor < _neighborhoodSize; neighbor++) {
            diff_from_neigh = _attribute_data[pointInd * _numDims + dim] - neighborValues[neighbor * _numDims + dim];
            diff_from_neigh_sum += _neighborhoodWeights[neighbor] * (diff_from_neigh * diff_from_neigh);
        }
        _outFeatures[pointInd * _numDims + dim] = diff_from_neigh_sum / _varVals[dim];
    }
}

void FeatureExtraction::calculateAllNeighborhoods(size_t pointInd, std::vector<float> neighborValues) {
    assert(_outFeatures.size() == _numPoints * _numDims * _neighborhoodSize);

    // copy neighborValues into _outFeatures
    std::swap_ranges(neighborValues.begin(), neighborValues.end(), _outFeatures.begin() + (pointInd * _numDims * _neighborhoodSize));
}

void FeatureExtraction::weightNeighborhood(loc_Neigh_Weighting weighting) {
    _neighborhoodWeights.resize(_neighborhoodSize);
    switch (weighting)
    {
    case loc_Neigh_Weighting::WEIGHT_UNIF: std::fill(_neighborhoodWeights.begin(), _neighborhoodWeights.end(), 1); break; 
    case loc_Neigh_Weighting::WEIGHT_BINO: _neighborhoodWeights = BinomialKernel2D(_kernelWidth, norm_vec::NORM_MAX); break;
    case loc_Neigh_Weighting::WEIGHT_GAUS: _neighborhoodWeights = GaussianKernel2D(_kernelWidth, 1.0, norm_vec::NORM_NOT); break;
    default:  break;
    }
}

void FeatureExtraction::setNeighborhoodWeighting(loc_Neigh_Weighting weighting) {
    _neighborhoodWeighting = weighting;
    weightNeighborhood(weighting);
}

void FeatureExtraction::setNumLocNeighbors(size_t size) {
    _locNeighbors = size;
    _kernelWidth = (2 * size) + 1;
    _neighborhoodSize = _kernelWidth * _kernelWidth;
}

void FeatureExtraction::setNumHistBins(size_t size) {
    _numHistBins = size;
}


loc_Neigh_Weighting FeatureExtraction::getNeighborhoodWeighting()
{
    return _neighborhoodWeighting;
}

std::vector<float>* FeatureExtraction::output()
{
    return &_outFeatures;
}

void FeatureExtraction::stopFeatureCopmutation()
{
    _stopFeatureComputation = false;
}

bool FeatureExtraction::requestedStop()
{
    return _stopFeatureComputation;
}
