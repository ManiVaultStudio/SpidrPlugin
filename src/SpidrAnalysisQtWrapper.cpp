#include "SpidrAnalysisQtWrapper.h"

#include "SpidrPlugin.h"

#include <cmath>
#include <algorithm>
#include <tuple>


SpidrAnalysisQtWrapper::SpidrAnalysisQtWrapper()
{

}


SpidrAnalysisQtWrapper::~SpidrAnalysisQtWrapper()
{
}

void SpidrAnalysisQtWrapper::setup(const std::vector<float>& attribute_data, const std::vector<unsigned int>& pointIDsGlobal, \
        const size_t numDimensions, const ImgSize imgSize, const QString embeddingName, std::vector<unsigned int>& backgroundIDsGlobal, \
        const distance_metric distMetric, const feature_type featType, const loc_Neigh_Weighting kernelType, const size_t numLocNeighbors, const size_t numHistBins, \
        const knn_library aknnAlgType, const int numIterations, const int perplexity, const int exaggeration, const int expDecay, float pixelWeight, \
        bool publishFeaturesToCore, bool forceBackgroundFeatures)
{
    _attribute_data = attribute_data;
    _pointIDsGlobal = pointIDsGlobal;
    _backgroundIDsGlobal = backgroundIDsGlobal;
    _numDimensions = numDimensions;
    _imgSize = imgSize;
    _embeddingName = embeddingName;
    _distMetric = distMetric;
    _featType = featType;
    _kernelType = kernelType;
    _numNeighborsInEachDirection = numLocNeighbors;
    _numHistBins = numHistBins;
    _aknnAlgType = aknnAlgType;
    _numIterations = numIterations;
    _perplexity = perplexity;
    _exaggeration = exaggeration;
    _expDecay = expDecay;
    _pixelWeight = pixelWeight;
    _publishFeaturesToCore = publishFeaturesToCore;
    _forceBackgroundFeatures = forceBackgroundFeatures;
}

void SpidrAnalysisQtWrapper::setup(const std::vector<float>& attribute_data, const std::vector<unsigned int>& pointIDsGlobal, \
    const QString embeddingName, std::vector<unsigned int>& backgroundIDsGlobal, \
    const SpidrParameters& spidrParameters) {
    _attribute_data = attribute_data;
    _pointIDsGlobal = pointIDsGlobal;
    _backgroundIDsGlobal = backgroundIDsGlobal;
    _numDimensions = spidrParameters._numDims;
    _imgSize = spidrParameters._imgSize;
    _embeddingName = embeddingName;
    _distMetric = spidrParameters._aknn_metric;
    _featType = spidrParameters._featureType;
    _kernelType = spidrParameters._neighWeighting;
    _numNeighborsInEachDirection = spidrParameters.get_numNeighborsInEachDirection();
    _numHistBins = spidrParameters._numHistBins;
    _aknnAlgType = spidrParameters._aknn_algorithm;
    _numIterations = spidrParameters._numIterations;
    _perplexity = spidrParameters.get_perplexity();
    _exaggeration = spidrParameters._exaggeration;
    _expDecay = spidrParameters._expDecay;
    _pixelWeight = spidrParameters._pixelWeight;
    _publishFeaturesToCore = false;     // TODO not really used as all
    _forceBackgroundFeatures = spidrParameters._forceCalcBackgroundFeatures;

}



void SpidrAnalysisQtWrapper::spatialAnalysis() {

    _SpidrAnalysis = std::make_unique<SpidrAnalysis>();

    // Pass data to SpidrLib
    if (_backgroundIDsGlobal.empty())
        _SpidrAnalysis->setupData(_attribute_data, _pointIDsGlobal, _numDimensions, _imgSize, _embeddingName.toStdString());
    else
    {
        _SpidrAnalysis->setupData(_attribute_data, _pointIDsGlobal, _numDimensions, _imgSize, _embeddingName.toStdString(), _backgroundIDsGlobal);
    }

    // Init all settings (setupData must have been called before initing the settings.)
    _SpidrAnalysis->initializeAnalysisSettings(_featType, _kernelType, _numNeighborsInEachDirection, _numHistBins, _pixelWeight,
        _aknnAlgType, _distMetric, _numIterations, _perplexity, _exaggeration, _expDecay, _forceBackgroundFeatures);

    // Compute data features
    emit progressSection("Calculate features");
    _SpidrAnalysis->computeFeatures();
    _dataFeats = _SpidrAnalysis->getDataFeatures();

    // Publish feature to the core
    // TODO: Re-enable publishing features to core, maybe? See SpidrPlugin::onPublishFeatures
    if (_publishFeaturesToCore || _forceBackgroundFeatures)
    {
        //emit publishFeatures(_dataFeats.size() / _SpidrAnalysis->getParameters()._numFeatureValsPerPoint);
        emit publishFeatures(0);
    }
    
    // Compute knn dists and inds
    emit progressSection("Calculate distances and kNN");
    _SpidrAnalysis->computekNN();
    //std::tie(_knnIds, _knnDists) = _SpidrAnalysis->getKnn();
    emit finishedKnn(); // this connects to SpidrPlugin::tsneComputation, which triggers the t-SNE computation in TsneComputationQtWrapper
    // We don't do the following but instead transform in TsneComputationQtWrapper so that we can easily update the embedding view live
    //_SpidrAnalysis->computeEmbedding();
    //_emd_with_backgound = _SpidrAnalysis->outputWithBackground();
}

const std::tuple<std::vector<int>, std::vector<float>> SpidrAnalysisQtWrapper::getKnn() {
    return _SpidrAnalysis->getKnn();
}

void SpidrAnalysisQtWrapper::addBackgroundToEmbedding(std::vector<float>& emb, std::vector<float>& emb_wo_bg) {
    if (_backgroundIDsGlobal.empty())
    {
        std::swap(emb, emb_wo_bg);
    }
    else
    {
        _SpidrAnalysis->addBackgroundToEmbedding(emb, emb_wo_bg);
    }

}

const size_t SpidrAnalysisQtWrapper::getNumForegroundPoints() {
    return _SpidrAnalysis->getParameters()._numForegroundPoints;
}

const size_t SpidrAnalysisQtWrapper::getNumFeatureValsPerPoint() {
    return _SpidrAnalysis->getParameters()._numFeatureValsPerPoint;
}

const size_t SpidrAnalysisQtWrapper::getNumImagePoints() {
    assert(_pointIDsGlobal.size() == _SpidrAnalysis->getParameters()._numForegroundPoints + _backgroundIDsGlobal.size());
    return _SpidrAnalysis->getParameters()._numPoints;
}

const Feature SpidrAnalysisQtWrapper::getFeatures() {
    return _dataFeats;
}

bool SpidrAnalysisQtWrapper::embeddingIsRunning() {
    return _SpidrAnalysis->embeddingIsRunning();
}

const std::vector<float>& SpidrAnalysisQtWrapper::output() {
    return _SpidrAnalysis->output();
}

const std::vector<float>& SpidrAnalysisQtWrapper::outputWithBackground() {
    return _SpidrAnalysis->outputWithBackground();
}

const SpidrParameters SpidrAnalysisQtWrapper::getParameters() {
    return _SpidrAnalysis->getParameters();
}