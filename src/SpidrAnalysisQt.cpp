#include "SpidrAnalysisQt.h"

#include "SpidrPlugin.h"

#include <cmath>
#include <algorithm>
#include <tuple>

#include <QDebug>


SpidrAnalysisQtWrapper::SpidrAnalysisQtWrapper() 
{

}


SpidrAnalysisQtWrapper::~SpidrAnalysisQtWrapper()
{
}

void SpidrAnalysisQtWrapper::setup(const std::vector<float>& attribute_data, const std::vector<unsigned int>& pointIDsGlobal, \
        const size_t numDimensions, const ImgSize imgSize, const QString embeddingName, std::vector<unsigned int>& backgroundIDsGlobal, \
        const unsigned int aknnMetric, const unsigned int featType, const unsigned int kernelType, const size_t numLocNeighbors, const size_t numHistBins, \
        const unsigned int aknnAlgType, const int numIterations, const int perplexity, const int exaggeration, const int expDecay, const float MVNweight, \
        bool publishFeaturesToCore, bool forceBackgroundFeatures)
{
    _attribute_data = attribute_data;
    _pointIDsGlobal = pointIDsGlobal;
    _backgroundIDsGlobal = backgroundIDsGlobal;
    _numDimensions = numDimensions;
    _imgSize = imgSize;
    _embeddingName = embeddingName;
    _aknnMetric = aknnMetric;
    _featType = featType;
    _kernelType = kernelType;
    _numLocNeighbors = numLocNeighbors;
    _numHistBins = numHistBins;
    _aknnAlgType = aknnAlgType;
    _numIterations = numIterations;
    _perplexity = perplexity;
    _exaggeration = exaggeration;
    _expDecay = expDecay;
    _MVNweight = MVNweight;
    _publishFeaturesToCore = publishFeaturesToCore;
    _forceBackgroundFeatures = forceBackgroundFeatures;
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
    _SpidrAnalysis->initializeAnalysisSettings(static_cast<feature_type> (_featType), static_cast<loc_Neigh_Weighting> (_kernelType), _numLocNeighbors, _numHistBins, 
        static_cast<knn_library> (_aknnAlgType), static_cast<distance_metric> (_aknnMetric), _MVNweight, _numIterations, _perplexity, _exaggeration, _expDecay, _forceBackgroundFeatures);

    // Compute data features
#ifdef NDEBUG
    emit progressMessage("Calculate features");
#endif
    _SpidrAnalysis->computeFeatures();
    _dataFeats = _SpidrAnalysis->getDataFeatures();

    // Publish feature to the core
    if (_publishFeaturesToCore || _forceBackgroundFeatures)
    {
        emit publishFeatures(_dataFeats.size() / _SpidrAnalysis->getParameters()._numFeatureValsPerPoint);
    }
    
    // Compute knn dists and inds
#ifdef NDEBUG
    emit progressMessage("Calculate distances and kNN");
#endif
    _SpidrAnalysis->computekNN();
    //std::tie(_knnIds, _knnDists) = _SpidrAnalysis->getKNN();
    emit finishedKnn(); // this connects to SpidrPlugin::tsneComputation, which triggers the t-SNE computation in TsneComputationQt
    // We don't do the following but instead transform in TsneComputationQt so that we can easily update the embedding view live
    //_SpidrAnalysis->computeEmbedding();
    //_emd_with_backgound = _SpidrAnalysis->outputWithBackground();
    //emit finishedEmbedding();
}

const std::tuple<std::vector<int>, std::vector<float>> SpidrAnalysisQtWrapper::getKNN() {
    return _SpidrAnalysis->getKNN();
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

const std::vector<float>* SpidrAnalysisQtWrapper::getFeatures() {
    return &_dataFeats;
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