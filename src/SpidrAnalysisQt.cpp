#include "SpidrAnalysisQt.h"

#include "SpidrPlugin.h"

#include <cmath>
#include <algorithm>
#include <QDebug>

SpidrAnalysisQt::SpidrAnalysisQt(QObject* parent) : QThread(parent)
{
    // Connect embedding
    // connect(&_tsne, &TsneComputationQt::computationStopped, this, &SpidrAnalysisQt::embeddingComputationStopped);
    connect(&_tsne, &TsneComputationQt::newEmbedding, this, &SpidrAnalysisQt::newEmbedding);

    connect(&_tsne, &TsneComputationQt::progressMessage, this, &SpidrAnalysisQt::progressMessage);

}

SpidrAnalysisQt::~SpidrAnalysisQt()
{
}

void SpidrAnalysisQt::run() {
    spatialAnalysis();
}

void SpidrAnalysisQt::setupData(const std::vector<float>& attribute_data, const std::vector<unsigned int>& pointIDsGlobal, const size_t numDimensions, const ImgSize imgSize, const QString embeddingName, std::vector<unsigned int>& backgroundIDsGlobal) {
    // Set data
    _attribute_data = attribute_data;
    _pointIDsGlobal = pointIDsGlobal;
    _backgroundIDsGlobal = backgroundIDsGlobal;
    std::sort(_backgroundIDsGlobal.begin(), _backgroundIDsGlobal.end());

    // Set parameters
    _params._numPoints = _pointIDsGlobal.size();
    _params._numDims = numDimensions;
    _params._imgSize = imgSize;
    _params._embeddingName = embeddingName.toStdString();
    _params._dataVecBegin = _attribute_data.data();          // used in point cloud distance
    _params._forceCalcBackgroundFeatures = _forcePublishFeaturesToCore;

    qDebug() << "SpidrAnalysis: Num data points: " << _params._numPoints << " Num dims: " << _params._numDims << " Image size (width, height): " << _params._imgSize.width << ", " << _params._imgSize.height;
    if (!_backgroundIDsGlobal.empty())
        qDebug() << "SpidrAnalysis: Excluding "<< _backgroundIDsGlobal.size() << " background points and respective features";
}

void SpidrAnalysisQt::initializeAnalysisSettings(const unsigned int featType, const unsigned int kernelWeightType, const size_t numLocNeighbors, const size_t numHistBins, \
    const unsigned int aknnAlgType, const unsigned int aknnMetric, const float MVNweight, \
    const int numIterations, const int perplexity, const int exaggeration, const int expDecay, bool publishTicked, bool forcePublishTicked) {
    // initialize Feature Extraction Settings
    setFeatureType(featType);
    setKernelWeight(kernelWeightType);
    setNumLocNeighbors(numLocNeighbors);    // Sets _params._kernelWidth and _params._neighborhoodSize as well
    setNumHistBins(numHistBins);

    // initialize Distance Calculation Settings
    // number of nn is dertermined by perplexity, set in setPerplexity
    setKnnAlgorithm(aknnAlgType);
    setDistanceMetric(aknnMetric);
    setMVNWeight(MVNweight);

    // Initialize the tSNE computation
    setNumIterations(numIterations);
    setPerplexity(perplexity);
    setExaggeration(exaggeration);
    setExpDecay(expDecay);

    // Derived parameters
    setNumFeatureValsPerPoint();

    // Publish features to core?
    setPublishFeaturesToCore(publishTicked);
    setForcePublishFeaturesToCore(forcePublishTicked);
}


void SpidrAnalysisQt::spatialAnalysis() {

    // Extract features
    emit progressMessage("Calculate features");
    _featExtraction.setup(_pointIDsGlobal, _attribute_data, _params, &_backgroundIDsGlobal);
    _featExtraction.compute();
    qDebug() << "SpidrAnalysis: Get computed feature values";
    _dataFeats = _featExtraction.output();

    // Publish feature to the core
    if (_publishFeaturesToCore || _forcePublishFeaturesToCore)
    {
        emit publishFeatures();
    }

    // Caclculate distances and kNN
    emit progressMessage("Calculate distances and kNN");
    _distCalc.setup(_dataFeats, _backgroundIDsGlobal, _params);
    _distCalc.compute();
    const std::vector<int> knn_indices = _distCalc.get_knn_indices();
    const std::vector<float> knn_distances_squared = _distCalc.get_knn_distances_squared();

    // Compute t-SNE with the given data
    _tsne.setup(knn_indices, knn_distances_squared, _params);
    _tsne.compute();

    emit finishedEmbedding();
}

void SpidrAnalysisQt::embeddingComputationStopped() {

}

void SpidrAnalysisQt::setFeatureType(const int feature_type_index) {
    _params._featureType = static_cast<feature_type> (feature_type_index);
}

void SpidrAnalysisQt::setKernelWeight(const int loc_Neigh_Weighting_index) {
    _params._neighWeighting = static_cast<loc_Neigh_Weighting> (loc_Neigh_Weighting_index);
}

void SpidrAnalysisQt::setNumLocNeighbors(const size_t num) {
    _params._numLocNeighbors = num;
    _params._kernelWidth = (2 * _params._numLocNeighbors) + 1;
    _params._neighborhoodSize = _params._kernelWidth * _params._kernelWidth;;
}

void SpidrAnalysisQt::setNumHistBins(const size_t num) {
    _params._numHistBins = num;
}

void SpidrAnalysisQt::setKnnAlgorithm(const int knn_library_index) {
    _params._aknn_algorithm = static_cast<knn_library> (knn_library_index);
}

void SpidrAnalysisQt::setDistanceMetric(const int distance_metric_index) {
    _params._aknn_metric = static_cast<distance_metric> (distance_metric_index);
}

void SpidrAnalysisQt::setPerplexity(const unsigned perplexity) {
    _params._perplexity = perplexity;
    _params._nn = (perplexity * _params._perplexity_multiplier) + 1;    // see Van Der Maaten, L. (2014). Accelerating t-SNE using tree-based algorithms. The Journal of Machine Learning Research, 15(1), 3221-3245.

    // For small images, use less kNN
    if (_params._nn > _params._numPoints)
        _params._nn = _params._numPoints;
}

void SpidrAnalysisQt::setNumIterations(const unsigned numIt) {
    _params._numIterations = numIt;
}

void SpidrAnalysisQt::setExaggeration(const unsigned exag) {
    _params._exaggeration = exag;
}

void SpidrAnalysisQt::setExpDecay(const unsigned expDecay) {
    _params._expDecay = expDecay;
}

void SpidrAnalysisQt::setNumFeatureValsPerPoint() {
    _params._numFeatureValsPerPoint = NumFeatureValsPerPoint(_params._featureType, _params._numDims, _params._numHistBins, _params._neighborhoodSize);
}

void SpidrAnalysisQt::setMVNWeight(const float weight) {
    _params._MVNweight = weight;
}

void SpidrAnalysisQt::setPublishFeaturesToCore(const bool publishTicked) {
    _publishFeaturesToCore = publishTicked;
}

void SpidrAnalysisQt::setForcePublishFeaturesToCore(const bool ForcePublishTicked) {
    _forcePublishFeaturesToCore = ForcePublishTicked;
}

const size_t SpidrAnalysisQt::getNumEmbPoints() {
    return _params._numPoints;
}

const size_t SpidrAnalysisQt::getNumFeatureValsPerPoint() {
    return _params._numFeatureValsPerPoint;
}

const size_t SpidrAnalysisQt::getNumImagePoints() {
    assert(_pointIDsGlobal.size() == _params._numPoints + _backgroundIDsGlobal.size());
    return _pointIDsGlobal.size();
}

const std::vector<float>* SpidrAnalysisQt::getFeatures() {
    return &_dataFeats;
}

bool SpidrAnalysisQt::embeddingIsRunning() {
    return _tsne.isTsneRunning();
}

const std::vector<float>& SpidrAnalysisQt::output() {
    return _tsne.output();
}

const std::vector<float>& SpidrAnalysisQt::outputWithBackground() {
    const std::vector<float>& emb = _tsne.output();
    _emd_with_backgound.resize(_pointIDsGlobal.size() * 2);

    if (_backgroundIDsGlobal.empty())
    {
        return emb;
    }
    else
    {
        emit progressMessage("Add background back to embedding");
        qDebug() << "SpidrAnalysis: Add background back to embedding";

        qDebug() << "SpidrAnalysis: Determine background position in embedding";

        // find min x and min y embedding positions
        float minx = emb[0];
        float miny = emb[1];

        for (size_t i = 0; i < emb.size(); i += 2) {
            if (emb[i] < minx)
                minx = emb[i];

            if (emb[i + 1] < miny)
                miny = emb[i + 1];
        }

        minx -= std::abs(minx) * 0.05;
        miny -= std::abs(miny) * 0.05;

        qDebug() << "SpidrAnalysis: Inserting background in embedding";

        // add (0,0) to embedding at background positions
        size_t emdCounter = 0;
        for (int globalIDCounter = 0; globalIDCounter < _pointIDsGlobal.size(); globalIDCounter++) {
            // if background, insert (0,0)
            if (std::find(_backgroundIDsGlobal.begin(), _backgroundIDsGlobal.end(), globalIDCounter) != _backgroundIDsGlobal.end()) {
                _emd_with_backgound[2 * globalIDCounter] = minx;
                _emd_with_backgound[2 * globalIDCounter + 1] = miny;
            }
            else {
                _emd_with_backgound[2 * globalIDCounter] = emb[2 * emdCounter];
                _emd_with_backgound[2 * globalIDCounter + 1] = emb[2 * emdCounter + 1];
                emdCounter++;
            }
        }

        return _emd_with_backgound;
    }
}

void SpidrAnalysisQt::stopComputation() {
    _featExtraction.stopFeatureCopmutation();
    _tsne.stopGradientDescent();
}

const SpidrParameters SpidrAnalysisQt::getParameters() {
    return _params;
}