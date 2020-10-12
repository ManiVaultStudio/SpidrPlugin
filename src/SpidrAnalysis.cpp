#include "SpidrAnalysis.h"

#include <QDebug>

SpidrAnalysis::SpidrAnalysis(QObject* parent) : QThread(parent)
{
    // Connect embedding
    connect(&_tsne, &TsneComputation::computationStopped, this, &SpidrAnalysis::embeddingComputationStopped);
    connect(&_tsne, &TsneComputation::newEmbedding, this, &SpidrAnalysis::newEmbedding);

}

SpidrAnalysis::~SpidrAnalysis()
{
}

void SpidrAnalysis::run() {
    spatialAnalysis();
}

void SpidrAnalysis::setupData(const std::vector<float>& attribute_data, const std::vector<unsigned int>& pointIDsGlobal, const size_t numDimensions, const QSize imgSize) {
    // Set data
    _attribute_data = attribute_data;
    _pointIDsGlobal = pointIDsGlobal;

    // Set parameters
    _params._numPoints = pointIDsGlobal.size();
    _params._numDims = numDimensions;
    _params._imgSize = imgSize;

    qDebug() << "SpidrAnalysis: Num data points: " << _params._numPoints << " Num dims: " << _params._numDims << " Image size (width, height): " << imgSize.width() << ", " << imgSize.height();
}

void SpidrAnalysis::initializeAnalysisSettings(const int featType, const int kernelInd, const size_t numLocNeighbors, const size_t numHistBins,\
                                               const int aknnAlgInd, const int aknnMetric, \
                                               const int numIterations, const int perplexity, const int exaggeration) {
    // initialize Feature Extraction Settings
    setFeatureType(featType);
    setKernelWeight(kernelInd);
    setNumLocNeighbors(numLocNeighbors);
    setNumHistBins(numHistBins);

    // initialize Distance Calculation Settings
    setKnnAlgorithm(aknnAlgInd);
    setDistanceMetric(aknnMetric);
    // number of nn is dertermined by perplexity, set in setPerplexity

    // Initialize the tSNE computation
    setNumIterations(numIterations);
    setPerplexity(perplexity);
    setExaggeration(exaggeration);
}


void SpidrAnalysis::spatialAnalysis() {

    // Extract features
    _featExtraction.setup(_pointIDsGlobal, _attribute_data, _params);
    _featExtraction.compute();
    std::vector<float>* dataFeats = _featExtraction.output();

    // Caclculate distances and kNN
    _distCalc.setup(_pointIDsGlobal, _attribute_data, dataFeats, _params);
    _distCalc.compute();
    std::vector<int>* indices = _distCalc.get_knn_indices();
    std::vector<float>* distances_squared = _distCalc.get_knn_distances_squared();

    // Compute t-SNE with the given data
    _tsne.setup(indices, distances_squared, _params);
    _tsne.compute();
}

void SpidrAnalysis::setFeatureType(const int index) {
    _params._featureType = static_cast<feature_type> (index);
}

void SpidrAnalysis::setKernelWeight(const int index) {
    _params._neighWeighting = static_cast<loc_Neigh_Weighting> (index);
}

void SpidrAnalysis::setNumLocNeighbors(const size_t num) {
    _params._numLocNeighbors = num;
}

void SpidrAnalysis::setNumHistBins(const size_t num) {
    _params._numHistBins = num;
}

void SpidrAnalysis::setKnnAlgorithm(const int index) {
    _params._aknn_algorithm = static_cast<knn_library> (index);
}

void SpidrAnalysis::setDistanceMetric(const int index) {
    _params._aknn_metric = static_cast<distance_metric> (index);
}

void SpidrAnalysis::setPerplexity(const unsigned perplexity) {
    _params._perplexity = perplexity;
    _params._nn = (perplexity * _params._perplexity_multiplier) + 1;

    // For small images, use less kNN
    if (_params._nn > _params._numPoints)
        _params._nn = _params._numPoints;
}

void SpidrAnalysis::setNumIterations(const unsigned num) {
    _params._numIterations = num;
}

void SpidrAnalysis::setExaggeration(const unsigned exag) {
    _params._exaggeration = exag;
}

const size_t SpidrAnalysis::getNumPoints() {
    return _pointIDsGlobal.size();
}

bool SpidrAnalysis::embeddingIsRunning() {
    return _tsne.isTsneRunning();
}

const std::vector<float>& SpidrAnalysis::output() {
    return _tsne.output();
}

void SpidrAnalysis::stopComputation() {
    _featExtraction.stopFeatureCopmutation();
    _tsne.stopGradientDescent();
}

const Parameters SpidrAnalysis::getParameters() {
    return _params;
}