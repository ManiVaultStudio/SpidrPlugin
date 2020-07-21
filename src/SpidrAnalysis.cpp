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

void SpidrAnalysis::initializeAnalysisSettings(const int kernelInd, const size_t numLocNeighbors, const size_t numHistBins,\
                                               const int aknnAlgInd, const int aknnMetInd, \
                                               const int numIterations, const int perplexity, const int exaggeration) {
    // initialize Feature Extraction Settings
    setKernelWeight(kernelInd);
    setNumLocNeighbors(numLocNeighbors);
    setNumHistBins(numHistBins);

    // initialize Distance Calculation Settings
    setKnnAlgorithm(aknnAlgInd);
    setDistanceMetric(aknnMetInd);
    _params._nn = (perplexity * _params._perplexity_multiplier) + 1;

    // Initialize the tSNE computation
    setNumIterations(numIterations);
    setPerplexity(perplexity);
    setExaggeration(exaggeration);
}


void SpidrAnalysis::spatialAnalysis() {

    // Extract features
    _featExtraction.setup(_pointIDsGlobal, _attribute_data, _params);
    _featExtraction.compute();
    std::vector<float>* histoFeats = _featExtraction.output();

    // Caclculate distances and kNN
    _distCalc.setup(histoFeats, _params);
    _distCalc.compute();
    std::vector<int>* indices = _distCalc.get_knn_indices();
    std::vector<float>* distances_squared = _distCalc.get_knn_distances_squared();

    // Compute t-SNE with the given data
    _tsne.setup(indices, distances_squared, _params);
    _tsne.compute();
}


void SpidrAnalysis::setKernelWeight(const int index) {
    switch (index)
    {
    case 0: _params._neighWeighting = loc_Neigh_Weighting::WEIGHT_UNIF; break;
    case 1: _params._neighWeighting = loc_Neigh_Weighting::WEIGHT_BINO; break;
    case 2: _params._neighWeighting = loc_Neigh_Weighting::WEIGHT_GAUS; break;
    default: _params._neighWeighting = loc_Neigh_Weighting::WEIGHT_UNIF; break;
    }
}

void SpidrAnalysis::setNumLocNeighbors(const size_t num) {
    _params._numLocNeighbors = num;
}

void SpidrAnalysis::setNumHistBins(const size_t num) {
    _params._numHistBins = num;
}

void SpidrAnalysis::setKnnAlgorithm(const int index) {
    // index corresponds to order in which algorithm were added to widget
    switch (index)
    {
    case 0: _params._aknn_algorithm = knn_library::KNN_HNSW; break;
    default: _params._aknn_algorithm = knn_library::KNN_HNSW;
    }
}

void SpidrAnalysis::setDistanceMetric(const int index) {
    // index corresponds to order in which algorithm were added to widget
    switch (index)
    {
    case 0: _params._aknn_metric = knn_distance_metric::KNN_METRIC_QF; break;
        //case 1: _knn_metric = knn_distance_metric::KNN_METRIC_EMD; break;
    case 1: _params._aknn_metric = knn_distance_metric::KNN_METRIC_HEL; break;
    default: _params._aknn_metric = knn_distance_metric::KNN_METRIC_QF;
    }
}

void SpidrAnalysis::setPerplexity(const unsigned num) {
    _params._perplexity = num;
}

void SpidrAnalysis::setNumIterations(const unsigned num) {
    _params._numIterations = num;
}

void SpidrAnalysis::setExaggeration(const unsigned num) {
    _params._exaggeration = num;
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
    _tsne.stopGradientDescent();
}

const Parameters SpidrAnalysis::getParameters() {
    return _params;
}