#include "SpidrAnalysis.h"

#include "FeatureUtils.h"

SpidrAnalysis::SpidrAnalysis()
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

void SpidrAnalysis::setup(const std::vector<float>& attribute_data, const std::vector<unsigned int>& pointIDsGlobal, unsigned int numDimensions, QSize imgSize) {
    // Set data
    _attribute_data = attribute_data;
    _pointIDsGlobal = pointIDsGlobal;

    // Set parameters
    _params._numPoints = pointIDsGlobal.size();
    _params._numDims = numDimensions;
    _params._imgSize = imgSize;
}


void SpidrAnalysis::spatialAnalysis() {

    // Extract features
    _featExtraction.setupData(_pointIDsGlobal, _attribute_data, _params);
    _featExtraction.run();
    std::vector<float>* histoFeats = _featExtraction.output();

    // Caclculate distances and kNN
    _distCalc.setupData(histoFeats, _params);
    _distCalc.run();
    const std::vector<int>* indices = _distCalc.get_knn_indices();
    const std::vector<float>* distances_squared = _distCalc.get_knn_distances_squared();

    // Compute t-SNE with the given data
    _tsne.initTSNE(indices, distances_squared, _params);
    _tsne.run();
}

void SpidrAnalysis::initializeTsneSettings(int numIterations, int perplexity, int exaggeration) {

    // Initialize the tSNE computation with the settings from the settings widget
    _tsne.setIterations(numIterations);
    _tsne.setPerplexity(perplexity);
    _tsne.setExaggerationIter(exaggeration);
}


void SpidrAnalysis::setKnnAlgorithm(const int index) {
    _distCalc.setKnnAlgorithm(index);
}

void SpidrAnalysis::setDistanceMetric(const int index) {
    _distCalc.setDistanceMetric(index);
}

void SpidrAnalysis::setKernelWeight(const int index) {
    switch (index)
    {
    case 0 : _params._neighWeighting = loc_Neigh_Weighting::WEIGHT_UNIF; break;
    case 1 : _params._neighWeighting = loc_Neigh_Weighting::WEIGHT_BINO; break;
    case 2 : _params._neighWeighting = loc_Neigh_Weighting::WEIGHT_GAUS; break;
    default: _params._neighWeighting = loc_Neigh_Weighting::WEIGHT_UNIF; break;
    }
}


const unsigned int SpidrAnalysis::getNumPoints() {
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