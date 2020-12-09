#include "SpidrAnalysis.h"

#include <cmath>

#include <QDebug>

SpidrAnalysis::SpidrAnalysis(QObject* parent) : QThread(parent)
{
    // Connect embedding
    // connect(&_tsne, &TsneComputation::computationStopped, this, &SpidrAnalysis::embeddingComputationStopped);
    connect(&_tsne, &TsneComputation::newEmbedding, this, &SpidrAnalysis::newEmbedding);

}

SpidrAnalysis::~SpidrAnalysis()
{
}

void SpidrAnalysis::run() {
    spatialAnalysis();
}

void SpidrAnalysis::setupData(const std::vector<float>& attribute_data, const std::vector<unsigned int>& pointIDsGlobal, const size_t numDimensions, const QSize imgSize, const QString embeddingName, std::vector<unsigned int>& backgroundIDsGlobal) {
    // Set data
    _attribute_data = attribute_data;
    _pointIDsGlobal = pointIDsGlobal;
    _backgroundIDsGlobal = backgroundIDsGlobal;

    // Set parameters
    _params._numPoints = _pointIDsGlobal.size();
    _params._numDims = numDimensions;
    _params._imgSize = imgSize;
    _params._embeddingName = embeddingName.toStdString();
    _params._dataVecBegin = _attribute_data.data();          // used in point cloud distance

    qDebug() << "SpidrAnalysis: Num data points: " << _params._numPoints << " Num dims: " << _params._numDims << " Image size (width, height): " << _params._imgSize.width() << ", " << _params._imgSize.height();
}

void SpidrAnalysis::initializeAnalysisSettings(const int featType, const int kernelInd, const size_t numLocNeighbors, const size_t numHistBins,\
                                               const int aknnAlgInd, const int aknnMetric, \
                                               const int numIterations, const int perplexity, const int exaggeration) {
    // initialize Feature Extraction Settings
    setFeatureType(featType);
    setKernelWeight(kernelInd);
    setNumLocNeighbors(numLocNeighbors);    // Sets _params._kernelWidth and _params._neighborhoodSize as well
    setNumHistBins(numHistBins);

    // initialize Distance Calculation Settings
    setKnnAlgorithm(aknnAlgInd);
    setDistanceMetric(aknnMetric);
    // number of nn is dertermined by perplexity, set in setPerplexity

    // Initialize the tSNE computation
    setNumIterations(numIterations);
    setPerplexity(perplexity);
    setExaggeration(exaggeration);

    // Derived parameters
    setNumFeatureValsPerPoint(); 
}


void SpidrAnalysis::spatialAnalysis() {

    // Extract features
    _featExtraction.setup(_pointIDsGlobal, _attribute_data, _params);
    _featExtraction.compute();
    const std::vector<float> dataFeats = _featExtraction.output();

    // Caclculate distances and kNN
    _distCalc.setup(dataFeats, _backgroundIDsGlobal, _params);
    _distCalc.compute();
    const std::vector<int> knn_indices = _distCalc.get_knn_indices();
    const std::vector<float> knn_distances_squared = _distCalc.get_knn_distances_squared();

    // Compute t-SNE with the given data
    _tsne.setup(knn_indices, knn_distances_squared, _params);
    _tsne.compute();

    emit finishedEmbedding();
}

void SpidrAnalysis::embeddingComputationStopped() {
    
}

void SpidrAnalysis::setFeatureType(const int feature_type_index) {
    _params._featureType = static_cast<feature_type> (feature_type_index);
}

void SpidrAnalysis::setKernelWeight(const int loc_Neigh_Weighting_index) {
    _params._neighWeighting = static_cast<loc_Neigh_Weighting> (loc_Neigh_Weighting_index);
}

void SpidrAnalysis::setNumLocNeighbors(const size_t num) {
    _params._numLocNeighbors = num;
    _params._kernelWidth = (2 * _params._numLocNeighbors) + 1;
    _params._neighborhoodSize = _params._kernelWidth * _params._kernelWidth;;
}

void SpidrAnalysis::setNumHistBins(const size_t num) {
    _params._numHistBins = num;
}

void SpidrAnalysis::setKnnAlgorithm(const int knn_library_index) {
    _params._aknn_algorithm = static_cast<knn_library> (knn_library_index);
}

void SpidrAnalysis::setDistanceMetric(const int distance_metric_index) {
    _params._aknn_metric = static_cast<distance_metric> (distance_metric_index);
}

void SpidrAnalysis::setPerplexity(const unsigned perplexity) {
    _params._perplexity = perplexity;
    _params._nn = (perplexity * _params._perplexity_multiplier) + 1;    // see Van Der Maaten, L. (2014). Accelerating t-SNE using tree-based algorithms. The Journal of Machine Learning Research, 15(1), 3221-3245.

    // For small images, use less kNN
    if (_params._nn > _params._numPoints)
        _params._nn = _params._numPoints;
}

void SpidrAnalysis::setNumIterations(const unsigned numIt) {
    _params._numIterations = numIt;
}

void SpidrAnalysis::setExaggeration(const unsigned exag) {
    _params._exaggeration = exag;
}

void SpidrAnalysis::setNumFeatureValsPerPoint() {
    _params._numFeatureValsPerPoint = NumFeatureValsPerPoint(_params._featureType, _params._numDims, _params._numHistBins, _params._neighborhoodSize);
}


const size_t SpidrAnalysis::getNumEmbPoints() {
    return _params._numPoints;
}

const size_t SpidrAnalysis::getNumImagePoints() {
    assert(_pointIDsGlobal.size() == _params._numPoints + _backgroundIDsGlobal.size());
    return _pointIDsGlobal.size();
}

bool SpidrAnalysis::embeddingIsRunning() {
    return _tsne.isTsneRunning();
}

const std::vector<float>& SpidrAnalysis::output() {
    return _tsne.output();
}

const std::vector<float>& SpidrAnalysis::outputWithBackground() {
    const std::vector<float>& emb = _tsne.output();
    _emd_with_backgound.resize(_pointIDsGlobal.size() * 2);

    if (_backgroundIDsGlobal.empty())
    {
        return emb;
    }
    else
    {
        qDebug() << "SpidrAnalysis: Add background back to embedding";

        qDebug() << "SpidrAnalysis: Determine background position in embedding";

        // find min x and min y embedding positions
        float minx = emb[0];
        float miny = emb[1];

        for (size_t i = 0; i < emb.size(); i += 2) {
            if (emb[i] < minx)
                minx = emb[i];

            if (emb[i+1] < miny)
                miny = emb[i+1];
        }

        minx -= std::abs(minx) * 0.05;
        miny -= std::abs(miny) * 0.05;

        qDebug() << "SpidrAnalysis: Inserting background in embedding";

        // add (0,0) to embedding at background positions
        size_t emdCounter = 0;
        for (size_t globalIDCounter = 0; globalIDCounter < _pointIDsGlobal.size(); globalIDCounter++) {
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

void SpidrAnalysis::stopComputation() {
    _featExtraction.stopFeatureCopmutation();
    _tsne.stopGradientDescent();
}

const Parameters SpidrAnalysis::getParameters() {
    return _params;
}