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
        const size_t numDimensions, const ImgSize imgSize, const QString embeddingName, std::vector<unsigned int>& backgroundIDsGlobal, std::vector<unsigned int>& contextIDsGlobal, \
        const distance_metric distMetric, const feature_type featType, const loc_Neigh_Weighting kernelType, const size_t numLocNeighbors, const size_t numHistBins, \
        const knn_library aknnAlgType, const int numIterations, const int perplexity, const int exaggeration, const int expDecay, float pixelWeight, \
        bool forceBackgroundFeatures)
{
    _attribute_data = attribute_data;
    _pointIDsGlobal = pointIDsGlobal;
    _backgroundIDsGlobal = backgroundIDsGlobal;
    _contextIDsGlobal = contextIDsGlobal;
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
    _forceBackgroundFeatures = forceBackgroundFeatures;

    if (_contextIDsGlobal.size() > 0 || _backgroundIDsGlobal.size() > 0)
    {
        // combine contextIDsGlobal and backgroundIDsGlobal
        std::set_union(_contextIDsGlobal.begin(), _contextIDsGlobal.end(),
                       _backgroundIDsGlobal.begin(), _backgroundIDsGlobal.end(),
                       std::back_inserter(_contextAndBackgroundIDsGlobal));

        // sort _contextAndBackgroundIDsGlobal
        std::sort(_contextAndBackgroundIDsGlobal.begin(), _contextAndBackgroundIDsGlobal.end());
    }

    std::set_difference(_pointIDsGlobal.begin(), _pointIDsGlobal.end(),
                        _contextAndBackgroundIDsGlobal.begin(), _contextAndBackgroundIDsGlobal.end(),
                        std::inserter(_foregroundIDsGlobal, _foregroundIDsGlobal.end()));

}

void SpidrAnalysisQtWrapper::setup(const std::vector<float>& attribute_data, const std::vector<unsigned int>& pointIDsGlobal, \
    const QString embeddingName, std::vector<unsigned int>& backgroundIDsGlobal, std::vector<unsigned int>& contextIDsGlobal, \
    const SpidrParameters& spidrParameters) {
    _attribute_data = attribute_data;
    _pointIDsGlobal = pointIDsGlobal;
    _backgroundIDsGlobal = backgroundIDsGlobal;
    _contextIDsGlobal = contextIDsGlobal;
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
    _forceBackgroundFeatures = spidrParameters._forceCalcBackgroundFeatures;

    if (_contextIDsGlobal.size() > 0 || _backgroundIDsGlobal.size() > 0)
    {
        // set_union expect sorted ranges
        std::sort(_contextIDsGlobal.begin(), _contextIDsGlobal.end());
        std::sort(_backgroundIDsGlobal.begin(), _backgroundIDsGlobal.end());

        // combine contextIDsGlobal and backgroundIDsGlobal
        std::set_union(_contextIDsGlobal.begin(), _contextIDsGlobal.end(),
                       _backgroundIDsGlobal.begin(), _backgroundIDsGlobal.end(),
                       std::back_inserter(_contextAndBackgroundIDsGlobal));
    }

    // set_difference expect sorted ranges
    std::sort(_pointIDsGlobal.begin(), _pointIDsGlobal.end());

    std::set_difference(_pointIDsGlobal.begin(), _pointIDsGlobal.end(),
                        _contextAndBackgroundIDsGlobal.begin(), _contextAndBackgroundIDsGlobal.end(),
                        std::inserter(_foregroundIDsGlobal, _foregroundIDsGlobal.end()));

}



void SpidrAnalysisQtWrapper::spatialAnalysis() {

    _SpidrAnalysis = std::make_unique<SpidrAnalysis>();

    // Pass data to SpidrLib
    if (_contextAndBackgroundIDsGlobal.empty())
        _SpidrAnalysis->setupData(_attribute_data, _pointIDsGlobal, _numDimensions, _imgSize, _embeddingName.toStdString());
    else
    {
        _SpidrAnalysis->setupData(_attribute_data, _pointIDsGlobal, _numDimensions, _imgSize, _embeddingName.toStdString(), _contextAndBackgroundIDsGlobal);
    }

    // Init all settings (setupData must have been called before initing the settings.)
    _SpidrAnalysis->initializeAnalysisSettings(_featType, _kernelType, _numNeighborsInEachDirection, _numHistBins, _pixelWeight,
        _aknnAlgType, _distMetric, _numIterations, _perplexity, _exaggeration, _expDecay, _forceBackgroundFeatures);

    // Compute data features
    emit progressSection("Calculate features");
    _SpidrAnalysis->computeFeatures();
    
    // Compute knn dists and inds
    emit progressSection("Calculate distances and kNN");
    _SpidrAnalysis->computekNN();

    // trigger SpidrPlugin::tsneComputation, which starts the t-SNE computation in TsneComputationQtWrapper
    emit finishedKnn(); 
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
        qDebug() << "SpidrAnalysisQtWrapper: Add background back to embedding";
        assert(_foregroundIDsGlobal.size() + _backgroundIDsGlobal.size() == _pointIDsGlobal.size() - _contextIDsGlobal.size());
        auto numEmbPoints = _foregroundIDsGlobal.size() + _backgroundIDsGlobal.size();
        emb.resize(numEmbPoints * 2);

        // find min x and min y embedding positions
        float minx = emb_wo_bg[0];
        float miny = emb_wo_bg[1];

        for (size_t i = 0; i < emb_wo_bg.size(); i += 2) {
            if (emb_wo_bg[i] < minx)
                minx = emb_wo_bg[i];

            if (emb_wo_bg[i + 1] < miny)
                miny = emb_wo_bg[i + 1];
        }

        minx -= std::abs(minx) * 0.05f;
        miny -= std::abs(miny) * 0.05f;


        // Iterate over the embedding:
        // Add the embedding coordinates if the current ID is in the foreground
        // or place it in the lower left corner if it's a background ID
        auto fgIt = _foregroundIDsGlobal.begin();
        auto bgIt = _backgroundIDsGlobal.begin();
        size_t idInEmbWoBg = 0;
        for (size_t i = 0; i < numEmbPoints; i++)
        {
            auto addFg = [&]() -> void {
                emb[2 * i] = emb_wo_bg[2 * idInEmbWoBg];
                emb[2 * i + 1] = emb_wo_bg[2 * idInEmbWoBg + 1];

                fgIt++;
                idInEmbWoBg++;
            };

            auto addBg = [&]() {
                emb[2 * i] = minx;
                emb[2 * i + 1] = miny;

                bgIt++;
            };

            if (bgIt == _backgroundIDsGlobal.end())
                addFg();
            else if (fgIt == _foregroundIDsGlobal.end())
                addBg();
            else if (*fgIt < *bgIt)
                addFg();
            else
                addBg();

       }

    }

}

const size_t SpidrAnalysisQtWrapper::getNumForegroundPoints() {
    return _SpidrAnalysis->getParameters()._numForegroundPoints;
}

const size_t SpidrAnalysisQtWrapper::getNumFeatureValsPerPoint() {
    return _SpidrAnalysis->getParameters()._numFeatureValsPerPoint;
}

const size_t SpidrAnalysisQtWrapper::getNumEmbPoints() {
    assert(_pointIDsGlobal.size() == _SpidrAnalysis->getParameters()._numForegroundPoints + _contextIDsGlobal.size() +  _backgroundIDsGlobal.size());
    return _foregroundIDsGlobal.size() + _backgroundIDsGlobal.size();
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