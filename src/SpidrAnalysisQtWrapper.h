#pragma once

#include "SpidrAnalysis.h"

#include <QSize>
#include <QVariant>
#include <memory>
#include <vector>

/*!
 *
 *
 */
class SpidrAnalysisQtWrapper : public QObject
{
    Q_OBJECT
public:
    SpidrAnalysisQtWrapper();
    ~SpidrAnalysisQtWrapper() override;

    /*!
     *
     */
    void setup(const std::vector<float>& attribute_data, const std::vector<unsigned int>& pointIDsGlobal, \
        const size_t numDimensions, const ImgSize imgSize, const QString embeddingName, std::vector<unsigned int>& backgroundIDsGlobal, \
        const distance_metric distMetric, const feature_type featType, const loc_Neigh_Weighting kernelType, const size_t numLocNeighbors,  const size_t numHistBins, \
        const knn_library aknnAlgType, const int numIterations, const int perplexity, const int exaggeration, const int expDecay, float pixelWeight,\
        bool publishFeaturesToCore, bool forceBackgroundFeatures);

    void setup(const std::vector<float>& attribute_data, const std::vector<unsigned int>& pointIDsGlobal, \
        const QString embeddingName, std::vector<unsigned int>& backgroundIDsGlobal, \
        const SpidrParameters& spidrParameters);


    // Getter
    const size_t getNumForegroundPoints();

    const size_t getNumImagePoints();

    const size_t getNumFeatureValsPerPoint();

    const Feature getFeatures();

    bool embeddingIsRunning();
    
    /*!
     *
     *
     * \return
     */
    const std::vector<float> &output();

    const std::vector<float> &outputWithBackground();

    const SpidrParameters getParameters();

    /* Returns _knn_indices, _knn_distances, use with std::tie(_knnIds, _knnDists) = getKnn(); */
    const std::tuple<std::vector<int>, std::vector<float>> getKnn();

    /* Add bg points to emb */
    void addBackgroundToEmbedding(std::vector<float>& emb, std::vector<float>& emb_wo_bg);


public slots:

    /*!
     *
     *
     */
    void spatialAnalysis();

signals:
    void finishedKnn();

    void publishFeatures(const unsigned int dataFeatsSize);
    void progressSection(const QString& section);


private:

    std::unique_ptr<SpidrAnalysis> _SpidrAnalysis;

    // data
    std::vector<float> _attribute_data;
    std::vector<unsigned int> _pointIDsGlobal;
    std::vector<unsigned int> _backgroundIDsGlobal;
    size_t _numDimensions;
    ImgSize _imgSize;
    QString _embeddingName;

    // parameters
    distance_metric _distMetric;
    feature_type _featType;
    loc_Neigh_Weighting _kernelType;
    knn_library _aknnAlgType;
    size_t _numNeighborsInEachDirection;
    size_t _numHistBins;
    size_t _numIterations;
    size_t _perplexity;
    size_t _exaggeration;
    size_t _expDecay;
    float _pixelWeight;
    bool _publishFeaturesToCore;
    bool _forceBackgroundFeatures;

    // output
    std::vector<float> _emd_with_backgound;
    Feature _dataFeats;
    std::vector<int> _knnIds;
    std::vector<float> _knnDists;

};


