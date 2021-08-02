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
        const unsigned int aknnMetric, const unsigned int featType, const unsigned int kernelType, const size_t numLocNeighbors,  const size_t numHistBins, \
        const unsigned int aknnAlgType, const int numIterations, const int perplexity, const int exaggeration, const int expDecay, const float MVNweight, \
        bool publishFeaturesToCore, bool forceBackgroundFeatures);


    // Getter
    const size_t getNumEmbPoints();

    const size_t getNumImagePoints();

    const size_t getNumFeatureValsPerPoint();

    const std::vector<float>* getFeatures();

    bool embeddingIsRunning();
    
    /*!
     *
     *
     * \return
     */
    const std::vector<float> &output();

    const std::vector<float> &outputWithBackground();

    const SpidrParameters getParameters();

    /* Returns _knn_indices, _knn_distances_squared, use with std::tie(_knnIds, _knnDists) = getKNN(); */
    const std::tuple<std::vector<int>, std::vector<float>> getKNN();

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
    void finishedEmbedding();

    void publishFeatures(const unsigned int dataFeatsSize);
    void progressMessage(const QString& message);

private:


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
    unsigned int _aknnMetric;
    unsigned int _featType;
    unsigned int _kernelType;
    unsigned int _aknnAlgType;
    size_t _numLocNeighbors;
    size_t _numHistBins;
    int _numIterations;
    int _perplexity;
    int _exaggeration;
    int _expDecay;
    float _MVNweight;
    bool _publishFeaturesToCore;
    bool _forceBackgroundFeatures;

    // output
    std::vector<float> _emd_with_backgound;
    std::vector<float> _dataFeats;
    std::vector<int> _knnIds;
    std::vector<float> _knnDists;

};


