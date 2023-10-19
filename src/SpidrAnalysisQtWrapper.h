#pragma once

#include "SpidrAnalysis.h"

#include <QSize>
#include <QVariant>
#include <memory>
#include <vector>

#include <Task.h>

class SpidrWorkerTasks : public QObject
{
public:
    SpidrWorkerTasks(QObject* parent, mv::Task* parentTask);

    mv::Task& getComputeFeaturesTask() { return _computeFeaturesTask; };
    mv::Task& getComputekNNTask() { return _computekNN; };

private:
    mv::Task    _computeFeaturesTask;
    mv::Task    _computekNN;
};


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
        const size_t numDimensions, const ImgSize imgSize, const QString embeddingName, std::vector<unsigned int>& backgroundIDsGlobal, std::vector<unsigned int>& contextIDsGlobal, \
        const distance_metric distMetric, const feature_type featType, const loc_Neigh_Weighting kernelType, const size_t numLocNeighbors,  const size_t numHistBins, \
        const knn_library aknnAlgType, const int numIterations, const int perplexity, const int exaggeration, const int expDecay, float pixelWeight,\
        bool forceBackgroundFeatures);

    void setup(const std::vector<float>& attribute_data, const std::vector<unsigned int>& pointIDsGlobal, \
        const QString embeddingName, std::vector<unsigned int>& backgroundIDsGlobal, std::vector<unsigned int>& contextIDsGlobal, \
        const SpidrParameters& spidrParameters);


    // Getter
    const size_t getNumForegroundPoints();

    const size_t getNumEmbPoints();

    const size_t getNumFeatureValsPerPoint();

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

    void setTask(mv::Task* parentTask) { _parentTask = parentTask; }
    void createTasks();

public slots:

    /*!
     *
     *
     */
    void spatialAnalysis();

signals:
    void finishedKnn();

    void progressSection(const QString& section);


private:

    std::unique_ptr<SpidrAnalysis> _SpidrAnalysis;

    // data
    std::vector<float> _attribute_data;
    std::vector<unsigned int> _pointIDsGlobal;
    std::vector<unsigned int> _backgroundIDsGlobal;
    std::vector<unsigned int> _contextIDsGlobal;
    std::vector<unsigned int> _contextAndBackgroundIDsGlobal;
    std::vector<unsigned int> _foregroundIDsGlobal;
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
    bool _forceBackgroundFeatures;

    // output
    std::vector<float> _emd_with_backgound;
    std::vector<int> _knnIds;
    std::vector<float> _knnDists;

private: // Task
    mv::Task* _parentTask;
    SpidrWorkerTasks* _tasks;

};


