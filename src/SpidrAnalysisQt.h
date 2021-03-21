#pragma once

#include "TsneComputationQt.h"
#include "DistanceCalculation.h"
#include "FeatureExtraction.h"
#include "SpidrAnalysisParameters.h"
#include "FeatureUtils.h"
#include "KNNUtils.h"

#include <QThread>
#include <QSize>
#include <QVariant>

#include <vector>

/*!
 *
 *
 */
class SpidrAnalysisQt : public QThread
{
    Q_OBJECT
public:
    SpidrAnalysisQt(QObject* parent);
    ~SpidrAnalysisQt() override;

    /*!
     *
     *
     * \param attribute_data
     * \param pointIDsGlobal
     * \param numDimensions
     * \param imgSize
     * \param embeddingName
     * \param backgroundIDsGlobal ID of points which are not used during the t-SNE embedding - but will inform the feature extraction and distance calculation
     */
    void setupData(const std::vector<float>& attribute_data, const std::vector<unsigned int>& pointIDsGlobal, \
        const size_t numDimensions, const ImgSize imgSize, const QString embeddingName, std::vector<unsigned int>& backgroundIDsGlobal);

    // release openGL context of the t-SNE computation
    /*!
     *
     *
     */
    void stopComputation();

    /*! Set the parameters of the entire Analysis
     * Use the input from e.g a GUI
     *
     * \param featType
     * \param kernelInd
     * \param numLocNeighbors
     * \param numHistBins
     * \param aknnAlgInd
     * \param aknnMetInd
     * \param MVNweight
     * \param numIterations
     * \param perplexity
     * \param exaggeration
     * \param publishTicked
     */
    void initializeAnalysisSettings(const unsigned int featType, const unsigned int kernelType, const size_t numLocNeighbors, const size_t numHistBins, \
        const unsigned int aknnAlgType, const unsigned int aknnMetric, const float MVNweight, \
        const int numIterations, const int perplexity, const int exaggeration, const int expDecay, bool publishTicked);

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

public slots:
    void embeddingComputationStopped();

signals:
    void newEmbedding();
    void finishedEmbedding();

    void publishFeatures();
    void progressMessage(const QString& message);

private:
    void run() override;

    /*!
     *
     *
     */
    void spatialAnalysis();

    // Setter

    /*! Sets feature type as in enum class feature_type in FeatureUtils.h
    *
    * \param feature_type_index, see enum class feature_type in FeatureUtils.h
    */
    void setFeatureType(const int feature_type_index);

    /*! Sets feature type as in enum class loc_Neigh_Weighting in FeatureUtils.h
    *
    * \param loc_Neigh_Weighting_index, see enum class loc_Neigh_Weighting in FeatureUtils.h
    */
    void setKernelWeight(const int loc_Neigh_Weighting_index);

    /*! Sets the number of spatially local pixel neighbors in each direction. Sets _params._kernelWidth and _params._neighborhoodSize as well*/
    void setNumLocNeighbors(const size_t num);

    /*! Sets the number of histogram bins */
    void setNumHistBins(const size_t num);

    /*! Sets knn algorithm type as in enum class feature_type in KNNUtils.h
    *
    * \param knn_library_index, see enum class feature_type in KNNUtils.h
    */
    void setKnnAlgorithm(const int knn_library_index);

    /*! Sets knn algorithm type as in enum class distance_metric in KNNUtils.h
    *
    * \param distance_metric_index, see enum class distance_metric in KNNUtils.h
    */
    void setDistanceMetric(const int distance_metric_index);

    /*! Sets the perplexity and automatically determines the number of approximated kNN
    * nn = 3 * perplexity
    *
    * \param perplexity
    */
    void setPerplexity(const unsigned perplexity);
    /*! Sets the number of histogram bins */

    /*! Sets the number of gradient descent iteration */
    void setNumIterations(const unsigned numIt);

    /*! Sets the exageration during gradient descent */
    void setExaggeration(const unsigned exag);

    /*! Sets the exponential decay during gradient descent */
    void setExpDecay(const unsigned expDacay);

    /*! Sets the size of a feature, derived from other parameters */
    void setNumFeatureValsPerPoint();

    /*! Sets the spatial-attribut distance weight, 0 for only attributes and 1 for only spatial */
    void setMVNWeight(const float weight);

    /*! Sets whether features should be published to the core */
    void setPublishFeaturesToCore(const bool publishTicked);

private:
    // worker classes
    FeatureExtraction _featExtraction;          /*!<> */
    DistanceCalculation _distCalc;              /*!<> */
    TsneComputationQt _tsne;                      /*!<> */

    // data and setting
    std::vector<float> _attribute_data;         /*!<> */
    std::vector<unsigned int> _pointIDsGlobal;  /*!<> */
    std::vector<unsigned int> _backgroundIDsGlobal;  /*!< ID of points which are not used during the t-SNE embedding - but will inform the feature extraction and distance calculation > */
    SpidrParameters _params;                         /*!<> */
    std::vector<float> _emd_with_backgound;

    bool publishFeaturesToCore;
    std::vector<float> _dataFeats;
};


