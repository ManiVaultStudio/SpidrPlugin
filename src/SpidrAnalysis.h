#pragma once

#include "TsneComputation.h"
#include "DistanceCalculation.h"
#include "FeatureExtraction.h"
#include "AnalysisParameters.h"
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
class SpidrAnalysis : public QThread
{
    Q_OBJECT
public:
    SpidrAnalysis(QObject* parent);
    ~SpidrAnalysis() override;

    /*!
     * 
     * 
     * \param attribute_data
     * \param pointIDsGlobal
     * \param numDimensions
     * \param imgSize
     */
    void setupData(const std::vector<float>& attribute_data, const std::vector<unsigned int>& pointIDsGlobal, const size_t numDimensions, const QSize imgSize, const QString embeddingName);

    // release openGL context of the t-SNE computation
    /*!
     * 
     * 
     */
    void stopComputation();

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
    
    /*! Sets the number of spatially local pixel neighbors in each direction*/
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

    /*! Set the parameters of the entire Analysis
     * Use the input from e.g a GUI
     * 
     * \param featType
     * \param kernelInd
     * \param numLocNeighbors
     * \param numHistBins
     * \param aknnAlgInd
     * \param aknnMetInd
     * \param numIterations
     * \param perplexity
     * \param exaggeration
     */
    void initializeAnalysisSettings(const int featType, const int kernelInd, const size_t numLocNeighbors, const size_t numHistBins, \
                                    const int aknnAlgInd, const int aknnMetric, \
                                    const int numIterations, const int perplexity, const int exaggeration);

    // Getter
    const size_t getNumPoints();
    bool embeddingIsRunning();
    /*!
     * 
     * 
     * \return 
     */
    const std::vector<float> &output();
    const Parameters getParameters();

signals:
    void embeddingComputationStopped();
    void newEmbedding();

private:
    void run() override;
    
    /*!
     * 
     * 
     */
    void spatialAnalysis();

    // worker classes
    FeatureExtraction _featExtraction;          /*!<> */
    DistanceCalculation _distCalc;              /*!<> */
    TsneComputation _tsne;                      /*!<> */
    
    // data and setting
    std::vector<float> _attribute_data;         /*!<> */
    std::vector<unsigned int> _pointIDsGlobal;  /*!<> */
    Parameters _params;                         /*!<> */
};


