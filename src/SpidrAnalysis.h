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
    void setFeatureType(const int index);
    void setKernelWeight(const int index);
    void setNumLocNeighbors(const size_t index);
    void setNumHistBins(const size_t index);
    void setKnnAlgorithm(const int index);
    void setDistanceMetric(const int index);
    void setPerplexity(const unsigned  index);
    void setNumIterations(const unsigned  index);
    void setExaggeration(const unsigned  index);

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


