#pragma once

#include "TsneComputation.h"
#include "DistanceCalculation.h"
#include "FeatureExtraction.h"
#include "AnalysisParameters.h"
#include "FeatureUtils.h"
#include "KNNUtils.h"

#include <QThread>
#include <QSize>

#include <vector>

class SpidrAnalysis : public QThread
{
    Q_OBJECT
public:
    SpidrAnalysis(QObject* parent);
    ~SpidrAnalysis() override;

    void setupData(const std::vector<float>& attribute_data, const std::vector<unsigned int>& pointIDsGlobal, unsigned int numDimensions, QSize imgSize);

    // release openGL context of the t-SNE computation
    void stopComputation();

    // Setter
    void setKernelWeight(const int index);
    void setNumLocNeighbors(const int index);
    void setNumHistBins(const unsigned int index);
    void setKnnAlgorithm(const int index);
    void setDistanceMetric(const int index);
    void setPerplexity(const unsigned  index);
    void setNumIterations(const unsigned  index);
    void setExaggeration(const unsigned  index);

    void initializeAnalysisSettings(const int kernelInd, unsigned int numLocNeighbors, unsigned int numHistBins, \
                                    const int aknnAlgInd, const int aknnMetInd, \
                                    int numIterations, int perplexity, int exaggeration);

    // Getter
    const unsigned int getNumPoints();
    bool embeddingIsRunning();
    const std::vector<float> &output();
    const Parameters getParameters();

signals:
    void embeddingComputationStopped();
    void newEmbedding();


private:
    void run() override;

    void spatialAnalysis();

    // worker classes
    FeatureExtraction _featExtraction;
    DistanceCalculation _distCalc;
    TsneComputation _tsne;
    
    // data and setting
    std::vector<float> _attribute_data;
    std::vector<unsigned int> _pointIDsGlobal;
    Parameters _params;
};


