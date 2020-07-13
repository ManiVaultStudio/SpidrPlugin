#pragma once

#include "TsneComputation.h"
#include "DistanceCalculation.h"
#include "FeatureExtraction.h"
#include "AnalysisParameters.h"

#include <QThread>
#include <QSize>

#include <vector>

class SpidrAnalysis : public QThread
{
    Q_OBJECT
public:
    SpidrAnalysis();
    ~SpidrAnalysis() override;

    void setup(const std::vector<float>& attribute_data, const std::vector<unsigned int>& pointIDsGlobal, unsigned int numDimensions, QSize imgSize);

    void initializeTsneSettings(int numIterations, int perplexity, int exaggeration);

    // release openGL context of the t-SNE computation
    void stopComputation();

    // Setter
    void setKnnAlgorithm(const int index);
    void setDistanceMetric(const int index);
    void setKernelWeight(const int index);

    // Getter
    const unsigned int getNumPoints();
    bool embeddingIsRunning();
    const std::vector<float> &output();

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


