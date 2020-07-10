#pragma once

#include "TsneComputation.h"
#include "DistanceCalculation.h"
#include "FeatureExtraction.h"

#include <QThread>
#include <QSize>

#include <vector>

class Parameters {
public:
    Parameters() :
        _perplexity(30),
        _perplexity_multiplier(3),
        _aknn_algorithm(knn_library::KNN_HNSW),
        _aknn_metric(knn_distance_metric::KNN_METRIC_QF),
        _numHistBins(-1), _nn(-1), _numPoints(-1), _numDims(-1), _imgSize(-1, -1)
    {}

public:
    float               _perplexity;            //! Perplexity value in evert distribution.
    int                 _perplexity_multiplier; //! Multiplied by the perplexity gives the number of nearest neighbors used
    knn_library         _aknn_algorithm;
    knn_distance_metric _aknn_metric;
    unsigned int        _numHistBins;           // to be set in FeatureExtraction
    unsigned int        _nn;                    // number of nearest neighbors, determined by _perplexity*_perplexity_multiplier + 1; to be set in DistanceCalculation
    unsigned int        _numPoints;             // to be set in SpidrAnalysis
    unsigned int        _numDims;               // to be set in SpidrAnalysis
    QSize               _imgSize;               // to be set in SpidrAnalysis
};

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


