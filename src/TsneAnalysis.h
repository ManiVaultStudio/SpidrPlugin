#pragma once

#include "TsneData.h"

#include "hdi/dimensionality_reduction/hd_joint_probability_generator.h"
#include "hdi/dimensionality_reduction/sparse_tsne_user_def_probabilities.h"
#include "hdi/dimensionality_reduction/gradient_descent_tsne_texture.h"

#include <QThread>

#include <vector>
#include <string>

class TsneAnalysis : public QThread
{
    Q_OBJECT
public:
    TsneAnalysis();
    ~TsneAnalysis() override;

    void setKnnAlgorithm(int algorithm);
    void setDistanceMetric(int metric);
    void setVerbose(bool verbose);
    void setIterations(int iterations);
    void setNumTrees(int numTrees);
    void setNumChecks(int numChecks);
    void setExaggerationIter(int exaggerationIter);
    void setPerplexity(int perplexity);
    void setNumDimensionsOutput(int numDimensionsOutput);

    inline bool verbose() { return _verbose; }
    inline int iterations() { return _iterations; }
    inline int numTrees() { return _numTrees; }
    inline int numChecks() { return _numChecks; }
    inline int exaggerationIter() { return _exaggerationIter; }
    inline int perplexity() { return _perplexity; }
    inline int numDimensionsOutput() { return _numDimensionsOutput; }

    void initTSNE(const std::vector<float>& data, const int numDimensions);
    void stopGradientDescent();
    void markForDeletion();

    const TsneData& output();

    inline bool isTsneRunning() { return _isTsneRunning; }
    inline bool isGradientDescentRunning() { return _isGradientDescentRunning; }
    inline bool isMarkedForDeletion() { return _isMarkedForDeletion; }

private:
    void run() override;

    void computeGradientDescent();
    void initGradientDescent();
    void embed();
    void copyFloatOutput();

signals:
    void newEmbedding();
    void computationStopped();

private:
    // TSNE structures
    hdi::utils::knn_library _knnLibrary = hdi::utils::KNN_FLANN;
    hdi::utils::knn_distance_metric _knnDistanceMetric = hdi::utils::KNN_METRIC_EUCLIDEAN;
    hdi::dr::HDJointProbabilityGenerator<float>::sparse_scalar_matrix_type _probabilityDistribution;
    hdi::dr::SparseTSNEUserDefProbabilities<float> _A_tSNE;
    hdi::dr::GradientDescentTSNETexture _GPGPU_tSNE;
    hdi::data::Embedding<float> _embedding;

    // Data
    TsneData _inputData;
    TsneData _outputData;

    // Options
    int _iterations;
    int _numTrees;
    int _numChecks;
    int _exaggerationIter;
    int _perplexity;
    int _numDimensionsOutput;

    // Flags
    bool _verbose;
    bool _isGradientDescentRunning;
    bool _isTsneRunning;
    bool _isMarkedForDeletion;

    int _continueFromIteration;
};
