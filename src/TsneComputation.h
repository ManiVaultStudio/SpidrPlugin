#pragma once

#include "hdi/dimensionality_reduction/hd_joint_probability_generator.h"
#include "hdi/dimensionality_reduction/sparse_tsne_user_def_probabilities.h"
#include "hdi/dimensionality_reduction/gradient_descent_tsne_texture.h"

#include <QThread>

#include <vector>
#include <string>

class Parameters;

class TsneComputation : public QThread
{
    Q_OBJECT
public:
    TsneComputation();
    ~TsneComputation() override;

    void setVerbose(bool verbose);
    void setIterations(int iterations);
    void setExaggerationIter(int exaggerationIter);
    void setPerplexity(int perplexity);
    void setNumDimensionsOutput(int numDimensionsOutput);

    inline bool verbose() { return _verbose; }
    inline int iterations() { return _iterations; }
    inline int exaggerationIter() { return _exaggerationIter; }
    inline int perplexity() { return _perplexity; }
    inline int numDimensionsOutput() { return _numDimensionsOutput; }

    void initTSNE(const std::vector<int>* knn_indices, const std::vector<float>* knn_distances, Parameters params);
    void stopGradientDescent();
    void markForDeletion();

    const std::vector<float>& output();

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
    hdi::dr::HDJointProbabilityGenerator<float>::sparse_scalar_matrix_type _probabilityDistribution;
    hdi::dr::SparseTSNEUserDefProbabilities<float> _A_tSNE;
    hdi::dr::GradientDescentTSNETexture _GPGPU_tSNE;
    hdi::data::Embedding<float> _embedding;

    // Data
    unsigned int _numPoints;
    std::vector<float> _outputData;

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
