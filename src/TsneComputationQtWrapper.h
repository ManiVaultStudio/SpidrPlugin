#pragma once
#include "SpidrAnalysisParameters.h"

#ifndef USE_MACOSSDK_OPENGL
#include "hdi/utils/glad/glad.h"
#endif
#include "OffscreenBuffer.h"

#include <Task.h>

#include "hdi/dimensionality_reduction/hd_joint_probability_generator.h"
#include "hdi/dimensionality_reduction/gradient_descent_tsne_texture.h"

#include <vector>
#include <string>

#include <QObject> 
#include <QThread>

class SpidrParameters;
class OffscreenBuffer;

class TsneWorkerTasks : public QObject
{
public:
    TsneWorkerTasks(QObject* parent, mv::Task* parentTask);

    mv::Task& getInitializeOffScreenBufferTask() { return _initializeOffScreenBufferTask; };
    mv::Task& getInitializeTsneTask() { return _initializeTsneTask; };
    mv::Task& getInitializeGradientDescentTask() { return _initializeGradientDescentTask; };
    mv::Task& getComputeGradientDescentTask() { return _computeGradientDescentTask; };

private:
    mv::Task    _initializeOffScreenBufferTask;
    mv::Task    _initializeGradientDescentTask;
    mv::Task    _initializeTsneTask;
    mv::Task    _computeGradientDescentTask;
};


class TsneComputationQtWrapper : public QObject
{
    Q_OBJECT
public:
    TsneComputationQtWrapper();
    ~TsneComputationQtWrapper();

    void setVerbose(bool verbose);
    void setIterations(int iterations);
    void setExaggerationIter(int exaggerationIter);
    void setExponentialDecay(int exponentialDecay);
    void setPerplexity(int perplexity);
    void setNumDimensionsOutput(int numDimensionsOutput);

    inline bool verbose() const { return _verbose; }
    inline size_t iterations() const { return _iterations; }
    inline int exaggerationIter() const { return _exaggerationIter; }
    inline int perplexity() const { return _perplexity; }
    inline int numDimensionsOutput() const { return _numDimensionsOutput; }
    inline int getNumCurrentIterations() const { return _currentIteration + 1; }

    /*!
     *
     *
     * \param knn_indices
     * \param knn_distances
     * \param params
     */
    void setup(const std::vector<int> knn_indices, const std::vector<float> knn_distances, const SpidrParameters params);

    /*!
     *
     *
     */
    void initTSNE();
    void stopGradientDescent();
    void markForDeletion();

    // Move the Offscreen buffer to the processing thread after creating it in the UI Thread
    void moveBufferToThread(QThread* thread);

    /*!
     * !
     *
     */
    void compute();

    /*!
     *
     *
     * \return
     */
    std::vector<float> output();

    std::vector<float>& outputRef();

    void setTask(mv::Task* parentTask) { _parentTask = parentTask; }
    void createTasks();

    inline bool isTsneRunning() { return _isTsneRunning; }
    inline bool isGradientDescentRunning() { return _isGradientDescentRunning; }
    inline bool isMarkedForDeletion() { return _isMarkedForDeletion; }

signals:
    void finishedEmbedding();
    void newEmbedding();
    void progressPercentage(const float& percentage);
    void progressSection(const QString& section);

private:
    void computeGradientDescent();
    void initGradientDescent();
    void embed();
    void copyFloatOutput();

private:
    // TSNE structures
    using HsneMatrix = hdi::dr::HDJointProbabilityGenerator<float>::sparse_scalar_matrix_type;
    HsneMatrix _probabilityDistribution;
    hdi::dr::GradientDescentTSNETexture<HsneMatrix> _GPGPU_tSNE;
    hdi::data::Embedding<float> _embedding;

    // Data
    std::vector<int> _knn_indices;               /*!<> */
    std::vector<float> _knn_distances;           /*!<> */
    size_t _numForegroundPoints;                            /*!<> */
    std::vector<float> _outputData;                     /*!<> */

    // Options
    size_t _iterations;                                    /*!<> */
    int _numTrees;
    int _numChecks;
    int _exaggerationIter;
    int _exponentialDecay;
    int _perplexity;                                    /*!<> */
    int _perplexity_multiplier;
    int _numDimensionsOutput;
    int _nn;                                            /*!<> */

    int _currentIteration;         /** Current iteration in the embedding / gradient descent process */

    // Evaluation (for determining the filename when saving the embedding to disk)
    std::string _embeddingName;                     /*!< Name of the embedding */
    size_t _numDataDims;

    // Flags
    bool _verbose;
    bool _isGradientDescentRunning;
    bool _isTsneRunning;
    bool _isMarkedForDeletion;

    size_t _continueFromIteration;

    /** Offscreen OpenGL buffer required to run the gradient descent */
    OffscreenBuffer* _offscreenBuffer; 

private: // Task
    mv::Task* _parentTask;
    TsneWorkerTasks* _tasks;
};