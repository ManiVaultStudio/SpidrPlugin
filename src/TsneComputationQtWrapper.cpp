#include "TsneComputationQtWrapper.h"

#include "EvalUtils.h"

#include <algorithm>            // std::min, max
#include <vector>
#include <assert.h>

#include "hdi/dimensionality_reduction/tsne_parameters.h"
#include "hdi/utils/scoped_timers.h"


TsneComputationQtWrapper::TsneComputationQtWrapper() :
    _currentIteration(0),
    _iterations(1000),
    _numTrees(4),
    _numChecks(1024),
    _exaggerationIter(250),
    _exponentialDecay(250),
    _perplexity(30),
    _perplexity_multiplier(3),
    _numDimensionsOutput(2),
    _verbose(false),
    _isGradientDescentRunning(false),
    _isTsneRunning(false),
    _isMarkedForDeletion(false),
    _continueFromIteration(0),
    _offscreenBuffer(nullptr)
{
    _nn = _perplexity * _perplexity_multiplier + 1;

    // Offscreen buffer must be created in the UI thread because it is a QWindow, afterwards we move it
    _offscreenBuffer = new OffscreenBuffer();
}

TsneComputationQtWrapper::~TsneComputationQtWrapper()
{
    delete _offscreenBuffer;
}

void TsneComputationQtWrapper::moveBufferToThread(QThread* thread)
{
    _offscreenBuffer->moveToThread(thread);
}


void TsneComputationQtWrapper::computeGradientDescent()
{
    initGradientDescent();

    embed();
}

void TsneComputationQtWrapper::setup(const std::vector<int> knn_indices, const std::vector<float> knn_distances, const SpidrParameters params) {
    // SpidrParameters
    _iterations = params._numIterations;
    _perplexity = static_cast<float> (params.get_perplexity());
    _exaggerationIter = static_cast<unsigned int> (params._exaggeration);
    _exponentialDecay = static_cast<unsigned int> (params._expDecay);
    _nn = static_cast<int> (params.get_nn());                       // same as in constructor = _perplexity * 3 + 1;
    _numForegroundPoints = params._numForegroundPoints;         // if no background IDs are given, _numForegroundPoints = _numPoints
    _perplexity_multiplier = static_cast<int> (params.get_perplexity_multiplier());

    // Evaluation (for determining the filename when saving the embedding to disk)
    _embeddingName = params._embeddingName;
    _numDataDims = params._numDims;

    // Data
    _knn_indices = knn_indices;
    _knn_distances = knn_distances;

    spdlog::info("t-SNE computation: Num data points: {0} with {1} precalculated nearest neighbors. Perplexity: {2}, Iterations: {3}", _numForegroundPoints, params.get_nn(), _perplexity, _iterations);

    assert(_knn_indices.size() == _numForegroundPoints * _nn);
}


void TsneComputationQtWrapper::initTSNE()
{
    emit progressSection("Initializing A-tSNE");

    // Computation of the high dimensional similarities
    {
        hdi::dr::HDJointProbabilityGenerator<float>::Parameters probGenParams;
        probGenParams._perplexity = _perplexity;
        probGenParams._perplexity_multiplier = _perplexity_multiplier;
        probGenParams._num_trees = _numTrees;
        probGenParams._num_checks = _numChecks;

        spdlog::info("tSNE initialized.");

        emit progressSection("Calculate probability distributions");

        _probabilityDistribution.clear();
        _probabilityDistribution.resize(_numForegroundPoints);
        spdlog::info("Sparse matrix allocated.");

        hdi::dr::HDJointProbabilityGenerator<float> probabilityGenerator;
        double t = 0.0;
        {
            hdi::utils::ScopedTimer<double> timer(t);
            probabilityGenerator.computeGaussianDistributions(_knn_distances, _knn_indices, _nn, _probabilityDistribution, probGenParams);
        }
        spdlog::info("Probability distributions calculated.");
        spdlog::info("================================================================================");
        spdlog::info("A-tSNE: Compute probability distribution: {} seconds", t / 1000);
        spdlog::info("--------------------------------------------------------------------------------");
    }

    emit progressSection("Probability distributions calculated");

}

void TsneComputationQtWrapper::initGradientDescent()
{
    emit progressSection("Initializing gradient descent");

    _continueFromIteration = 0;
    _isTsneRunning = true;

    hdi::dr::TsneParameters tsneParams;

    tsneParams._embedding_dimensionality = _numDimensionsOutput;
    tsneParams._mom_switching_iter = _exaggerationIter;
    tsneParams._remove_exaggeration_iter = _exaggerationIter;
    tsneParams._exponential_decay_iter = _exponentialDecay;
    tsneParams._exaggeration_factor = 4 + _numForegroundPoints / 60000.0;
    
    // Create a context local to this thread that shares with the global share context
    _offscreenBuffer->initialize();
    _offscreenBuffer->bindContext();

    // Initialize GPGPU-SNE
    _GPGPU_tSNE.initialize(_probabilityDistribution, &_embedding, tsneParams);

    copyFloatOutput();
}

// Computing gradient descent
void TsneComputationQtWrapper::embed()
{
    emit progressSection("Embedding");
    
    const auto emitEmbeddingUpdate = [this](const std::uint32_t& numProcessed, const std::uint32_t& numTotal) -> void {
        emit progressSection(QString("Embedding (step %1 of %2)").arg(QString::number(numProcessed), QString::number(numTotal)));
        emit progressPercentage(static_cast<float>(numProcessed) / static_cast<float>(numTotal));
    };

    double elapsed = 0;
    double t = 0;
    {
        spdlog::info("A-tSNE: Computing gradient descent..\n");
        _isGradientDescentRunning = true;

        const auto beginIteration = _currentIteration;

        // Performs gradient descent for every iteration
        for (_currentIteration = beginIteration; _currentIteration < _iterations; ++_currentIteration)
        {
            hdi::utils::ScopedTimer<double> timer(t);
            if (!_isGradientDescentRunning)
            {
                _continueFromIteration = _currentIteration;
                break;
            }

            // Perform a GPGPU-SNE iteration
            _GPGPU_tSNE.doAnIteration();

            if (_currentIteration > 0 && _currentIteration % 10 == 0)
            {
                copyFloatOutput();
                emit newEmbedding();
                emitEmbeddingUpdate(_currentIteration, _iterations);
            }

            if (t > 1000)
                spdlog::info("Time: {}", t);

            elapsed += t;

        }

        _offscreenBuffer->releaseContext();

        copyFloatOutput();
        emit newEmbedding();

        _isGradientDescentRunning = false;
        _isTsneRunning = false;

    }

    emit finishedEmbedding();

    spdlog::info("--------------------------------------------------------------------------------");
    spdlog::info("A-tSNE: Finished embedding of tSNE Analysis in: {} seconds", elapsed / 1000);
    spdlog::info("================================================================================");
}

void TsneComputationQtWrapper::compute() {
    assert(_offscreenBuffer->thread() == this->thread());   // ensure that this object and the buffer work in the same thread

    initTSNE();
    computeGradientDescent();
}

// Copy tSNE output to our output
void TsneComputationQtWrapper::copyFloatOutput()
{
    _outputData = _embedding.getContainer();
}

std::vector<float> TsneComputationQtWrapper::output()
{
    return _outputData;
}

void TsneComputationQtWrapper::setVerbose(bool verbose)
{
    _verbose = verbose;
}

void TsneComputationQtWrapper::setIterations(int iterations)
{
    _iterations = iterations;
}

void TsneComputationQtWrapper::setExaggerationIter(int exaggerationIter)
{
    _exaggerationIter = exaggerationIter;
}

void TsneComputationQtWrapper::setExponentialDecay(int exponentialDecay)
{
    _exponentialDecay = exponentialDecay;
}

void TsneComputationQtWrapper::setPerplexity(int perplexity)
{
    _perplexity = perplexity;
}

void TsneComputationQtWrapper::setNumDimensionsOutput(int numDimensionsOutput)
{
    _numDimensionsOutput = numDimensionsOutput;
}

void TsneComputationQtWrapper::stopGradientDescent()
{
    _isGradientDescentRunning = false;
}

void TsneComputationQtWrapper::markForDeletion()
{
    _isMarkedForDeletion = true;

    stopGradientDescent();
}

