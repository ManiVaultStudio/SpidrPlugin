#include "TsneComputationQtWrapper.h"

#include "EvalUtils.h"

#include <algorithm>            // std::min, max
#include <vector>
#include <assert.h>

#include "hdi/dimensionality_reduction/tsne_parameters.h"
#include "hdi/utils/scoped_timers.h"

// not present in glfw 3.1.2
#ifndef GLFW_FALSE
#define GLFW_FALSE 0
#endif

TsneComputationQtWrapper::TsneComputationQtWrapper() :
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
    _continueFromIteration(0)
{
    _nn = _perplexity * _perplexity_multiplier + 1;
}


void TsneComputationQtWrapper::computeGradientDescent()
{
    initGradientDescent();

    embed();
}

void TsneComputationQtWrapper::setup(const std::vector<int> knn_indices, const std::vector<float> knn_distances, const SpidrParameters params) {
    // SpidrParameters
    _iterations = params._numIterations;
    _perplexity = params.get_perplexity();
    _exaggerationIter = params._exaggeration;
    _exponentialDecay = params._expDecay;
    _nn = params.get_nn();                       // same as in constructor = _perplexity * 3 + 1;
    _numForegroundPoints = params._numForegroundPoints;
    _perplexity_multiplier = params.get_perplexity_multiplier();

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
#ifdef NDEBUG
    emit progressMessage("Initializing A-tSNE...");
#endif

    // Computation of the high dimensional similarities
    {
        hdi::dr::HDJointProbabilityGenerator<float>::Parameters probGenParams;
        probGenParams._perplexity = _perplexity;
        probGenParams._perplexity_multiplier = _perplexity_multiplier;
        probGenParams._num_trees = _numTrees;
        probGenParams._num_checks = _numChecks;

        spdlog::info("tSNE initialized.");

#ifdef NDEBUG
        emit progressMessage("Calculate probability distributions");
#endif

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

#ifdef NDEBUG
    emit progressMessage("Probability distributions calculated");
#endif

}

void TsneComputationQtWrapper::initGradientDescent()
{

#ifdef NDEBUG
    emit progressMessage("Initializing gradient descent");
#endif

    _continueFromIteration = 0;
    _isTsneRunning = true;

    hdi::dr::TsneParameters tsneParams;

    tsneParams._embedding_dimensionality = _numDimensionsOutput;
    tsneParams._mom_switching_iter = _exaggerationIter;
    tsneParams._remove_exaggeration_iter = _exaggerationIter;
    tsneParams._exponential_decay_iter = _exponentialDecay;
    tsneParams._exaggeration_factor = 4 + _numForegroundPoints / 60000.0;
    _A_tSNE.setTheta(std::min(0.5, std::max(0.0, (_numForegroundPoints - 1000.0)*0.00005)));

    // Create a offscreen window
    if (!glfwInit()) {
        throw std::runtime_error("Unable to initialize GLFW.");
    }
#ifdef __APPLE__
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#endif
    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);  // invisible - ie offscreen, window
    _offscreen_context = glfwCreateWindow(640, 480, "", NULL, NULL);
    if (_offscreen_context == NULL) {
        glfwTerminate();
        throw std::runtime_error("Failed to create GLFW window");
    }
    glfwMakeContextCurrent(_offscreen_context);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        glfwTerminate();
        throw std::runtime_error("Failed to initialize OpenGL context");
    }
    // Initialize GPGPU-SNE
    _GPGPU_tSNE.initialize(_probabilityDistribution, &_embedding, tsneParams);

    copyFloatOutput();
}

// Computing gradient descent
void TsneComputationQtWrapper::embed()
{
#ifdef NDEBUG
    emit progressMessage("Embedding");
#endif

    double elapsed = 0;
    double t = 0;
    {
        spdlog::info("A-tSNE: Computing gradient descent..\n");
        _isGradientDescentRunning = true;

        // Performs gradient descent for every iteration
        for (int iter = 0; iter < _iterations; ++iter)
        {
            hdi::utils::ScopedTimer<double> timer(t);
            if (!_isGradientDescentRunning)
            {
                _continueFromIteration = iter;
                break;
            }

            // Perform a GPGPU-SNE iteration
            _GPGPU_tSNE.doAnIteration();

            if (iter > 0 && iter % 10 == 0)
            {
                copyFloatOutput();
                emit newEmbedding();
            }

            if (t > 1000)
                spdlog::info("Time: {}", t);

            elapsed += t;

#ifdef NDEBUG
            emit progressMessage(QString("Computing gradient descent: %1 %").arg(QString::number(100.0f * static_cast<float>(iter) / static_cast<float>(_iterations), 'f', 1)));
#endif
        }
        glfwDestroyWindow(_offscreen_context);
        glfwTerminate();

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