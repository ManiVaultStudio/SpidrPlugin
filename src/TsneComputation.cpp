#include "TsneComputation.h"

#include "AnalysisParameters.h"
#include "EvalUtils.h"

#include <algorithm>            // std::min, max
#include <vector>
#include <assert.h>

#include <QDebug>

#include "hdi/dimensionality_reduction/tsne_parameters.h"
#include "hdi/utils/scoped_timers.h"

#include <QWindow>
#include <QOpenGLContext>

class OffscreenBuffer : public QWindow
{
public:
    OffscreenBuffer()
    {
        setSurfaceType(QWindow::OpenGLSurface);

        create();
    }

    QOpenGLContext* getContext() { return _context; }

    void initialize()
    {
        QOpenGLContext* globalContext = QOpenGLContext::globalShareContext();
        _context = new QOpenGLContext(this);
        _context->setFormat(globalContext->format());

        if (!_context->create())
            qFatal("Cannot create requested OpenGL context.");

        _context->makeCurrent(this);
        if (!gladLoadGL()) {
            qFatal("No OpenGL context is currently bound, therefore OpenGL function loading has failed.");
        }
    }

    void bindContext()
    {
        _context->makeCurrent(this);
    }

    void releaseContext()
    {
        _context->doneCurrent();
    }

private:
    QOpenGLContext* _context;
};

OffscreenBuffer* offBuffer;

TsneComputation::TsneComputation() :
_iterations(1000),
_numTrees(4),
_numChecks(1024),
_exaggerationIter(250),
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

TsneComputation::~TsneComputation()
{
}

void TsneComputation::computeGradientDescent()
{
    initGradientDescent();

    embed();
}

void TsneComputation::setup(const std::vector<int> knn_indices, const std::vector<float> knn_distances, const Parameters params) {
    // Parameters
    _iterations = params._numIterations;
    _perplexity = params._perplexity;
    _exaggerationIter = params._exaggeration;
    _nn = params._nn;                       // same as in constructor = _perplexity * 3 + 1;
    _numPoints = params._numPoints;
    _perplexity_multiplier = params._perplexity_multiplier;

    // Evaluation (for determining the filename when saving the embedding to disk)
    _embeddingName = params._embeddingName;
    _numDataDims = params._numDims;

    // Data
    _knn_indices = knn_indices;
    _knn_distances = knn_distances;

    qDebug() << "t-SNE computation: Num data points: " << _numPoints << " with " << params._nn << " precalculated nearest neighbors. Perplexity: " << _perplexity << ", Iterations: " << _iterations;

    assert(_knn_indices.size() == _numPoints * _nn);
}


void TsneComputation::initTSNE()
{
        
    // Computation of the high dimensional similarities
    {
        hdi::dr::HDJointProbabilityGenerator<float>::Parameters probGenParams;
        probGenParams._perplexity = _perplexity;
        probGenParams._perplexity_multiplier = _perplexity_multiplier;
        probGenParams._num_trees = _numTrees;
        probGenParams._num_checks = _numChecks;

        qDebug() << "tSNE initialized.";

        _probabilityDistribution.clear();
        _probabilityDistribution.resize(_numPoints);
        qDebug() << "Sparse matrix allocated.";

        hdi::dr::HDJointProbabilityGenerator<float> probabilityGenerator;
        double t = 0.0;
        {
            hdi::utils::ScopedTimer<double> timer(t);
            probabilityGenerator.computeGaussianDistributions(_knn_distances, _knn_indices, _nn, _probabilityDistribution, probGenParams);
        }
        qDebug() << "Probability distributions calculated.";
        qDebug() << "================================================================================";
        qDebug() << "A-tSNE: Compute probability distribution: " << t / 1000 << " seconds";
        qDebug() << "--------------------------------------------------------------------------------";
    }
}

void TsneComputation::initGradientDescent()
{
    _continueFromIteration = 0;

    _isTsneRunning = true;

    hdi::dr::TsneParameters tsneParams;

    tsneParams._embedding_dimensionality = _numDimensionsOutput;
    tsneParams._mom_switching_iter = _exaggerationIter;
    tsneParams._remove_exaggeration_iter = _exaggerationIter;
    tsneParams._exponential_decay_iter = 150;
    tsneParams._exaggeration_factor = 4 + _numPoints / 60000.0;
    _A_tSNE.setTheta(std::min(0.5, std::max(0.0, (_numPoints - 1000.0)*0.00005)));

    // Create a context local to this thread that shares with the global share context
    offBuffer = new OffscreenBuffer();
    offBuffer->initialize();

    // Initialize GPGPU-SNE
    offBuffer->bindContext();
    _GPGPU_tSNE.initialize(_probabilityDistribution, &_embedding, tsneParams);
    
    copyFloatOutput();
}

// Computing gradient descent
void TsneComputation::embed()
{
    double elapsed = 0;
    double t = 0;
    {
        qDebug() << "A-tSNE: Computing gradient descent..\n";
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
                qDebug() << "Time: " << t;

            elapsed += t;
        }
        offBuffer->releaseContext();

        copyFloatOutput();
        emit newEmbedding();
        
        _isGradientDescentRunning = false;
        _isTsneRunning = false;

        emit computationStopped();
    }

    qDebug() << "--------------------------------------------------------------------------------";
    qDebug() << "A-tSNE: Finished embedding of " << "tSNE Analysis" << " in: " << elapsed / 1000 << " seconds ";
    qDebug() << "================================================================================";

}

void TsneComputation::compute() {
    initTSNE();
    computeGradientDescent();
}

// Copy tSNE output to our output
void TsneComputation::copyFloatOutput()
{
    _outputData = _embedding.getContainer();
}

const std::vector<float>& TsneComputation::output()
{
    return _outputData;
}

void TsneComputation::setVerbose(bool verbose)
{
    _verbose = verbose;
}

void TsneComputation::setIterations(int iterations)
{
    _iterations = iterations;
}

void TsneComputation::setExaggerationIter(int exaggerationIter)
{
    _exaggerationIter = exaggerationIter;
}

void TsneComputation::setPerplexity(int perplexity)
{
    _perplexity = perplexity;
}

void TsneComputation::setNumDimensionsOutput(int numDimensionsOutput)
{
    _numDimensionsOutput = numDimensionsOutput;
}

void TsneComputation::stopGradientDescent()
{
    _isGradientDescentRunning = false;
}

void TsneComputation::markForDeletion()
{
    _isMarkedForDeletion = true;

    stopGradientDescent();
}
