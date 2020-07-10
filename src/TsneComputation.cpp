#include "TsneComputation.h"

#include <algorithm>            // std::min, max
#include <vector>
#include <assert.h>

#include "FeatureUtils.h"       // class Parameters
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
_numDimensionsOutput(2),
_verbose(false),
_isGradientDescentRunning(false),
_isTsneRunning(false),
_isMarkedForDeletion(false),
_continueFromIteration(0)
{
    
}

TsneComputation::~TsneComputation()
{
}

void TsneComputation::computeGradientDescent()
{
    initGradientDescent();

    embed();
}

void TsneComputation::initTSNE(const std::vector<int>* knn_indices, const std::vector<float>* knn_distances, Parameters params)
{
    _numPoints = knn_indices->size() / params._nn;
    qDebug() << "t-SNE computation. Num data points: " << _numPoints << " with " << params._nn << " precalculated nearest neighbors";
        
    // Computation of the high dimensional similarities
    qDebug() << "Output allocated.";
    {
        hdi::dr::HDJointProbabilityGenerator<float>::Parameters probGenParams;
        probGenParams._perplexity = _perplexity;
        probGenParams._perplexity_multiplier = 3;
        probGenParams._num_trees = _numTrees;
        probGenParams._num_checks = _numChecks;

        qDebug() << "tSNE initialized.";

        _probabilityDistribution.clear();
        _probabilityDistribution.resize(_numPoints);
        qDebug() << "Sparse matrix allocated.";

        qDebug() << "Computing high dimensional probability distributions.. Num data points: " << _numPoints;
        hdi::dr::HDJointProbabilityGenerator<float> probabilityGenerator;
        double t = 0.0;
        {
            hdi::utils::ScopedTimer<double> timer(t);
            probabilityGenerator.computeGaussianDistributions(*knn_distances, *knn_indices, params._nn, _probabilityDistribution, probGenParams);
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

void TsneComputation::run() {
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
