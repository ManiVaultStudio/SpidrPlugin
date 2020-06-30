#include "TsneAnalysis.h"

#include <vector>
#include <assert.h>

#include "SpidrPlugin.h"
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

TsneAnalysis::TsneAnalysis() :
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

TsneAnalysis::~TsneAnalysis()
{
}

void TsneAnalysis::computeGradientDescent()
{
    initGradientDescent();

    embed();
}

void TsneAnalysis::initTSNE(const std::vector<float>& data, const int numDimensions)
{
    unsigned int numPoints = data.size() / numDimensions;
    qDebug() << "Variables set. Num dims: " << numDimensions << " Num data points: " << numPoints;
    
    // Input data
    _inputData.assign(numPoints, numDimensions, data);
    qDebug() << "t-SNE input data assigned.";
    
    /// Computation of the high dimensional similarities
    qDebug() << "Output allocated.";
    {
        hdi::dr::HDJointProbabilityGenerator<float>::Parameters probGenParams;
        probGenParams._perplexity = _perplexity;
        probGenParams._perplexity_multiplier = 3;
        probGenParams._num_trees = _numTrees;
        probGenParams._num_checks = _numChecks;
        probGenParams._aknn_algorithm = _knnLibrary;
        probGenParams._aknn_metric = _knnDistanceMetric;

        qDebug() << "tSNE initialized.";

        _probabilityDistribution.clear();
        _probabilityDistribution.resize(numPoints);
        qDebug() << "Sparse matrix allocated.";

        qDebug() << "Computing high dimensional probability distributions.. Num dims: " << numDimensions << " Num data points: " << numPoints;
        hdi::dr::HDJointProbabilityGenerator<float> probabilityGenerator;
        double t = 0.0;
        {
            hdi::utils::ScopedTimer<double> timer(t);
            probabilityGenerator.computeJointProbabilityDistribution(_inputData.getDataNonConst().data(), numDimensions, numPoints, _probabilityDistribution, probGenParams);
        }
        qDebug() << "Probability distributions calculated.";
        qDebug() << "================================================================================";
        qDebug() << "A-tSNE: Compute probability distribution: " << t / 1000 << " seconds";
        qDebug() << "--------------------------------------------------------------------------------";
    }
}

void TsneAnalysis::initGradientDescent()
{
    _continueFromIteration = 0;

    _isTsneRunning = true;

    hdi::dr::TsneParameters tsneParams;

    tsneParams._embedding_dimensionality = _numDimensionsOutput;
    tsneParams._mom_switching_iter = _exaggerationIter;
    tsneParams._remove_exaggeration_iter = _exaggerationIter;
    tsneParams._exponential_decay_iter = 150;
    tsneParams._exaggeration_factor = 4 + _inputData.getNumPoints() / 60000.0;
    _A_tSNE.setTheta(std::min(0.5, std::max(0.0, (_inputData.getNumPoints() - 1000.0)*0.00005)));

    // Create a context local to this thread that shares with the global share context
    offBuffer = new OffscreenBuffer();
    offBuffer->initialize();

    // Initialize GPGPU-SNE
    offBuffer->bindContext();
    _GPGPU_tSNE.initialize(_probabilityDistribution, &_embedding, tsneParams);
    
    copyFloatOutput();
}

// Computing gradient descent
void TsneAnalysis::embed()
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

void TsneAnalysis::run() {
    computeGradientDescent();
}

// Copy tSNE output to our output
void TsneAnalysis::copyFloatOutput()
{
    _outputData.assign(_inputData.getNumPoints(), _numDimensionsOutput, _embedding.getContainer());
}

const TsneData& TsneAnalysis::output()
{
    return _outputData;
}

void TsneAnalysis::setKnnAlgorithm(int algorithm)
{
    switch (algorithm)
    {
    case 0: _knnLibrary = hdi::utils::KNN_FLANN; break;
    case 1: _knnLibrary = hdi::utils::KNN_HNSW; break;
    case 2: _knnLibrary = hdi::utils::KNN_ANNOY; break;
    default: _knnLibrary = hdi::utils::KNN_FLANN;
    }
}

void TsneAnalysis::setDistanceMetric(int metric)
{
    switch (metric)
    {
    case 0: _knnDistanceMetric = hdi::utils::KNN_METRIC_EUCLIDEAN; break;
    case 1: _knnDistanceMetric = hdi::utils::KNN_METRIC_COSINE; break;
    case 2: _knnDistanceMetric = hdi::utils::KNN_METRIC_INNER_PRODUCT; break;
    case 3: _knnDistanceMetric = hdi::utils::KNN_METRIC_MANHATTAN; break;
    case 4: _knnDistanceMetric = hdi::utils::KNN_METRIC_HAMMING; break;
    case 5: _knnDistanceMetric = hdi::utils::KNN_METRIC_DOT; break;
    default: _knnDistanceMetric = hdi::utils::KNN_METRIC_EUCLIDEAN;
    }
}

void TsneAnalysis::setVerbose(bool verbose)
{
    _verbose = verbose;
}

void TsneAnalysis::setIterations(int iterations)
{
    _iterations = iterations;
}

void TsneAnalysis::setNumTrees(int numTrees)
{
    _numTrees = numTrees;
}

void TsneAnalysis::setNumChecks(int numChecks)
{
    _numChecks = numChecks;
}

void TsneAnalysis::setExaggerationIter(int exaggerationIter)
{
    _exaggerationIter = exaggerationIter;
}

void TsneAnalysis::setPerplexity(int perplexity)
{
    _perplexity = perplexity;
}

void TsneAnalysis::setNumDimensionsOutput(int numDimensionsOutput)
{
    _numDimensionsOutput = numDimensionsOutput;
}

void TsneAnalysis::stopGradientDescent()
{
    _isGradientDescentRunning = false;
}

void TsneAnalysis::markForDeletion()
{
    _isMarkedForDeletion = true;

    stopGradientDescent();
}
