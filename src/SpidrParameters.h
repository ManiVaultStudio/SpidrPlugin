#pragma once

#include "SpidrAnalysisParameters.h

class SpidrParameters
{
public:
    SpidrParameters() :
        _knnLibrary(hdi::dr::KNN_FLANN),
        _knnDistanceMetric(hdi::dr::KNN_METRIC_EUCLIDEAN),
        _numIterations(1000),
        _perplexity(30),
        _numTrees(4),
        _numChecks(1024),
        _exaggerationIter(250),
        _exponentialDecayIter(150),
        _numDimensionsOutput(2)
    {

    }

    void setKnnAlgorithm(hdi::dr::knn_library knnLibrary) { _knnLibrary = knnLibrary; }
    void setKnnDistanceMetric(hdi::dr::knn_distance_metric knnDistanceMetric) { _knnDistanceMetric = knnDistanceMetric; }
    void setNumIterations(int numIterations) { _numIterations = numIterations; }
    void setPerplexity(int perplexity) { _perplexity = perplexity; }
    void setNumTrees(int numTrees) { _numTrees = numTrees; }
    void setNumChecks(int numChecks) { _numChecks = numChecks; }
    void setExaggerationIter(int exaggerationIter) { _exaggerationIter = exaggerationIter; }
    void setExponentialDecayIter(int exponentialDecayIter) { _exponentialDecayIter = exponentialDecayIter; }
    void setNumDimensionsOutput(int numDimensionsOutput) { _numDimensionsOutput = numDimensionsOutput; }

    hdi::dr::knn_library getKnnAlgorithm() { return _knnLibrary; }
    hdi::dr::knn_distance_metric getKnnDistanceMetric() { return _knnDistanceMetric; }
    int getNumIterations() { return _numIterations; }
    int getPerplexity() { return _perplexity; }
    int getNumTrees() { return _numTrees; }
    int getNumChecks() { return _numChecks; }
    int getExaggerationIter() { return _exaggerationIter; }
    int getExponentialDecayIter() { return _exponentialDecayIter; }
    int getNumDimensionsOutput() { return _numDimensionsOutput; }

private:
    hdi::dr::knn_library _knnLibrary;
    hdi::dr::knn_distance_metric _knnDistanceMetric;
    int _numIterations;
    int _perplexity;
    int _numTrees;
    int _numChecks;
    int _exaggerationIter;
    int _exponentialDecayIter;
    int _numDimensionsOutput;
};
