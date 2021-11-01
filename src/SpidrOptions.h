#include "TsneParameters.h"

#include <QLabel>
#include <QLineEdit>
#include <QComboBox>

/** This class serves as a container of all the UI elements that provide parameter options for t-SNE */
class TsneOptions
{
public:
    TsneOptions(TsneParameters defaultParameters)
    {
        // UI Inputs
        knnOptions = new QComboBox();
        knnOptions->addItem("FLANN");
        knnOptions->addItem("HNSW");
        knnOptions->addItem("ANNOY");

        distanceMetric = new QComboBox();
        distanceMetric->addItem("Euclidean");
        distanceMetric->addItem("Cosine");
        distanceMetric->addItem("Inner Product");
        distanceMetric->addItem("Manhattan");
        distanceMetric->addItem("Hamming");
        distanceMetric->addItem("Dot");

        numIterations = new QLineEdit(QString::number(defaultParameters.getNumIterations()));
        perplexity = new QLineEdit(QString::number(defaultParameters.getPerplexity()));
        numTrees = new QLineEdit(QString::number(defaultParameters.getNumTrees()));
        numChecks = new QLineEdit(QString::number(defaultParameters.getNumChecks()));
        exaggeration = new QLineEdit(QString::number(defaultParameters.getExaggerationIter()));
        expDecay = new QLineEdit(QString::number(defaultParameters.getExponentialDecayIter()));

        // UI Labels
        knnAlgorithmLabel = new QLabel("KNN Algorithm");
        distanceMetricLabel = new QLabel("Distance Metric");

        iterationLabel = new QLabel("Iteration Count");
        perplexityLabel = new QLabel("Perplexity");
        numTreesLabel = new QLabel("# Trees for A-KNN");
        numChecksLabel = new QLabel("# Checks for A-KNN");
        exaggerationLabel = new QLabel("Exaggeration");
        expDecayLabel = new QLabel("Exponential Decay");
    }

    TsneParameters getSpidrParameters() const
    {
        TsneParameters parameters;

        // TODO Check if parameters are valid

        switch(knnOptions->currentIndex())
        {
        case 0: parameters.setKnnAlgorithm(hdi::dr::KNN_FLANN); break;
        case 1: parameters.setKnnAlgorithm(hdi::dr::KNN_HNSW); break;
        case 2: parameters.setKnnAlgorithm(hdi::dr::KNN_ANNOY); break;
        default: parameters.setKnnAlgorithm(hdi::dr::KNN_FLANN);
        }

        switch (knnOptions->currentIndex())
        {
        case 0: parameters.setKnnDistanceMetric(hdi::dr::KNN_METRIC_EUCLIDEAN); break;
        case 1: parameters.setKnnDistanceMetric(hdi::dr::KNN_METRIC_COSINE); break;
        case 2: parameters.setKnnDistanceMetric(hdi::dr::KNN_METRIC_INNER_PRODUCT); break;
        case 3: parameters.setKnnDistanceMetric(hdi::dr::KNN_METRIC_MANHATTAN); break;
        case 4: parameters.setKnnDistanceMetric(hdi::dr::KNN_METRIC_HAMMING); break;
        case 5: parameters.setKnnDistanceMetric(hdi::dr::KNN_METRIC_DOT); break;
        default: parameters.setKnnDistanceMetric(hdi::dr::KNN_METRIC_EUCLIDEAN);
        }

        parameters.setNumIterations(numIterations->text().toInt());
        parameters.setPerplexity(perplexity->text().toInt());
        parameters.setNumTrees(numTrees->text().toInt());
        parameters.setNumChecks(numChecks->text().toInt());
        parameters.setExaggerationIter(exaggeration->text().toInt());
        parameters.setExponentialDecayIter(expDecay->text().toInt());

        return parameters;
    }

    // UI Parameter control
    QComboBox* knnOptions;
    QComboBox* distanceMetric;

    QLineEdit* numIterations;
    QLineEdit* perplexity;
    QLineEdit* numTrees;
    QLineEdit* numChecks;
    QLineEdit* exaggeration;
    QLineEdit* expDecay;

    // Labels
    QLabel* knnAlgorithmLabel;
    QLabel* distanceMetricLabel;

    QLabel* iterationLabel;
    QLabel* perplexityLabel;
    QLabel* numTreesLabel;
    QLabel* numChecksLabel;
    QLabel* exaggerationLabel;
    QLabel* expDecayLabel;
};
