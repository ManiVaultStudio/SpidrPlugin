#include "SpidrSettingsWidget.h"

#include "DimensionSelectionWidget.h"
#include "SpidrPlugin.h"

// Qt header files:
#include <QDebug>
#include <QFileDialog>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QMessageBox>
#include <QScrollArea>
#include <QVBoxLayout>


SpidrSettingsWidget::SpidrSettingsWidget(SpidrPlugin& analysisPlugin)
:
_analysisPlugin(analysisPlugin)
{
    const auto minimumWidth = 200;
    setMinimumWidth(minimumWidth);
    setMaximumWidth(2 * minimumWidth);

    knnOptions.addItem("HNSW");

    distanceMetric.addItem("QF");
    distanceMetric.addItem("Hellinger");
//    distanceMetric.addItem("EMD");

    kernelWeight.addItem("Uniform");
    kernelWeight.addItem("Binomial");
    kernelWeight.addItem("Gaussian");

    connect(&dataOptions,   SIGNAL(currentIndexChanged(QString)), this, SIGNAL(dataSetPicked(QString)));
    connect(&knnOptions,    SIGNAL(currentIndexChanged(int)), this, SIGNAL(knnAlgorithmPicked(int)));
    connect(&distanceMetric,SIGNAL(currentIndexChanged(int)), this, SIGNAL(distanceMetricPicked(int)));

    connect(&kernelWeight, SIGNAL(currentIndexChanged(int)), this, SIGNAL(kernelWeightPicked(int)));
    connect(&kernelSize, SIGNAL(textChanged(QString)), SLOT(kernelSizeChanged(QString)));
    connect(&histBinSize, SIGNAL(textChanged(QString)), SLOT(histBinSizeChanged(QString)));

    connect(&numIterations, SIGNAL(textChanged(QString)), SLOT(numIterationsChanged(QString)));
    connect(&perplexity,    SIGNAL(textChanged(QString)), SLOT(perplexityChanged(QString)));
    connect(&exaggeration,  SIGNAL(textChanged(QString)), SLOT(exaggerationChanged(QString)));
    connect(&expDecay,      SIGNAL(textChanged(QString)), SLOT(expDecayChanged(QString)));
    connect(&numTrees,      SIGNAL(textChanged(QString)), SLOT(numTreesChanged(QString)));
    connect(&numChecks,     SIGNAL(textChanged(QString)), SLOT(numChecksChanged(QString)));
    connect(&theta,         SIGNAL(textChanged(QString)), SLOT(thetaChanged(QString)));

    connect(&startButton, &QPushButton::toggled, this, &SpidrSettingsWidget::onStartToggled);

    // Create group boxes for grouping together various settings
    QGroupBox* settingsBox = new QGroupBox("Basic settings");
    QGroupBox* advancedSettingsBox = new QGroupBox("Advanced Settings");
    
    advancedSettingsBox->setCheckable(true);
    advancedSettingsBox->setChecked(false);
    
    // Build the labels for all the options
    QLabel* iterationLabel = new QLabel("Iteration Count");
    QLabel* perplexityLabel = new QLabel("Perplexity");
    QLabel* knnAlgorithmLabel = new QLabel("KNN Algorithm");
    QLabel* distanceMetricLabel = new QLabel("Distance Metric");
    QLabel* exaggerationLabel = new QLabel("Exaggeration");
    QLabel* expDecayLabel = new QLabel("Exponential Decay");
    QLabel* numTreesLabel = new QLabel("Number of Trees");
    QLabel* numChecksLabel = new QLabel("Number of Checks");

    QLabel* kernelWeightLabel = new QLabel("Kernel Weighting");
    QLabel* kernelSizeLabel = new QLabel("Kernel Size");
    QLabel* histBinSizeLabel = new QLabel("Histogram Bins");

    // Set option default values
    numIterations.setFixedWidth(50);
    perplexity.setFixedWidth(50);
    exaggeration.setFixedWidth(50);
    expDecay.setFixedWidth(50);
    numTrees.setFixedWidth(50);
    numChecks.setFixedWidth(50);
    kernelSize.setFixedWidth(50);
    histBinSize.setFixedWidth(50);

    numIterations.setValidator(new QIntValidator(1, 10000, this));
    perplexity.setValidator(new QIntValidator(2, 50, this));
    exaggeration.setValidator(new QIntValidator(1, 10000, this));
    expDecay.setValidator(new QIntValidator(1, 10000, this));
    numTrees.setValidator(new QIntValidator(1, 10000, this));
    numChecks.setValidator(new QIntValidator(1, 10000, this));
    kernelSize.setValidator(new QIntValidator(1, 10000, this));
    histBinSize.setValidator(new QIntValidator(1, 10000, this));

    numIterations.setText("1000");
    perplexity.setText("30");
    exaggeration.setText("250");
    expDecay.setText("70");
    numTrees.setText("4");
    numChecks.setText("1024");
    kernelSize.setText("1");
    histBinSize.setText("5");

    startButton.setText("Start Computation");
    startButton.setFixedSize(QSize(150, 50));
    startButton.setCheckable(true);

    // Add options to their appropriate group box
    auto* const settingsLayout = new QGridLayout();

    settingsLayout->addWidget(knnAlgorithmLabel, 0, 0, 1, 2);
    settingsLayout->addWidget(&knnOptions, 1, 0, 1, 2);

    settingsLayout->addWidget(distanceMetricLabel, 2, 0, 1, 2);
    settingsLayout->addWidget(&distanceMetric, 3, 0, 1, 2);
    
    settingsLayout->addWidget(kernelWeightLabel, 4, 0, 1, 2);
    settingsLayout->addWidget(&kernelWeight, 5, 0, 1, 2);
    
    settingsLayout->addWidget(iterationLabel, 6, 0);
    settingsLayout->addWidget(&numIterations, 7, 0);
    
    settingsLayout->addWidget(perplexityLabel, 8, 0);
    settingsLayout->addWidget(&perplexity, 9, 0);

    settingsLayout->addWidget(kernelSizeLabel, 6, 1);
    settingsLayout->addWidget(&kernelSize, 7, 1);
    
    settingsLayout->addWidget(histBinSizeLabel, 8, 1);
    settingsLayout->addWidget(&histBinSize, 9, 1);
        
    settingsBox->setLayout(settingsLayout);

    auto* const advancedSettingsLayout = new QGridLayout();
    advancedSettingsLayout->addWidget(exaggerationLabel, 0, 0);
    advancedSettingsLayout->addWidget(&exaggeration, 1, 0);
    advancedSettingsLayout->addWidget(expDecayLabel, 0, 1);
    advancedSettingsLayout->addWidget(&expDecay, 1, 1);
    advancedSettingsLayout->addWidget(numTreesLabel, 2, 0);
    advancedSettingsLayout->addWidget(&numTrees, 3, 0);
    advancedSettingsLayout->addWidget(numChecksLabel, 2, 1);
    advancedSettingsLayout->addWidget(&numChecks, 3, 1);
    advancedSettingsBox->setLayout(advancedSettingsLayout);

    // Add all the parts of the settings widget together
    addWidget(&dataOptions);
    addWidget(settingsBox);
    addWidget(&_dimensionSelectionWidget);
    addWidget(advancedSettingsBox);
    addWidget(&startButton);
}

void SpidrSettingsWidget::computationStopped()
{
    startButton.setText("Start Computation");
    startButton.setChecked(false);
}

void SpidrSettingsWidget::dataChanged(const Points& points)
{
    _dimensionSelectionWidget.dataChanged(points);
}

std::vector<bool> SpidrSettingsWidget::getEnabledDimensions()
{
    return _dimensionSelectionWidget.getEnabledDimensions();
}


QString SpidrSettingsWidget::currentData()
{
    return dataOptions.currentText();
}

// Check if all input values are valid
bool SpidrSettingsWidget::hasValidSettings()
{
    if (!numIterations.hasAcceptableInput())
        return false;
    if (!perplexity.hasAcceptableInput())
        return false;
    if (!exaggeration.hasAcceptableInput())
        return false;
    if (!expDecay.hasAcceptableInput())
        return false;
    if (!numTrees.hasAcceptableInput())
        return false;
    if (!numChecks.hasAcceptableInput())
        return false;

    return true;
}

void SpidrSettingsWidget::checkInputStyle(QLineEdit& input)
{
    if (input.hasAcceptableInput())
    {
        input.setStyleSheet("");
    }
    else
    {
        input.setStyleSheet("border: 1px solid red");
    }
}


// SLOTS
void SpidrSettingsWidget::onStartToggled(bool pressed)
{
    // Do nothing if we have no data set selected
    if (dataOptions.currentText().isEmpty()) {
        return;
    }

    // Check if the tSNE settings are valid before running the computation
    if (!hasValidSettings()) {
        QMessageBox warningBox;
        warningBox.setText(tr("Some settings are invalid or missing. Continue with default values?"));
        QPushButton *continueButton = warningBox.addButton(tr("Continue"), QMessageBox::ActionRole);
        QPushButton *abortButton = warningBox.addButton(QMessageBox::Abort);

        warningBox.exec();

        if (warningBox.clickedButton() == abortButton) {
            return;
        }
    }
    startButton.setText(pressed ? "Stop Computation" : "Start Computation");
    pressed ? _analysisPlugin.startComputation() : _analysisPlugin.stopComputation();;
}

void SpidrSettingsWidget::kernelSizeChanged(const QString &)
{
    checkInputStyle(kernelSize);
}

void SpidrSettingsWidget::histBinSizeChanged(const QString &)
{
    checkInputStyle(histBinSize);
}

void SpidrSettingsWidget::numIterationsChanged(const QString &)
{
    checkInputStyle(numIterations);
}

void SpidrSettingsWidget::perplexityChanged(const QString &)
{
    checkInputStyle(perplexity);
}

void SpidrSettingsWidget::exaggerationChanged(const QString &)
{
    checkInputStyle(exaggeration);
}

void SpidrSettingsWidget::expDecayChanged(const QString &)
{
    checkInputStyle(expDecay);
}

void SpidrSettingsWidget::numTreesChanged(const QString &)
{
    checkInputStyle(numTrees);
}

void SpidrSettingsWidget::numChecksChanged(const QString &)
{
    checkInputStyle(numChecks);
}

void SpidrSettingsWidget::thetaChanged(const QString& )
{
    checkInputStyle(theta);
}
