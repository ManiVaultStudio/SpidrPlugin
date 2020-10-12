#include "SpidrSettingsWidget.h"

#include "DimensionSelectionWidget.h"
#include "SpidrPlugin.h"
#include "FeatureUtils.h"

// Qt header files:
#include <QDebug>
#include <QFileDialog>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QVariant>
#include <QPoint>
#include <QMessageBox>
#include <QScrollArea>
#include <QVBoxLayout>
#include <QStandardItemModel> 

SpidrSettingsWidget::SpidrSettingsWidget(SpidrPlugin& analysisPlugin)
:
_analysisPlugin(analysisPlugin)
{
    const auto minimumWidth = 200;
    setMinimumWidth(minimumWidth);
    setMaximumWidth(2 * minimumWidth);
    
    // add data item according to enum knn_library (KNNUtils)
    knnOptions.addItem("HNSW", QVariant(1));
    knnOptions.addItem("Exact", QVariant(0));

    // add options in the order as defined in enums in utils files
    // data values (QPoint) store feature_type (FeatureUtils) and distance_metric (KNNUtils) values as x and y 
    // this is used as a nice way to cast this information internally in SpidrAnalysis
    distanceMetric.addItem("Quadratic form (TH)", QVariant(QPoint(0, 0)));
    distanceMetric.addItem("Earth Mover (TH)", QVariant(QPoint(0, 1)));
    distanceMetric.addItem("Hellinger (TH)", QVariant(QPoint(0, 2)));
    distanceMetric.addItem("Euclidean (LISA)", QVariant(QPoint(1, 3)));
    distanceMetric.addItem("Euclidean (GC)", QVariant(QPoint(2, 3)));
    distanceMetric.addItem("Euclidean (PC)", QVariant(QPoint(3, 4)));
    distanceMetric.setToolTip("TH: Texture Histogram (vector feature) \nLISA: Local Indicator of Spatial Association (scalar feature) \n GC: local Geary's C\n PC: Point Collection Distance (no feature)");

    kernelWeight.addItem("Uniform");
    kernelWeight.addItem("Binomial");
    kernelWeight.addItem("Gaussian");

    histBinSizeHeur.addItem("Manual");  
    histBinSizeHeur.addItem("Sqrt");
    histBinSizeHeur.addItem("Sturges");
    histBinSizeHeur.addItem("Rice");
    histBinSizeHeur.setToolTip("Sqrt: ceil(sqrt(n)) \nSturges: ceil(log_2(n))+1 \nRice: ceil(2*pow(n, 1/3))");

    connect(&dataOptions,   SIGNAL(currentIndexChanged(QString)), this, SIGNAL(dataSetPicked(QString)));
    
    connect(&distanceMetric, SIGNAL(currentIndexChanged(int)), this, SLOT(onDistanceMetricPicked(int)));

    // as the kernel changes, the histogram bin number might change if it is not manually set
    connect(&kernelSize, &QSpinBox::textChanged, this, &SpidrSettingsWidget::onKernelSizeChanged);
    // change the hist bin size heuristic
    connect(&histBinSizeHeur, SIGNAL(currentIndexChanged(int)), this, SLOT(onHistBinSizeHeurPicked(int)));

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
    QLabel* knnAlgorithmLabel = new QLabel("KNN Approx.");
    QLabel* distanceMetricLabel = new QLabel("KNN Distance Metric");
    QLabel* exaggerationLabel = new QLabel("Exaggeration");
    QLabel* expDecayLabel = new QLabel("Exponential Decay");
    QLabel* numTreesLabel = new QLabel("Number of Trees");
    QLabel* numChecksLabel = new QLabel("Number of Checks");

    QLabel* kernelWeightLabel = new QLabel("Kernel Weighting");
    QLabel* kernelSizeLabel = new QLabel("Kernel Size");
    QLabel* histBinSizeHeurLabel = new QLabel("Histo. Bin Heuristic");
    QLabel* histBinSizeLabel = new QLabel("Number Bins");

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
    kernelSize.setRange(1, 10000);
    histBinSize.setRange(1, 10000);

    numIterations.setText("1000");
    perplexity.setText("30");
    exaggeration.setText("250");
    expDecay.setText("70");
    numTrees.setText("4");
    numChecks.setText("1024");
    kernelSize.setValue(1);
    histBinSize.setValue(5);

    startButton.setText("Start Computation");
    startButton.setFixedSize(QSize(150, 50));
    startButton.setCheckable(true);

    // Add options to their appropriate group box
    auto* const settingsLayout = new QGridLayout();

    settingsLayout->addWidget(knnAlgorithmLabel, 0, 0);
    settingsLayout->addWidget(&knnOptions, 1, 0);

    settingsLayout->addWidget(distanceMetricLabel, 0, 1);
    settingsLayout->addWidget(&distanceMetric, 1, 1);
    
    settingsLayout->addWidget(kernelWeightLabel, 2, 0);
    settingsLayout->addWidget(&kernelWeight, 3, 0);

    settingsLayout->addWidget(kernelSizeLabel, 2, 1);
    settingsLayout->addWidget(&kernelSize, 3, 1);

    settingsLayout->addWidget(histBinSizeHeurLabel, 4, 0);
    settingsLayout->addWidget(&histBinSizeHeur, 5, 0);

    settingsLayout->addWidget(histBinSizeLabel, 4, 1);
    settingsLayout->addWidget(&histBinSize, 5, 1);

    settingsLayout->addWidget(iterationLabel, 6, 0);
    settingsLayout->addWidget(&numIterations, 7, 0);
    
    settingsLayout->addWidget(perplexityLabel, 6, 1);
    settingsLayout->addWidget(&perplexity, 7, 1);    
        
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

void SpidrSettingsWidget::onKernelSizeChanged(const QString &kernelSizeField) {
    int activeHeur = histBinSizeHeur.currentIndex();

    if (activeHeur == 0) 
        return;
    else  {
        const int kernelSize_ = kernelSizeField.toInt();
        const int numLocNeighbors = (2 * kernelSize_ + 1) * (2 * kernelSize_ + 1);
        int binNum = 0;
        switch (activeHeur)
        {
        case 1: binNum = SqrtBinSize(numLocNeighbors); break;
        case 2: binNum = SturgesBinSize(numLocNeighbors); break;
        case 3: binNum = RiceBinSize(numLocNeighbors); break;
        default:
            break;
        }
        histBinSize.setValue(binNum);
        histBinSize.setReadOnly(true);

    }

}

void SpidrSettingsWidget::onDistanceMetricPicked(int value) {
    
    // if the metric works on vector features
    // provide options for the vector size
    // also, check if neighborhood weighting is 
    // available for the specific feature
    if (value >= 5) {   
        // PCD, no features
        histBinSizeHeur.setEnabled(false);
        histBinSize.setEnabled(false);
    }
    else if (value >= 3) {   
        // LISA, GC, scalar features
        histBinSizeHeur.setEnabled(false);
        histBinSize.setEnabled(false);
    }
    else {
        histBinSizeHeur.setEnabled(true);
        histBinSize.setEnabled(true);
    }
}

void SpidrSettingsWidget::onHistBinSizeHeurPicked(int value) {

    if (value == 0) {
        histBinSize.setReadOnly(false);
    }
    else if (value > 0) {
        int kernelSize_ = kernelSize.text().toInt();
        int numLocNeighbors = (2 * kernelSize_ + 1) * (2 * kernelSize_ + 1);
        switch (value)
        {
        case 1: histBinSize.setValue(SqrtBinSize(numLocNeighbors)); break;
        case 2: histBinSize.setValue(SturgesBinSize(numLocNeighbors)); break;
        case 3: histBinSize.setValue(RiceBinSize(numLocNeighbors)); break;
        default:
            break;
        }
        histBinSize.setReadOnly(true);
    }

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
