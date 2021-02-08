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

SpidrSettingsWidget::SpidrSettingsWidget(SpidrPlugin& analysisPlugin) :
    SettingsWidget(),
    _analysisPlugin(analysisPlugin),
    backgroundNameLine(""), embNameLine("")
{
    const auto guiName = analysisPlugin.getGuiName();
    setObjectName(guiName);
    setIcon(hdps::Application::getIconFont("FontAwesome").getIcon("border-none"));
    setTitle(guiName);
    setSubtitle("");

    const auto minimumWidth = 200;
    setMinimumWidth(minimumWidth);
    setMaximumWidth(2 * minimumWidth);
    
    // add data item according to enum knn_library (KNNUtils)
    knnOptions.addItem("HNSW", static_cast<unsigned int> (knn_library::KNN_HNSW));
    knnOptions.addItem("Exact", static_cast<unsigned int> (knn_library::EXACT));
    knnOptions.addItem("Eval Full", static_cast<unsigned int> (knn_library::EVAL_EXACT));
    knnOptions.addItem("Eval akNN", static_cast<unsigned int> (knn_library::EVAL_KNN));
    knnOptions.setToolTip("HNSW: Approximate kNN (fast) \nExact: precise (slow) \nEval Full: precise and saves (all+kNN) indices&distances and features to disk (slow) \nEval Full: Like Eval Full but for akNN (fast)");

    // data values (QVariant) store feature_type (FeatureUtils) and distance_metric (KNNUtils) values as x and y 
    // this is used as a nice way to cast this information internally in SpidrAnalysis
    distanceMetric.addItem("Texture Hist. (QF)", MakeMetricPair(feature_type::TEXTURE_HIST_1D, distance_metric::METRIC_QF));
    distanceMetric.addItem("Texture Hist. (EMD)", MakeMetricPair(feature_type::TEXTURE_HIST_1D, distance_metric::METRIC_EMD));
    distanceMetric.addItem("Texture Hist. (Hel)", MakeMetricPair(feature_type::TEXTURE_HIST_1D, distance_metric::METRIC_HEL));
    distanceMetric.addItem("Local Moran's I (L2)", MakeMetricPair(feature_type::LOCALMORANSI, distance_metric::METRIC_EUC));
    distanceMetric.addItem("Local Geary's C (L2)", MakeMetricPair(feature_type::LOCALGEARYC, distance_metric::METRIC_EUC));
    distanceMetric.addItem("Point Clound (Chamfer)", MakeMetricPair(feature_type::PCLOUD, distance_metric::METRIC_CHA));
    distanceMetric.addItem("Point Clound (SSD)", MakeMetricPair(feature_type::PCLOUD, distance_metric::METRIC_SSD));
    distanceMetric.addItem("Point Clound (Hausdorff)", MakeMetricPair(feature_type::PCLOUD, distance_metric::METRIC_HAU));
    distanceMetric.addItem("MVN (Attr./Spatial)", MakeMetricPair(feature_type::MVN, distance_metric::METRIC_MVN));
    distanceMetric.addItem("Hausdorff (Min)", MakeMetricPair(feature_type::PCLOUD, distance_metric::METRIC_HAU_min));
    distanceMetric.addItem("Hausdorff (Median)", MakeMetricPair(feature_type::PCLOUD, distance_metric::METRIC_HAU_med));
    distanceMetric.addItem("Hausdorff (MedianMedian)", MakeMetricPair(feature_type::PCLOUD, distance_metric::METRIC_HAU_medmed));
    distanceMetric.addItem("Hausdorff (MinMax)", MakeMetricPair(feature_type::PCLOUD, distance_metric::METRIC_HAU_minmax));
    distanceMetric.setToolTip("Vector feature: Texture histograms \nScalar features: Local indicators of spatial association (Local I and C) \nNo feature: Point Cloud (Chamfer distance, Sum of Squared differences, Hausdorff distance) \nMVN-Reduce (Combination of Spatial and Attribute distance)");

    // add data item according to enum loc_Neigh_Weighting (FeatureUtils)
    kernelWeight.addItem("Uniform", static_cast<unsigned int> (loc_Neigh_Weighting::WEIGHT_UNIF));
    kernelWeight.addItem("Binomial", static_cast<unsigned int> (loc_Neigh_Weighting::WEIGHT_BINO));
    kernelWeight.addItem("Gaussian", static_cast<unsigned int> (loc_Neigh_Weighting::WEIGHT_GAUS));

    // add data item according to enum histBinSizeHeuristic (FeatureUtils)
    histBinSizeHeur.addItem("Manual", static_cast<unsigned int> (histBinSizeHeuristic::MANUAL));
    histBinSizeHeur.addItem("Sqrt", static_cast<unsigned int> (histBinSizeHeuristic::SQRT));
    histBinSizeHeur.addItem("Sturges", static_cast<unsigned int> (histBinSizeHeuristic::STURGES));
    histBinSizeHeur.addItem("Rice", static_cast<unsigned int> (histBinSizeHeuristic::RICE));
    histBinSizeHeur.setToolTip("Sqrt: ceil(sqrt(n)) \nSturges: ceil(log_2(n))+1 \nRice: ceil(2*pow(n, 1/3))");

    // Initialize data options
    connect(&dataOptions,   SIGNAL(currentIndexChanged(QString)), this, SIGNAL(dataSetPicked(QString)));
    // Initialize distance metric options
    connect(&distanceMetric, SIGNAL(currentIndexChanged(int)), this, SLOT(onDistanceMetricPicked(int)));
    // Set embedding default name
    connect(&dataOptions, SIGNAL(currentIndexChanged(QString)), this, SLOT(setEmbName(QString)));
    // as the kernel changes, the histogram bin number might change if it is not manually set
    connect(&kernelSize, &QSpinBox::textChanged, this, &SpidrSettingsWidget::onKernelSizeChanged);
    // change the hist bin size heuristic
    connect(&histBinSizeHeur, SIGNAL(currentIndexChanged(int)), this, SLOT(onHistBinSizeHeurPicked(int)));
    // connect weight slider and spin box
    connect(&weightSpaAttrSlider, &QSlider::valueChanged, [this](const int& val) {weightSpaAttrNum.setValue(double(val)/100);});
    connect(&weightSpaAttrNum, QOverload<double>::of(&QDoubleSpinBox::valueChanged), [this](const double& val) {weightSpaAttrSlider.setValue(val * 100); });

    connect(&numIterations, SIGNAL(textChanged(QString)), SLOT(numIterationsChanged(QString)));
    connect(&perplexity,    SIGNAL(textChanged(QString)), SLOT(perplexityChanged(QString)));
    connect(&exaggeration,  SIGNAL(textChanged(QString)), SLOT(exaggerationChanged(QString)));
    connect(&expDecay,      SIGNAL(textChanged(QString)), SLOT(expDecayChanged(QString)));
    connect(&numTrees,      SIGNAL(textChanged(QString)), SLOT(numTreesChanged(QString)));
    connect(&numChecks,     SIGNAL(textChanged(QString)), SLOT(numChecksChanged(QString)));
    connect(&theta,         SIGNAL(textChanged(QString)), SLOT(thetaChanged(QString)));

    // Initialize start button
    startButton.setText("Start Computation");
    startButton.setFixedSize(QSize(150, 50));
    startButton.setCheckable(true);
    connect(&startButton, &QPushButton::toggled, this, &SpidrSettingsWidget::onStartToggled);

    // Create group boxes for grouping together various settings
    QGroupBox* settingsBox = new QGroupBox("Basic settings");
    QGroupBox* advancedSettingsBox = new QGroupBox("Advanced Settings");
    QGroupBox* computeBox = new QGroupBox();

    advancedSettingsBox->setCheckable(true);
    advancedSettingsBox->setChecked(false);
    
    // Build the labels for all the options
    QLabel* iterationLabel = new QLabel("Iteration Count");
    QLabel* perplexityLabel = new QLabel("Perplexity");
    QLabel* knnAlgorithmLabel = new QLabel("KNN Calculation");
    QLabel* distanceMetricLabel = new QLabel("Distance Measure");
    QLabel* exaggerationLabel = new QLabel("Exaggeration");
    QLabel* expDecayLabel = new QLabel("Exponential Decay");
    QLabel* numTreesLabel = new QLabel("Number of Trees");
    QLabel* numChecksLabel = new QLabel("Number of Checks");
    QLabel* embNameLabel = new QLabel("Embedding Name");
    QLabel* backgroundNameLabel = new QLabel("Background Dataset Name");
    QLabel* backgroundTickLabel = new QLabel("Background Indices In Data");

    QLabel* kernelWeightLabel = new QLabel("Kernel Weighting");
    QLabel* kernelSizeLabel = new QLabel("Kernel Size");
    QLabel* histBinSizeHeurLabel = new QLabel("Histo. Bin Heuristic");
    QLabel* histBinSizeLabel = new QLabel("Number Bins");

    QLabel* weightSpAttrLabel = new QLabel("MVN weight");
    weightSpAttrLabel->setToolTip("Weight Attribute (0) vs Spatial (1)");
    
    // Set option default values
    numIterations.setFixedWidth(50);
    perplexity.setFixedWidth(50);
    exaggeration.setFixedWidth(50);
    expDecay.setFixedWidth(50);
    numTrees.setFixedWidth(50);
    numChecks.setFixedWidth(50);
    kernelSize.setFixedWidth(50);
    histBinSize.setFixedWidth(50);
    weightSpaAttrSlider.setFixedWidth(50);
    weightSpaAttrNum.setFixedWidth(50);

    numIterations.setValidator(new QIntValidator(1, 10000, this));
    perplexity.setValidator(new QIntValidator(2, 90, this));
    exaggeration.setValidator(new QIntValidator(1, 10000, this));
    expDecay.setValidator(new QIntValidator(1, 10000, this));
    numTrees.setValidator(new QIntValidator(1, 10000, this));
    numChecks.setValidator(new QIntValidator(1, 10000, this));
    kernelSize.setRange(1, 10000);
    histBinSize.setRange(1, 10000);
    weightSpaAttrNum.setRange(0, 1);

    numIterations.setText("1000");
    perplexity.setText("30");
    exaggeration.setText("250");
    expDecay.setText("70");
    numTrees.setText("4");
    numChecks.setText("1024");
    kernelSize.setValue(1);
    histBinSize.setValue(5);

    weightSpaAttrSlider.setRange(0, 100);
    weightSpaAttrSlider.setSingleStep(1);
    weightSpaAttrSlider.setOrientation(Qt::Horizontal);

    weightSpaAttrNum.setDecimals(2);
    weightSpaAttrNum.setSingleStep(0.01);

    weightSpaAttrSlider.setEnabled(false);
    weightSpaAttrNum.setEnabled(false);

    // Add options to their appropriate group box
    auto* const settingsLayout = new QGridLayout();

    settingsLayout->addWidget(knnAlgorithmLabel, 0, 0);
    settingsLayout->addWidget(&knnOptions, 1, 0);

    settingsLayout->addWidget(distanceMetricLabel, 0, 1, 1, 3);  // (widget, row, col, rowSpan, colSpan)
    settingsLayout->addWidget(&distanceMetric, 1, 1, 1, 3);
    
    settingsLayout->addWidget(kernelWeightLabel, 2, 0);
    settingsLayout->addWidget(&kernelWeight, 3, 0);

    settingsLayout->addWidget(kernelSizeLabel, 2, 1);
    settingsLayout->addWidget(&kernelSize, 3, 1);

    settingsLayout->addWidget(weightSpAttrLabel, 2, 2);
    settingsLayout->addWidget(&weightSpaAttrSlider, 3, 2);
    settingsLayout->addWidget(&weightSpaAttrNum, 3, 3);

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
    advancedSettingsLayout->addWidget(backgroundNameLabel, 4, 0);
    advancedSettingsLayout->addWidget(&backgroundNameLine, 5, 0, 1, 2);
    advancedSettingsLayout->addWidget(backgroundTickLabel, 6, 0);
    advancedSettingsLayout->addWidget(&backgroundFromData, 6, 1);
    advancedSettingsBox->setLayout(advancedSettingsLayout);

    
    auto* const computeLayout = new QGridLayout();
    computeLayout->addWidget(embNameLabel, 0, 0);
    computeLayout->addWidget(&embNameLine, 1, 0, Qt::AlignTop);
    computeLayout->addWidget(&startButton, 0, 1, 2, 1, Qt::AlignCenter);
    computeBox->setLayout(computeLayout);

    // Add all the parts of the settings widget together
    addWidget(&dataOptions);
    addWidget(settingsBox);
    addWidget(&_dimensionSelectionWidget);
    addWidget(advancedSettingsBox);
    addWidget(computeBox);

}

void SpidrSettingsWidget::setEmbName(QString embName)
{
    embNameLine.setText(embName + "_sp-tsne_emb");
}

QString SpidrSettingsWidget::getEmbName()
{
    return embNameLine.text();
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
    histBinSizeHeuristic activeHeur = static_cast<histBinSizeHeuristic> (histBinSizeHeur.currentData().value<unsigned int>());

    if (activeHeur == histBinSizeHeuristic::MANUAL)
        return;
    else  {
        const int kernelSize_ = kernelSizeField.toInt();
        const int numLocNeighbors = (2 * kernelSize_ + 1) * (2 * kernelSize_ + 1);
        int binNum = 0;
        switch (activeHeur)
        {
        case histBinSizeHeuristic::SQRT: binNum = SqrtBinSize(numLocNeighbors); break;
        case histBinSizeHeuristic::STURGES: binNum = SturgesBinSize(numLocNeighbors); break;
        case histBinSizeHeuristic::RICE: binNum = RiceBinSize(numLocNeighbors); break;
        default:
            break;
        }
        histBinSize.setValue(binNum);
        histBinSize.setReadOnly(true);

    }

}

void SpidrSettingsWidget::onDistanceMetricPicked(int distMetricBoxIndex) {
    distance_metric distMetric = GetDistMetricFromMetricPair(distanceMetric.itemData(distMetricBoxIndex));

    // if the metric works on vector features provide options for the vector size
    // also, check if neighborhood weighting is available for the specific feature
    if (!(distMetric == distance_metric::METRIC_EMD || distMetric == distance_metric::METRIC_HEL || distMetric == distance_metric::METRIC_QF)) {
        // only for histogram features 
        histBinSizeHeur.setEnabled(false);
        histBinSize.setEnabled(false);
    }
    else {
        histBinSizeHeur.setEnabled(true);
        histBinSize.setEnabled(true);
    }

    if (distMetric == distance_metric::METRIC_MVN) {
        // only for MVN
        weightSpaAttrSlider.setEnabled(true);
        weightSpaAttrNum.setEnabled(true);

        kernelSize.setEnabled(false);
        kernelWeight.setEnabled(false);
    }
    else {
        weightSpaAttrSlider.setEnabled(false);
        weightSpaAttrNum.setEnabled(false);

        kernelSize.setEnabled(true);
        kernelWeight.setEnabled(true);
    }
}

void SpidrSettingsWidget::onHistBinSizeHeurPicked(int value) {
    histBinSizeHeuristic heuristic = static_cast<histBinSizeHeuristic> (value);

    if (heuristic == histBinSizeHeuristic::MANUAL) {
        histBinSize.setReadOnly(false);
    }
    else {
        int kernelSize_ = kernelSize.text().toInt();
        int numLocNeighbors = (2 * kernelSize_ + 1) * (2 * kernelSize_ + 1);
        switch (heuristic)
        {
        case histBinSizeHeuristic::SQRT: histBinSize.setValue(SqrtBinSize(numLocNeighbors)); break;
        case histBinSizeHeuristic::STURGES: histBinSize.setValue(SturgesBinSize(numLocNeighbors)); break;
        case histBinSizeHeuristic::RICE: histBinSize.setValue(RiceBinSize(numLocNeighbors)); break;
        default:
            qDebug() << "SpidrSettingsWidget::onHistBinSizeHeurPicked: heuristic not implemented";
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
