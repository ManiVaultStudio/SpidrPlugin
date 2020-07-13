#include "SpidrPlugin.h"
#include "SpidrSettingsWidget.h"

#include <QtCore>
#include <QSize>
#include <QtDebug>

#include <utility>      // std::as_const
#include <vector>       // std::vector
#include <windows.h>
Q_PLUGIN_METADATA(IID "nl.tudelft.SpidrPlugin")
#include <set>

// =============================================================================
// View
// =============================================================================

using namespace hdps;
SpidrPlugin::SpidrPlugin()
:
AnalysisPlugin("Spidr")
{
}

SpidrPlugin::~SpidrPlugin(void)
{
    stopComputation();
}

void SpidrPlugin::init()
{
    _settings = std::make_unique<SpidrSettingsWidget>(*this);

    // Connet settings
    connect(_settings.get(), &SpidrSettingsWidget::dataSetPicked, this, &SpidrPlugin::dataSetPicked);
    connect(_settings.get(), &SpidrSettingsWidget::knnAlgorithmPicked, this, &SpidrPlugin::onKnnAlgorithmPicked);
    connect(_settings.get(), &SpidrSettingsWidget::distanceMetricPicked, this, &SpidrPlugin::onDistanceMetricPicked);
    connect(_settings.get(), &SpidrSettingsWidget::kernelWeightPicked, this, &SpidrPlugin::onkernelWeightPicked);

    // Connect embedding
    connect(&_spidrAnalysis, &SpidrAnalysis::embeddingComputationStopped, _settings.get(), &SpidrSettingsWidget::computationStopped);
    connect(&_spidrAnalysis, &SpidrAnalysis::newEmbedding, this, &SpidrPlugin::onNewEmbedding);
}

void SpidrPlugin::dataAdded(const QString name)
{
    _settings->dataOptions.addItem(name);
}

void SpidrPlugin::dataChanged(const QString name)
{
    if (name != _settings->currentData()) {
        return;
    }

    Points& points = _core->requestData<Points>(name);

    _settings->dataChanged(points);
}

void SpidrPlugin::dataRemoved(const QString name)
{

}

void SpidrPlugin::selectionChanged(const QString dataName)
{

}


DataTypes SpidrPlugin::supportedDataTypes() const
{
    DataTypes supportedTypes;
    supportedTypes.append(PointType);
    return supportedTypes;
}

SettingsWidget* const SpidrPlugin::getSettings()
{
    return _settings.get();
}

void SpidrPlugin::dataSetPicked(const QString& name)
{
    Points& points = _core->requestData<Points>(name);

    _settings->dataChanged(points);
}

void SpidrPlugin::startComputation()
{
    // Get the data
    QString dataName = _settings->dataOptions.currentText();

    qDebug() << "SpidrPlugin: Read data.";

    std::vector<unsigned int> pointIDsGlobal;
    std::vector<float> attribute_data;        // Create list of data from the enabled dimensions
    QSize imgSize;
    unsigned int numDims;
    retrieveData(dataName, pointIDsGlobal, attribute_data, numDims, imgSize);

    qDebug() << "SpidrPlugin: Num data points: " << pointIDsGlobal.size() << " Num dims: " << numDims << " Image size (width, height): " << imgSize.width() << ", " << imgSize.height();

    // Create a new data set and hand it to the hdps core
    _embeddingName = _core->createDerivedData("Points", "Embedding", dataName);
    Points& embedding = _core->requestData<Points>(_embeddingName);
    embedding.setData(nullptr, 0, 2);
    _core->notifyDataAdded(_embeddingName);

    qDebug() << "SpidrPlugin: Created new data set for embedding";

    // Setup worker classes
    _spidrAnalysis.setup(attribute_data, pointIDsGlobal, numDims, imgSize);
    initializeTsneSettings();
    qDebug() << "SpidrPlugin: Initialized t-SNE computation settings";

    // Start spatial analysis
    _spidrAnalysis.start();

}

void SpidrPlugin::retrieveData(QString dataName, std::vector<unsigned int>& pointIDsGlobal, std::vector<float>& attribute_data, unsigned int& numDims, QSize& imgSize) {
    // For now, only handle underived data until Points implementation 
    // provides functionality to seamlessly obtain global IDs from derived data
    Points& points = _core->requestData<Points>(dataName);
    if (points.isDerivedData())
        exit(-1);

    imgSize = points.getProperty("ImageSize", QSize()).toSize();

    std::vector<bool> enabledDimensions = _settings->getEnabledDimensions();

    // Get number of enabled dimensions
    numDims = count_if(enabledDimensions.begin(), enabledDimensions.end(), [](bool b) { return b; });

    // Get indices of selected points
    pointIDsGlobal = points.indices;
    // If points represent all data set, select them all
    if (points.isFull()) {
        std::vector<std::uint32_t> all(points.getNumPoints());
        std::iota(std::begin(all), std::end(all), 0);

        pointIDsGlobal = all;
    }

    // For all selected points, retrieve values from each dimension
    attribute_data.reserve(pointIDsGlobal.size() * numDims);
    for (const auto& pointId : pointIDsGlobal)
    {
        for (unsigned int dimensionId = 0; dimensionId < points.getNumDimensions(); dimensionId++)
        {
            if (enabledDimensions[dimensionId]) {
                const auto index = pointId * points.getNumDimensions() + dimensionId;
                attribute_data.push_back(points[index]);
            }
        }
    }

}

void SpidrPlugin::onKnnAlgorithmPicked(const int index)
{
    _spidrAnalysis.setKnnAlgorithm(index);
}

void SpidrPlugin::onDistanceMetricPicked(const int index)
{
    _spidrAnalysis.setDistanceMetric(index);
}

void SpidrPlugin::onkernelWeightPicked(const int index)
{
    _spidrAnalysis.setKernelWeight(index);
}

void SpidrPlugin::onNewEmbedding() {
    const std::vector<float>& outputData = _spidrAnalysis.output();
    Points& embedding = _core->requestData<Points>(_embeddingName);
    
    embedding.setData(outputData.data(), _spidrAnalysis.getNumPoints(), 2);

    _core->notifyDataChanged(_embeddingName);
}

void SpidrPlugin::initializeTsneSettings() {
    
    // Initialize the tSNE computation with the settings from the settings widget
    _spidrAnalysis.initializeTsneSettings(_settings->numIterations.text().toInt(), \
                                          _settings->perplexity.text().toInt(), \
                                          _settings->exaggeration.text().toInt());
}

void SpidrPlugin::stopComputation() {
    // Request interruption of the computation
    if (_spidrAnalysis.isRunning())
    {
        // release openGL context 
        _spidrAnalysis.stopComputation(); 
        _spidrAnalysis.exit();

        // Wait until the thread has terminated (max. 3 seconds)
        if (!_spidrAnalysis.wait(3000))
        {
            qDebug() << "Spatial Analysis computation thread did not close in time, terminating...";
            _spidrAnalysis.terminate();
            _spidrAnalysis.wait();
        }
        qDebug() << "Spatial Analysis computation stopped.";

    }
}

// =============================================================================
// Factory
// =============================================================================

AnalysisPlugin* SpidrPluginFactory::produce()
{
    return new SpidrPlugin();
}
