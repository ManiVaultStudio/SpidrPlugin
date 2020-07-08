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
    _params = Parameters();
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

    // Connect feature extraction

    // Connect embedding
    connect(&_tsne, &TsneAnalysis::computationStopped, _settings.get(), &SpidrSettingsWidget::computationStopped);
    connect(&_tsne, SIGNAL(newEmbedding()), this, SLOT(onNewEmbedding()));
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

    QSize imgSize;
    std::vector<unsigned int> pointIDsGlobal;
    std::vector<float> data;        // Create list of data from the enabled dimensions
    retrieveData(dataName, imgSize, pointIDsGlobal, data, _params);

    //// Extract features
    _featExtraction.setupData(imgSize, pointIDsGlobal, data, _params);
    _featExtraction.start();
    std::vector<float>* histoFeats = _featExtraction.output();

    // Caclculate distances and kNN
    _distCalc.setupData(histoFeats, _params);
    _distCalc.start();
    const std::vector<int>* indices = _distCalc.get_knn_indices();
    const std::vector<float>* distances_squared = _distCalc.get_knn_distances_squared();

    // Embedding
    // First, create data set and hand it to the hdps core
    _embeddingName = _core->createDerivedData("Points", "Embedding", dataName);
    Points& embedding = _core->requestData<Points>(_embeddingName);
    embedding.setData(nullptr, 0, 2);
    _core->notifyDataAdded(_embeddingName);

    qDebug() << "Created new data set for embedding";

    // Second, compute t-SNE with the given data
    initializeTsneSettings();
    _tsne.initTSNE(indices, distances_squared, _params);    // TODO: change to use kNN 
    _tsne.start();
}

void SpidrPlugin::retrieveData(QString dataName, QSize& imgSize, std::vector<unsigned int>& pointIDsGlobal, std::vector<float>& data, Parameters& params) {
    // For now, only handle underived data until Points implementation 
    // provides functionality to seamlessly obtain global IDs from derived data
    Points& points = _core->requestData<Points>(dataName);
    if (points.isDerivedData())
        exit(-1);

    imgSize = points.getProperty("ImageSize", QSize()).toSize();

    std::vector<bool> enabledDimensions = _settings->getEnabledDimensions();

    // Get number of enabled dimensions
    unsigned int numDimensions = count_if(enabledDimensions.begin(), enabledDimensions.end(), [](bool b) { return b; });

    // Get indices of selected points
    pointIDsGlobal = points.indices;
    // If points represent all data set, select them all
    if (points.isFull()) {
        std::vector<std::uint32_t> all(points.getNumPoints());
        std::iota(std::begin(all), std::end(all), 0);

        pointIDsGlobal = all;
    }

    // For all selected points, retrieve values from each dimension
    data.reserve(pointIDsGlobal.size() * numDimensions);
    for (const auto& pointId : pointIDsGlobal)
    {
        for (unsigned int dimensionId = 0; dimensionId < points.getNumDimensions(); dimensionId++)
        {
            if (enabledDimensions[dimensionId]) {
                const auto index = pointId * points.getNumDimensions() + dimensionId;
                data.push_back(points[index]);
            }
        }
    }

    // Set Parameters
    _params._numPoints  = pointIDsGlobal.size();
    _params._numDims = numDimensions;

    qDebug() << "SpidrPlugin: Read data.";
    qDebug() << "SpidrPlugin. Num data points: " << _params._numPoints << " Num dims: " << _params._numDims << " Image size (width, height): " << imgSize.width() << ", " << imgSize.height();

}

void SpidrPlugin::onKnnAlgorithmPicked(const int index)
{
    _distCalc.setKnnAlgorithm(index);
}

void SpidrPlugin::onDistanceMetricPicked(const int index)
{
    _distCalc.setDistanceMetric(index);
}


void SpidrPlugin::onNewEmbedding() {
    const std::vector<float>& outputData = _tsne.output();
    Points& embedding = _core->requestData<Points>(_embeddingName);
    
    embedding.setData(outputData.data(), _params._numPoints, 2);

    _core->notifyDataChanged(_embeddingName);
}

void SpidrPlugin::initializeTsneSettings() {
    
    // Initialize the tSNE computation with the settings from the settings widget
    _tsne.setIterations(_settings->numIterations.text().toInt());
    _tsne.setPerplexity(_settings->perplexity.text().toInt());
    _tsne.setExaggerationIter(_settings->exaggeration.text().toInt());

    qDebug() << "t-SNE computation settings: perplexity " << _tsne.perplexity() << ", iterations " << _tsne.iterations();
}

void SpidrPlugin::stopComputation() {
    if (_tsne.isRunning())
    {
        // Request interruption of the computation
        _tsne.stopGradientDescent();
        _tsne.exit();

        // Wait until the thread has terminated (max. 3 seconds)
        if (!_tsne.wait(3000))
        {
            qDebug() << "tSNE computation thread did not close in time, terminating...";
            _tsne.terminate();
            _tsne.wait();
        }
        qDebug() << "tSNE computation stopped.";
    }
}

// =============================================================================
// Factory
// =============================================================================

AnalysisPlugin* SpidrPluginFactory::produce()
{
    return new SpidrPlugin();
}
