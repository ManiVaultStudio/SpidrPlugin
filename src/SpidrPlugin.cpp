#include "SpidrPlugin.h"
#include "SpidrSettingsWidget.h"

#include <QtCore>
#include <QSize>
#include <QtDebug>

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

void SpidrPlugin::onKnnAlgorithmPicked(const int index)
{
    _tsne.setKnnAlgorithm(index);
}

void SpidrPlugin::onDistanceMetricPicked(const int index)
{
    _tsne.setDistanceMetric(index);
}

void SpidrPlugin::startComputation()
{
    // Get the data
    QString setName = _settings->dataOptions.currentText();
    const Points& points = _core->requestData<Points>(setName);

    QSize imgSize = points.getProperty("ImageSize", QSize()).toSize();
    unsigned int numDimensions;
    std::vector<float> data;        // Create list of data from the enabled dimensions
    retrieveData(points, numDimensions, data);

    //// Extract features
    _featExtraction.setupData(data, points.indices, numDimensions, imgSize);
    _featExtraction.start();
    std::vector<float> histoFeats = _featExtraction.output();

    // Caclculate distances and kNN

    // Embedding
    // First, create data set and hand it to the hdps core
    _embeddingName = _core->createDerivedData("Points", "Embedding", points.getName());
    Points& embedding = _core->requestData<Points>(_embeddingName);
    embedding.setData(nullptr, 0, 2);
    _core->notifyDataAdded(_embeddingName);

    // Second, compute t-SNE with the given data
    initializeTsne();
    _tsne.initTSNE(data, numDimensions);    // TODO: change to use kNN 

    _tsne.start();
}

void SpidrPlugin::retrieveData(const Points points, unsigned int& numDimensions, std::vector<float>& data) {
    std::vector<bool> enabledDimensions = _settings->getEnabledDimensions();

    // Get number of enabled dimensions
    numDimensions = count_if(enabledDimensions.begin(), enabledDimensions.end(), [](bool b) { return b; });

    // Get indices of selected points
    auto selection = points.indices;
    // If points represent all data set, select them all
    if (points.isFull()) {
        std::vector<std::uint32_t> all(points.getNumPoints());
        std::iota(std::begin(all), std::end(all), 0);

        selection = all;
    }

    // For all selected points, retrieve values from each dimension
    data.reserve(selection.size() * numDimensions);
    for (const auto& pointId : selection)
    {
        for (unsigned int dimensionId = 0; dimensionId < points.getNumDimensions(); dimensionId++)
        {
            if (enabledDimensions[dimensionId]) {
                const auto index = pointId * points.getNumDimensions() + dimensionId;
                data.push_back(points[index]);
            }
        }
    }

}

void SpidrPlugin::onNewEmbedding() {
    const TsneData& outputData = _tsne.output();
    Points& embedding = _core->requestData<Points>(_embeddingName);
    
    embedding.setData(outputData.getData().data(), outputData.getNumPoints(), 2);

    _core->notifyDataChanged(_embeddingName);
}

void SpidrPlugin::initializeTsne() {
    // Initialize the tSNE computation with the settings from the settings widget
    _tsne.setIterations(_settings->numIterations.text().toInt());
    _tsne.setPerplexity(_settings->perplexity.text().toInt());
    _tsne.setExaggerationIter(_settings->exaggeration.text().toInt());
    _tsne.setNumTrees(_settings->numTrees.text().toInt());
    _tsne.setNumChecks(_settings->numChecks.text().toInt());
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
