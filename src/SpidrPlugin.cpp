#include "SpidrPlugin.h"
#include "SpidrSettingsWidget.h"

#include <QtCore>
#include <QSize>
#include <QtDebug>

#include <utility>      // std::as_const
#include <vector>       // std::vector

//#include <windows.h>
Q_PLUGIN_METADATA(IID "nl.tudelft.SpidrPlugin")
#include <set>

// =============================================================================
// View
// =============================================================================

using namespace hdps;
SpidrPlugin::SpidrPlugin()
:
AnalysisPlugin("Spidr"),
_spidrAnalysis(this)
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

    // Connect embedding
    connect(&_spidrAnalysis, &SpidrAnalysis::newEmbedding, this, &SpidrPlugin::onNewEmbedding);
    connect(&_spidrAnalysis, &SpidrAnalysis::finishedEmbedding, this, &SpidrPlugin::onFinishedEmbedding);
    //connect(this, &SpidrPlugin::embeddingComputationStopped, _settings.get(), &SpidrSettingsWidget::computationStopped);
}

void SpidrPlugin::dataAdded(const QString name)
{
    // For now, only handle underived data until Points implementation 
    // provides functionality to seamlessly obtain global IDs from derived data
    Points& points = _core->requestData<Points>(name);
    if (points.isDerivedData())
        return;
    // Only accept valid image data
    QSize imageSize = points.getProperty("ImageSize", QSize()).toSize();
    if ((imageSize.height() <= 0) || (imageSize.width() <= 0))
        return;

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
    // Currently, data sets cannot be removed through the UI at this moment
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

hdps::gui::SettingsWidget* const SpidrPlugin::getSettings()
{
    return _settings.get();
}

void SpidrPlugin::dataSetPicked(const QString& name)
{
    Points& points = _core->requestData<Points>(name);
    _settings->dataChanged(points);

    _settings->setTitle(QString("%1: %2").arg(getGuiName(), name));
}

void SpidrPlugin::startComputation()
{
    // Get the data
    qDebug() << "SpidrPlugin: Read data ";

    std::vector<unsigned int> pointIDsGlobal; // Global ID of each point in the image
    std::vector<float> attribute_data;        // Actual channel valures, only consider enabled dimensions
    std::vector<unsigned int> backgroundIDsGlobal;  // ID of points which are not used during the t-SNE embedding - but will inform the feature extraction and distance calculation
    QSize imgSize;
    unsigned int numDims;
    QString dataName = _settings->dataOptions.currentText();
    retrieveData(dataName, pointIDsGlobal, attribute_data, numDims, imgSize, backgroundIDsGlobal);

    // Create a new data set and hand it to the hdps core
    qDebug() << "SpidrPlugin: Create new data set for embedding";

    _embeddingName = _core->createDerivedData("Points", _settings->getEmbName(), dataName);
    // _embeddingName = _core->addData("Points", _settings->getEmbName());
    Points& embedding = _core->requestData<Points>(_embeddingName);
    embedding.setData(nullptr, 0, 2);
    _core->notifyDataAdded(_embeddingName);

    // Setup worker classes with data and parameters
    qDebug() << "SpidrPlugin: Initialize settings";

    _spidrAnalysis.setupData(attribute_data, pointIDsGlobal, numDims, imgSize, _embeddingName, backgroundIDsGlobal);
    initializeAnalysisSettings();

    // Start spatial analysis
    _spidrAnalysis.start();

}

void SpidrPlugin::retrieveData(QString dataName, std::vector<unsigned int>& pointIDsGlobal, std::vector<float>& attribute_data, unsigned int& numEnabledDimensions, QSize& imgSize, std::vector<unsigned int>& backgroundIDsGlobal) {
    Points& points = _core->requestData<Points>(dataName);
    imgSize = points.getProperty("ImageSize", QSize()).toSize();

    std::vector<bool> enabledDimensions = _settings->getEnabledDimensions();

    // Get number of enabled dimensions
    unsigned int numDimensions = points.getNumDimensions();
    numEnabledDimensions = count_if(enabledDimensions.begin(), enabledDimensions.end(), [](bool b) { return b; });

    // Get indices of selected points
    pointIDsGlobal = points.indices;
    // If points represent all data set, select them all
    if (points.isFull()) {
        std::vector<unsigned int> all(points.getNumPoints());
        std::iota(std::begin(all), std::end(all), 0);

        pointIDsGlobal = all;
    }

    // For all selected points, retrieve values from each dimension
    attribute_data.reserve(pointIDsGlobal.size() * numEnabledDimensions);
    
    points.visitFromBeginToEnd([&attribute_data, &pointIDsGlobal, &enabledDimensions, &numDimensions](auto beginOfData, auto endOfData)
    {
        for (const auto& pointId : pointIDsGlobal)
        {
            for (unsigned int dimensionId = 0; dimensionId < numDimensions; dimensionId++)
            {
                if (enabledDimensions[dimensionId]) {
                    const auto index = pointId * numDimensions + dimensionId;
                    attribute_data.push_back(beginOfData[index]);
                }
            }
        }
    });

    // If a background data set is given, store the background indices
    QString backgroundName = _settings->backgroundNameLine.text();
    if (!backgroundName.isEmpty()) {
        Points& backgroundPoints = _core->requestData<Points>(backgroundName);

        if (_settings->backgroundFromData.isChecked())
        {
            qDebug() << "SpidrPlugin: Read background from data set " << backgroundName << " (using the data set values)";
            auto totalNumPoints = backgroundPoints.getNumPoints();
            backgroundIDsGlobal.clear();
            backgroundIDsGlobal.reserve(totalNumPoints);
            backgroundPoints.visitFromBeginToEnd([&backgroundIDsGlobal, totalNumPoints](auto beginOfData, auto endOfData)
            {
                for (unsigned int i = 0; i < totalNumPoints; i++)
                {
                    backgroundIDsGlobal.push_back(beginOfData[i]);
                }
            });
        }
        else
        {
            qDebug() << "SpidrPlugin: Read background from data set " << backgroundName << " (using the data set indices)";
            backgroundIDsGlobal = backgroundPoints.indices;
        }
    }

}


void SpidrPlugin::onNewEmbedding() {
    const std::vector<float>& outputData = _spidrAnalysis.output();
    Points& embedding = _core->requestData<Points>(_embeddingName);
    
    embedding.setData(outputData.data(), _spidrAnalysis.getNumEmbPoints(), 2);

    _core->notifyDataChanged(_embeddingName);
}

void SpidrPlugin::onFinishedEmbedding() {
    const std::vector<float>& outputData = _spidrAnalysis.outputWithBackground();

    assert(outputData.size() % 2 == 0);
    assert(outputData.size() == _spidrAnalysis.getNumImagePoints() * 2);

    qDebug() << "SpidrPlugin: Publishing final embedding";

    Points& embedding = _core->requestData<Points>(_embeddingName);
    embedding.setData(outputData.data(), _spidrAnalysis.getNumImagePoints(), 2);
    _core->notifyDataChanged(_embeddingName);

    _settings.get()->computationStopped();

    qDebug() << "SpidrPlugin: Done.";
}


void SpidrPlugin::initializeAnalysisSettings() {
    // set all the parameters
    _spidrAnalysis.initializeAnalysisSettings(_settings->distanceMetric.currentData().toPoint().x(), _settings->kernelWeight.currentData().toInt(), _settings->kernelSize.text().toInt(),  \
                                              _settings->histBinSize.text().toInt(), _settings->knnOptions.currentData().toInt(), _settings->distanceMetric.currentData().toPoint().y(), \
                                              _settings->numIterations.text().toInt(), _settings->perplexity.text().toInt(), _settings->exaggeration.text().toInt());
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
