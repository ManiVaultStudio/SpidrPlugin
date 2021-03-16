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
    _spidrAnalysisQt(this)
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
    connect(&_spidrAnalysisQt, &SpidrAnalysisQt::newEmbedding, this, &SpidrPlugin::onNewEmbedding);
    connect(&_spidrAnalysisQt, &SpidrAnalysisQt::finishedEmbedding, this, &SpidrPlugin::onFinishedEmbedding);
    connect(&_spidrAnalysisQt, &SpidrAnalysisQt::publishFeatures, this, &SpidrPlugin::onPublishFeatures);
    //connect(this, &SpidrPlugin::embeddingComputationStopped, _settings.get(), &SpidrSettingsWidget::computationStopped);

    registerDataEventByType(PointType, std::bind(&SpidrPlugin::onDataEvent, this, std::placeholders::_1));

}

void SpidrPlugin::onDataEvent(hdps::DataEvent* dataEvent)
{
    if (dataEvent->getType() == EventType::DataAdded)
        _settings->addDataItem(static_cast<DataAddedEvent*>(dataEvent)->dataSetName);

    if (dataEvent->getType() == EventType::DataRemoved)
        _settings->removeDataItem(static_cast<DataRemovedEvent*>(dataEvent)->dataSetName);

    if (dataEvent->getType() == EventType::DataChanged) {
        auto dataChangedEvent = static_cast<DataChangedEvent*>(dataEvent);

        // If we are not looking at the changed dataset, ignore it
        if (dataChangedEvent->dataSetName != _settings->getCurrentDataItem())
            return;

        // Passes changes to the current dataset to the dimension selection widget
        Points& points = _core->requestData<Points>(dataChangedEvent->dataSetName);

        // Only handle underived data?
        //        if (points.isDerivedData())
        //            return;

        _settings->getDimensionSelectionWidget().dataChanged(points);
    }
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
    ImgSize imgSize;
    unsigned int numDims;
    QString dataName = _settings->getCurrentDataItem();
    retrieveData(dataName, pointIDsGlobal, attribute_data, numDims, imgSize, backgroundIDsGlobal);

    // Create a new data set and hand it to the hdps core
    qDebug() << "SpidrPlugin: Create new data set for embedding";
    _embeddingName = _core->createDerivedData(_settings->getEmbName(), dataName);
    Points& embedding = _core->requestData<Points>(_embeddingName);
    embedding.setData(nullptr, 0, 2);
    _core->notifyDataAdded(_embeddingName);

    // Setup worker classes with data and parameters
    qDebug() << "SpidrPlugin: Initialize settings";

    _spidrAnalysisQt.setupData(attribute_data, pointIDsGlobal, numDims, imgSize, _embeddingName, backgroundIDsGlobal);
    initializeAnalysisSettings();

    // Start spatial analysis
    _spidrAnalysisQt.start();

}

void SpidrPlugin::retrieveData(QString dataName, std::vector<unsigned int>& pointIDsGlobal, std::vector<float>& attribute_data, unsigned int& numEnabledDimensions, ImgSize& imgSize, std::vector<unsigned int>& backgroundIDsGlobal) {
    Points& points = _core->requestData<Points>(dataName);
    QSize qtImgSize = points.getProperty("ImageSize", QSize()).toSize();
    imgSize.width = qtImgSize.width();
    imgSize.height = qtImgSize.height();

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
    const std::vector<float>& outputData = _spidrAnalysisQt.output();
    Points& embedding = _core->requestData<Points>(_embeddingName);

    embedding.setData(outputData.data(), _spidrAnalysisQt.getNumEmbPoints(), 2);

    _core->notifyDataChanged(_embeddingName);
}

void SpidrPlugin::onFinishedEmbedding() {
    const std::vector<float>& outputData = _spidrAnalysisQt.outputWithBackground();

    assert(outputData.size() % 2 == 0);
    assert(outputData.size() == _spidrAnalysisQt.getNumImagePoints() * 2);

    qDebug() << "SpidrPlugin: Publishing final embedding";

    Points& embedding = _core->requestData<Points>(_embeddingName);
    embedding.setData(outputData.data(), _spidrAnalysisQt.getNumImagePoints(), 2);
    _core->notifyDataChanged(_embeddingName);

    _settings.get()->computationStopped();

    qDebug() << "SpidrPlugin: Done.";
}

void SpidrPlugin::onPublishFeatures() {
    qDebug() << "SpidrPlugin: Publish features to core";
    QString featureDataSetName = _core->createDerivedData(_settings->getEmbName() + "_Features", _settings->getCurrentDataItem());
    Points& featureDataSet = _core->requestData<Points>(featureDataSetName);
    featureDataSet.setData(_spidrAnalysisQt.getFeatures()->data(), _spidrAnalysisQt.getNumEmbPoints(), _spidrAnalysisQt.getNumFeatureValsPerPoint());
}


void SpidrPlugin::initializeAnalysisSettings() {
    // set all the parameters
    // TODO: use the strongly typed enum classes instead of all the int values
    _spidrAnalysisQt.initializeAnalysisSettings(_settings->distanceMetric.currentData().toPoint().x(), _settings->kernelWeight.currentData().value<unsigned int>(), _settings->kernelSize.text().toInt(), \
        _settings->histBinSize.text().toInt(), _settings->knnOptions.currentData().value<unsigned int>(), _settings->distanceMetric.currentData().toPoint().y(), \
        _settings->weightSpaAttrNum.value(), _settings->numIterations.text().toInt(), _settings->perplexity.text().toInt(), _settings->exaggeration.text().toInt(), _settings->expDecay.text().toInt(), \
        _settings->publishFeaturesToCore.isChecked());
}


void SpidrPlugin::stopComputation() {
    // Request interruption of the computation
    if (_spidrAnalysisQt.isRunning())
    {
        // release openGL context 
        _spidrAnalysisQt.stopComputation();
        _spidrAnalysisQt.exit();

        // Wait until the thread has terminated (max. 3 seconds)
        if (!_spidrAnalysisQt.wait(3000))
        {
            qDebug() << "Spatial Analysis computation thread did not close in time, terminating...";
            _spidrAnalysisQt.terminate();
            _spidrAnalysisQt.wait();
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
