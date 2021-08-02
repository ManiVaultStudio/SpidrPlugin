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
    AnalysisPlugin("Spidr")
{
}

SpidrPlugin::~SpidrPlugin(void)
{
    stopComputation();
    delete _spidrAnalysisWrapper;
    delete _tnseWrapper;
}

void SpidrPlugin::init()
{
    _settings = std::make_unique<SpidrSettingsWidget>(*this);
    _spidrAnalysisWrapper = NULL;
    _tnseWrapper = NULL;

    // Connet settings
    connect(_settings.get(), &SpidrSettingsWidget::dataSetPicked, this, &SpidrPlugin::dataSetPicked);

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

    // Start spatial analysis in worker thread
    delete _spidrAnalysisWrapper; delete _tnseWrapper;
    _spidrAnalysisWrapper = new SpidrAnalysisQtWrapper();
    _tnseWrapper = new TsneComputationQt();

    // set the data and all the parameters
    _spidrAnalysisWrapper->setup(attribute_data, pointIDsGlobal, numDims, imgSize, _embeddingName, backgroundIDsGlobal,
        _settings->distanceMetric.currentData().toPoint().y(),  // aknnMetric
        _settings->distanceMetric.currentData().toPoint().x(),  // featType
        _settings->kernelWeight.currentData().value<unsigned int>(),   // kernelType
        _settings->kernelSize.text().toInt(),       // numLocNeighbors
        _settings->histBinSize.text().toInt(), 
        _settings->knnOptions.currentData().value<unsigned int>(),   // aknnAlgType
        _settings->numIterations.text().toInt(),
        _settings->perplexity.text().toInt(), 
        _settings->exaggeration.text().toInt(), 
        _settings->expDecay.text().toInt(), \
        _settings->weightSpaAttrNum.value(),        // MVNweight
        _settings->publishFeaturesToCore.isChecked(),
        _settings->forceBackgroundFeatures.isChecked()
    );

    _spidrAnalysisWrapper->moveToThread(&workerThreadSpidr);
    _tnseWrapper->moveToThread(&workerThreadtSNE);

    //    connect(&workerThreadSpidr, &QThread::finished, _spidrAnalysisWrapper, &QObject::deleteLater);
    // TODO delete once embedding finished
    connect(this, &SpidrPlugin::startAnalysis, _spidrAnalysisWrapper, &SpidrAnalysisQtWrapper::spatialAnalysis);
    connect(_spidrAnalysisWrapper, &SpidrAnalysisQtWrapper::finishedKnn, this, &SpidrPlugin::tsneComputation);
    connect(_spidrAnalysisWrapper, &SpidrAnalysisQtWrapper::publishFeatures, this, &SpidrPlugin::onPublishFeatures);
    connect(_spidrAnalysisWrapper, &SpidrAnalysisQtWrapper::progressMessage, [this](const QString& message) {_settings->setSubtitle(message); });
    connect(this, &SpidrPlugin::starttSNE, _tnseWrapper, &TsneComputationQt::compute);

    // Connect embedding
    connect(_tnseWrapper, &TsneComputationQt::newEmbedding, this, &SpidrPlugin::onNewEmbedding);
    connect(_tnseWrapper, &TsneComputationQt::finishedEmbedding, this, &SpidrPlugin::onFinishedEmbedding);
    connect(_tnseWrapper, &TsneComputationQt::progressMessage, [this](const QString& message) {_settings->setSubtitle(message); });
    connect(_tnseWrapper, &TsneComputationQt::computationStopped, _settings.get(), &SpidrSettingsWidget::computationStopped);

    workerThreadSpidr.start();
    emit startAnalysis();

}

void SpidrPlugin::tsneComputation()
{
    // is called once knn computation is finished in _spidrAnalysisWrapper
    std::vector<int> _knnIds;
    std::vector<float> _knnDists;
    std::tie(_knnIds, _knnDists) = _spidrAnalysisWrapper->getKNN();
    _tnseWrapper->setup(_knnIds, _knnDists, _spidrAnalysisWrapper->getParameters()); // maybe I have to do this differently by sending a signal and getting hte values as a return...
    workerThreadtSNE.start();
    emit starttSNE();
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
    QString backgroundName = _settings->backgroundNameLine->currentText();

    if (!backgroundName.isEmpty()) {
        Points& backgroundPoints = _core->requestData<Points>(backgroundName);

        if (_settings->backgroundFromData.isChecked())
        {
            qDebug() << "SpidrPlugin: Read background from data set " << backgroundName << " (using the data set values)";
            auto totalNumPoints = backgroundPoints.getNumPoints();
            backgroundIDsGlobal.clear();
            backgroundIDsGlobal.resize(totalNumPoints);
            backgroundPoints.visitFromBeginToEnd([&backgroundIDsGlobal, totalNumPoints](auto beginOfData, auto endOfData)
            {
#ifdef NDEBUG
#pragma omp parallel for
#endif
                for (int i = 0; i < totalNumPoints; i++)
                {
                    backgroundIDsGlobal[i] = beginOfData[i];
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
    // TODO: check if the interactive selection actually works before a background is inserted
    const std::vector<float>& outputData = _tnseWrapper->output();
    Points& embedding = _core->requestData<Points>(_embeddingName);

    embedding.setData(outputData.data(), _spidrAnalysisWrapper->getNumEmbPoints(), 2);  // getNumEmbPoints might not agree with the numPoints w/o bg

    _core->notifyDataChanged(_embeddingName);
}

void SpidrPlugin::onFinishedEmbedding() {
    std::vector<float> outputData = _tnseWrapper->output();
    std::vector<float> embWithBg;
    _spidrAnalysisWrapper->addBackgroundToEmbedding(embWithBg, outputData);

    assert(embWithBg.size() % 2 == 0);
    assert(embWithBg.size() == _spidrAnalysisWrapper->getNumImagePoints() * 2);

    qDebug() << "SpidrPlugin: Publishing final embedding";

    Points& embedding = _core->requestData<Points>(_embeddingName);
    embedding.setData(embWithBg.data(), _spidrAnalysisWrapper->getNumImagePoints(), 2);
    _core->notifyDataChanged(_embeddingName);

    _settings.get()->computationStopped();

    qDebug() << "SpidrPlugin: Done.";
    _settings->setSubtitle("");
}

void SpidrPlugin::onPublishFeatures(const unsigned int dataFeatsSize) {
    qDebug() << "SpidrPlugin: Publish features to core";
    QString featureDataSetName = _core->createDerivedData(_settings->getEmbName() + "_Features", _settings->getCurrentDataItem());
    Points& featureDataSet = _core->requestData<Points>(featureDataSetName);
    featureDataSet.setData(_spidrAnalysisWrapper->getFeatures()->data(), dataFeatsSize, _spidrAnalysisWrapper->getNumFeatureValsPerPoint());

    // Set dimension names of feature data set
    std::vector<bool> enabledDimensions = _settings->getEnabledDimensions();
    std::vector<QString> dimensionNames = _core->requestData<Points>(_settings->getCurrentDataItem()).getDimensionNames();
    std::vector<QString> enabledDimensionNames;

    assert(enabledDimensions.size() == dimensionNames.size());

    for (int i = 0; i < enabledDimensions.size(); i++) {
        if (enabledDimensions[i])
            enabledDimensionNames.push_back(dimensionNames[i]);
    }
    featureDataSet.setDimensionNames(enabledDimensionNames);
}


void SpidrPlugin::stopComputation() {
    // Request interruption of the computation
    if (workerThreadtSNE.isRunning())
    {
        // release openGL context 
        _tnseWrapper->stopGradientDescent();
        workerThreadtSNE.exit();

        // Wait until the thread has terminated (max. 3 seconds)
        if (!workerThreadtSNE.wait(3000))
        {
            qDebug() << "Spatial Analysis computation thread did not close in time, terminating...";
            workerThreadtSNE.terminate();
            workerThreadtSNE.wait();
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
