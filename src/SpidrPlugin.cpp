#include "SpidrPlugin.h"
#include "SpidrSettingsWidget.h"

#include <QtCore>
#include <QSize>
#include <QtDebug>

#include <utility>      // std::as_const
#include <vector>       // std::vector

Q_PLUGIN_METADATA(IID "nl.tudelft.SpidrPlugin")

#include <set>

// =============================================================================
// View
// =============================================================================

using namespace hdps;
SpidrPlugin::SpidrPlugin(const PluginFactory* factory) :
    AnalysisPlugin(factory),
    _spidrSettingsAction(this),
    _dimensionSelectionAction(this), 
    _spidrAnalysisWrapper(),
    _tnseWrapper()
{
}

SpidrPlugin::~SpidrPlugin(void)
{
    stopComputation();
}

void SpidrPlugin::init()
{
    setOutputDatasetName(_core->createDerivedData("sp-tsne_embedding", getInputDatasetName()));

    // Get input/output datasets
    auto& inputDataset = getInputDataset<Points>();
    auto& outputDataset = getOutputDataset<Points>();

    // Set up output data
    std::vector<float> initialData;
    const auto numEmbeddingDimensions = 2;
    initialData.resize(inputDataset.getNumPoints() * numEmbeddingDimensions);
    outputDataset.setData(initialData.data(), inputDataset.getNumPoints(), numEmbeddingDimensions);
    _core->getDataHierarchyItem(outputDataset.getName())->select();

    // Set up action connections
    outputDataset.addAction(_spidrSettingsAction.getGeneralSpidrSettingsAction());

    auto& computationAction = _spidrSettingsAction.getComputationAction();

    // 
    const auto updateComputationAction = [this, &computationAction]() {
        const auto isRunning = computationAction.getRunningAction().isChecked();

        computationAction.getStartComputationAction().setEnabled(!isRunning);
        computationAction.getStopComputationAction().setEnabled(isRunning);
    };


    // Update task description in GUI
    connect(&_spidrAnalysisWrapper, &SpidrAnalysisQtWrapper::progressSection, this, [this](const QString& section) {
        if (getTaskStatus() == DataHierarchyItem::TaskStatus::Aborted)
            return;

        setTaskDescription(section);
        });

    connect(&_tnseWrapper, &TsneComputationQtWrapper::progressPercentage, this, [this](const float& percentage) {
        if (getTaskStatus() == DataHierarchyItem::TaskStatus::Aborted)
            return;

        setTaskProgress(percentage);
        });

    connect(&_tnseWrapper, &TsneComputationQtWrapper::progressSection, this, [this](const QString& section) {
        if (getTaskStatus() == DataHierarchyItem::TaskStatus::Aborted)
            return;

        setTaskDescription(section);
        });

    connect(&_spidrAnalysisWrapper, &SpidrAnalysisQtWrapper::progressSection, this, [this](const QString& section) {
        if (getTaskStatus() == DataHierarchyItem::TaskStatus::Aborted)
            return;

        setTaskDescription(section);
        });

    // Embedding finished
    connect(&_tnseWrapper, &TsneComputationQtWrapper::finishedEmbedding, this, [this, &computationAction]() {
        onFinishedEmbedding();

        setTaskFinished();

        computationAction.getRunningAction().setChecked(false);

        _spidrSettingsAction.getGeneralSpidrSettingsAction().setReadOnly(false);
        _spidrSettingsAction.getAdvancedTsneSettingsAction().setReadOnly(false);
        });

    // start computation
    connect(&computationAction.getStartComputationAction(), &TriggerAction::triggered, this, [this, &computationAction]() {
        _spidrSettingsAction.getGeneralSpidrSettingsAction().setReadOnly(true);
        _spidrSettingsAction.getAdvancedTsneSettingsAction().setReadOnly(true);

        startComputation();
        });

    // abort t-SNE
    connect(&computationAction.getStopComputationAction(), &TriggerAction::triggered, this, [this]() {
        setTaskDescription("Aborting TSNE");

        qApp->processEvents();

        stopComputation();
        });

    // embedding changed
    connect(&_tnseWrapper, &TsneComputationQtWrapper::newEmbedding, this, [this]() {
        Points& embedding = getOutputDataset<Points>();

        // TODO: check if the interactive selection actually works before a background is inserted
        const std::vector<float>& outputData = _tnseWrapper.output();

        embedding.setData(outputData.data(), _spidrAnalysisWrapper.getNumForegroundPoints(), 2);
        _core->notifyDataChanged(getOutputDatasetName());
        });


    _dimensionSelectionAction.dataChanged(inputDataset);

    connect(&computationAction.getRunningAction(), &ToggleAction::toggled, this, [this, &computationAction, updateComputationAction](bool toggled) {
        _dimensionSelectionAction.setEnabled(!toggled);

        updateComputationAction();
        });


    updateComputationAction();

    registerDataEventByType(PointType, std::bind(&SpidrPlugin::onDataEvent, this, std::placeholders::_1));

    setTaskName("Spidr");

    //_spidrAnalysisWrapper = NULL;
    //_tnseWrapper = NULL;


}

void SpidrPlugin::onDataEvent(hdps::DataEvent* dataEvent)
{

    if (dataEvent->getType() == EventType::DataChanged) {
        auto dataChangedEvent = static_cast<DataChangedEvent*>(dataEvent);

        // If we are not looking at the changed dataset, ignore it
        if (dataChangedEvent->dataSetName != getInputDatasetName())
            return;

        // Passes changes to the current dataset to the dimension selection widget
        Points& points = _core->requestData<Points>(dataChangedEvent->dataSetName);

        _dimensionSelectionAction.dataChanged(_core->requestData<Points>(dataChangedEvent->dataSetName));
    }
}


void SpidrPlugin::startComputation()
{
    setTaskRunning();
    setTaskProgress(0.0f);
    setTaskDescription("Preparing data");

    // Get the data
    qDebug() << "SpidrPlugin: Read data ";
    const auto& inputPoints = getInputDataset<Points>();

    std::vector<unsigned int> pointIDsGlobal; // Global ID of each point in the image
    std::vector<float> attribute_data;        // Actual channel valures, only consider enabled dimensions
    std::vector<unsigned int> backgroundIDsGlobal;  // ID of points which are not used during the t-SNE embedding - but will inform the feature extraction and distance calculation
    ImgSize imgSize;

    // Extract the enabled dimensions from the data
    std::vector<bool> enabledDimensions = _dimensionSelectionAction.getEnabledDimensions();
    const auto numEnabledDimensions = count_if(enabledDimensions.begin(), enabledDimensions.end(), [](bool b) { return b; });

    // Populate selected data attributes
    attribute_data.resize((inputPoints.isFull() ? inputPoints.getNumPoints() : inputPoints.indices.size()) * numEnabledDimensions);

    for (int i = 0; i < inputPoints.getNumDimensions(); i++)
        if (enabledDimensions[i])
            pointIDsGlobal.push_back(i);

    inputPoints.populateDataForDimensions<std::vector<float>, std::vector<unsigned int>>(attribute_data, pointIDsGlobal);

    // Image size
    QSize qtImgSize = inputPoints.getProperty("ImageSize", QSize()).toSize();
    imgSize.width = qtImgSize.width();
    imgSize.height = qtImgSize.height();

    // Setup worker classes with data and parameters
    qDebug() << "SpidrPlugin: Initialize settings";

    // deleted in stopComputation(), which is also called by the destructor
    // TODO: remove pointer references
    //_spidrAnalysisWrapper = new SpidrAnalysisQtWrapper();
    //_tnseWrapper = new TsneComputationQtWrapper();
    
    // set the data and all the parameters
    _spidrAnalysisWrapper.setup(attribute_data, pointIDsGlobal, numEnabledDimensions, imgSize, QString("sp-emb_hdps"), backgroundIDsGlobal, _spidrSettingsAction.getSpidrParameters());

    // Start spatial analysis in worker thread
    _workerThreadSpidr = new QThread();
    _workerThreadtSNE = new QThread();

    _spidrAnalysisWrapper.moveToThread(_workerThreadSpidr);
    _tnseWrapper.moveToThread(_workerThreadtSNE);

    // delete threads after work is done
    connect(_workerThreadSpidr, &QThread::finished, _workerThreadSpidr, &QObject::deleteLater);
    connect(_workerThreadtSNE, &QThread::finished, _workerThreadtSNE, &QObject::deleteLater);

    // connect wrappers
    //connect(this, &SpidrPlugin::startAnalysis, &_spidrAnalysisWrapper, &SpidrAnalysisQtWrapper::spatialAnalysis);
    connect(&_spidrAnalysisWrapper, &SpidrAnalysisQtWrapper::finishedKnn, this, &SpidrPlugin::tsneComputation);
    connect(&_spidrAnalysisWrapper, &SpidrAnalysisQtWrapper::publishFeatures, this, &SpidrPlugin::onPublishFeatures);
    connect(this, &SpidrPlugin::starttSNE, &_tnseWrapper, &TsneComputationQtWrapper::compute);


    qDebug() << "SpidrPlugin: Start Analysis";
    _spidrSettingsAction.getComputationAction().getRunningAction().setChecked(true);
    _workerThreadSpidr->start();
    _spidrAnalysisWrapper.spatialAnalysis();
}

void SpidrPlugin::tsneComputation()
{
    // is called once knn computation is finished in _spidrAnalysisWrapper
    std::vector<int> _knnIds;
    std::vector<float> _knnDists;
    std::tie(_knnIds, _knnDists) = _spidrAnalysisWrapper.getKnn();
    _tnseWrapper.setup(_knnIds, _knnDists, _spidrAnalysisWrapper.getParameters()); // maybe I have to do this differently by sending a signal and getting hte values as a return...
    _workerThreadtSNE->start();
    emit starttSNE();
}

//void SpidrPlugin::retrieveData(QString dataName, std::vector<unsigned int>& pointIDsGlobal, std::vector<float>& attribute_data, unsigned int& numEnabledDimensions, ImgSize& imgSize, std::vector<unsigned int>& backgroundIDsGlobal) {
//    Points& points = _core->requestData<Points>(dataName);
//    QSize qtImgSize = points.getProperty("ImageSize", QSize()).toSize();
//    imgSize.width = qtImgSize.width();
//    imgSize.height = qtImgSize.height();
//
//    std::vector<bool> enabledDimensions = _settings->getEnabledDimensions();
//
//    // Get number of enabled dimensions
//    unsigned int numDimensions = points.getNumDimensions();
//    numEnabledDimensions = count_if(enabledDimensions.begin(), enabledDimensions.end(), [](bool b) { return b; });
//
//    // Get indices of selected points
//    pointIDsGlobal = points.indices;
//    // If points represent all data set, select them all
//    if (points.isFull()) {
//        std::vector<unsigned int> all(points.getNumPoints());
//        std::iota(std::begin(all), std::end(all), 0);
//
//        pointIDsGlobal = all;
//    }
//
//    // For all selected points, retrieve values from each dimension
//    attribute_data.reserve(pointIDsGlobal.size() * numEnabledDimensions);
//
//    points.visitFromBeginToEnd([&attribute_data, &pointIDsGlobal, &enabledDimensions, &numDimensions](auto beginOfData, auto endOfData)
//    {
//        for (const auto& pointId : pointIDsGlobal)
//        {
//            for (unsigned int dimensionId = 0; dimensionId < numDimensions; dimensionId++)
//            {
//                if (enabledDimensions[dimensionId]) {
//                    const auto index = pointId * numDimensions + dimensionId;
//                    attribute_data.push_back(beginOfData[index]);
//                }
//            }
//        }
//    });
//
//    // If a background data set is given, store the background indices
//    QString backgroundName = _settings->backgroundNameLine->currentText();
//
//    if (!backgroundName.isEmpty()) {
//        Points& backgroundPoints = _core->requestData<Points>(backgroundName);
//
//        if (_settings->backgroundFromData.isChecked())
//        {
//            qDebug() << "SpidrPlugin: Read background from data set " << backgroundName << " (using the data set values)";
//            auto totalNumPoints = backgroundPoints.getNumPoints();
//            backgroundIDsGlobal.clear();
//            backgroundIDsGlobal.resize(totalNumPoints);
//            backgroundPoints.visitFromBeginToEnd([&backgroundIDsGlobal, totalNumPoints](auto beginOfData, auto endOfData)
//            {
//#ifdef NDEBUG
//#pragma omp parallel for
//#endif
//                for (int i = 0; i < totalNumPoints; i++)
//                {
//                    backgroundIDsGlobal[i] = beginOfData[i];
//                }
//            });
//        }
//        else
//        {
//            qDebug() << "SpidrPlugin: Read background from data set " << backgroundName << " (using the data set indices)";
//            backgroundIDsGlobal = backgroundPoints.indices;
//        }
//    }
//
//}


//void SpidrPlugin::onNewEmbedding() {
//    // TODO: check if the interactive selection actually works before a background is inserted
//    const std::vector<float>& outputData = _tnseWrapper.output();
//    Points& embedding = _core->requestData<Points>(_embeddingName);
//
//    embedding.setData(outputData.data(), _spidrAnalysisWrapper.getNumForegroundPoints(), 2);
//
//    _core->notifyDataChanged(_embeddingName);
//}

void SpidrPlugin::onFinishedEmbedding() {
    Points& embedding = getOutputDataset<Points>();

    std::vector<float> outputData = _tnseWrapper.output();
    std::vector<float> embWithBg;
    _spidrAnalysisWrapper.addBackgroundToEmbedding(embWithBg, outputData);

    assert(embWithBg.size() % 2 == 0);
    assert(embWithBg.size() == _spidrAnalysisWrapper.getNumImagePoints() * 2);

    qDebug() << "SpidrPlugin: Publishing final embedding";

    embedding.setData(embWithBg.data(), _spidrAnalysisWrapper.getNumImagePoints(), 2);
    _core->notifyDataChanged(getOutputDatasetName());

    qDebug() << "SpidrPlugin: Done.";
}

void SpidrPlugin::onPublishFeatures(const unsigned int dataFeatsSize) {
    qDebug() << "SpidrPlugin: Publish features to core (WARNING: Not currently enabled - no features are published to the core)";
    //QString featureDataSetName = _core->createDerivedData(_settings->getEmbName() + "_Features", _settings->getCurrentDataItem());
    //Points& featureDataSet = _core->requestData<Points>(featureDataSetName);
    //featureDataSet.setData(_spidrAnalysisWrapper->getFeatures()->data(), dataFeatsSize, _spidrAnalysisWrapper->getNumFeatureValsPerPoint());

    //// Set dimension names of feature data set
    //std::vector<bool> enabledDimensions = _settings->getEnabledDimensions();
    //std::vector<QString> dimensionNames = _core->requestData<Points>(_settings->getCurrentDataItem()).getDimensionNames();
    //std::vector<QString> enabledDimensionNames;

    //assert(enabledDimensions.size() == dimensionNames.size());

    //for (int i = 0; i < enabledDimensions.size(); i++) {
    //    if (enabledDimensions[i])
    //        enabledDimensionNames.push_back(dimensionNames[i]);
    //}
    //featureDataSet.setDimensionNames(enabledDimensionNames);
}


void SpidrPlugin::stopComputation() {
    // Request interruption of the computation

    if (_workerThreadSpidr->isRunning())
    {
        _workerThreadSpidr->exit();

        // Wait until the thread has terminated (max. 3 seconds)
        if (!_workerThreadSpidr->wait(3000))
        {
            qDebug() << "SpidrPlugin: Spidr computation thread did not close in time, terminating...";
            _workerThreadSpidr->terminate();
            _workerThreadSpidr->wait();
        }

    }

    if (_workerThreadtSNE->isRunning())
    {
        // release openGL context 
        _tnseWrapper.stopGradientDescent();
        _workerThreadtSNE->exit();

        // Wait until the thread has terminated (max. 3 seconds)
        if (!_workerThreadtSNE->wait(3000))
        {
            qDebug() << "SpidrPlugin: t-SNE computation thread did not close in time, terminating...";
            _workerThreadtSNE->terminate();
            _workerThreadtSNE->wait();
        }

    }

    qDebug() << "SpidrPlugin: Spatial t-SNE Analysis computation stopped.";

    //delete _spidrAnalysisWrapper; _spidrAnalysisWrapper = NULL;
    //delete _tnseWrapper; _tnseWrapper = NULL;

}

// =============================================================================
// Factory
// =============================================================================

AnalysisPlugin* SpidrPluginFactory::produce()
{
    return new SpidrPlugin(this);
}
