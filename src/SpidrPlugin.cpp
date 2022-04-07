#include "SpidrPlugin.h"

#include "ImageData/Images.h"
#include "PointData.h"

#include <QtCore>
#include <QSize>
#include <QtDebug>
#include <QPainter>

#include <utility>      // std::as_const
#include <vector>       // std::vector

Q_PLUGIN_METADATA(IID "nl.tudelft.SpidrPlugin")

#include <set>

using namespace hdps;
using namespace hdps::gui;

// =============================================================================
// Analysis Plugin
// =============================================================================

SpidrPlugin::SpidrPlugin(const PluginFactory* factory) :
    AnalysisPlugin(factory),
    _spidrSettingsAction(this),
    _spidrAnalysisWrapper(),
    _tnseWrapper()
{
    setObjectName("Spidr");
}

SpidrPlugin::~SpidrPlugin(void)
{
    stopComputation();
}

void SpidrPlugin::init()
{
    // Get both image data and its parent data set
    auto imagesDataset = getInputDataset<Images>();
    auto inputDataset = static_cast<hdps::Dataset<Points>>(imagesDataset->getParent());
    
    // set the output data as derived from the parent data set (since the type differs from the input data set)
    setOutputDataset(_core->createDerivedDataset("sp-tsne_embedding", inputDataset, inputDataset));
    auto& outputDataset = getOutputDataset<Points>();

    // Automaticallt select the output data in the GUI data hierarchy
    getOutputDataset()->getDataHierarchyItem().select();

    // Set up output data
    std::vector<float> initialData;
    const auto numEmbeddingDimensions = 2;
    initialData.resize(inputDataset->getNumPoints() * numEmbeddingDimensions);
    outputDataset->setData(initialData.data(), inputDataset->getNumPoints(), numEmbeddingDimensions);
    outputDataset->getDataHierarchyItem().select();

    // Set up action connections
    outputDataset->addAction(_spidrSettingsAction.getGeneralSpidrSettingsAction());
    outputDataset->addAction(_spidrSettingsAction.getAdvancedTsneSettingsAction());
    outputDataset->addAction(_spidrSettingsAction.getDimensionSelectionAction());
    outputDataset->addAction(_spidrSettingsAction.getBackgroundSelectionAction());

    outputDataset->getDataHierarchyItem().select();

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
        const std::vector<float>& outputData = _tnseWrapper.output();

        // Update the output points dataset with new data from the TSNE analysis
        getOutputDataset<Points>()->setData(outputData.data(), _spidrAnalysisWrapper.getNumForegroundPoints(), 2);

        _spidrSettingsAction.getGeneralSpidrSettingsAction().getNumberOfComputatedIterationsAction().setValue(_tnseWrapper.getNumCurrentIterations() - 1);

        QCoreApplication::processEvents();

        // Notify others that the embedding data changed
        _core->notifyDatasetChanged(getOutputDataset());
        });


    _spidrSettingsAction.getDimensionSelectionAction().getPickerAction().setPointsDataset(inputDataset);

    connect(&computationAction.getRunningAction(), &ToggleAction::toggled, this, [this, &computationAction, updateComputationAction](bool toggled) {
        _spidrSettingsAction.getDimensionSelectionAction().setEnabled(!toggled);
        _spidrSettingsAction.getBackgroundSelectionAction().setEnabled(!toggled);

        updateComputationAction();
        });


    updateComputationAction();

    registerDataEventByType(PointType, std::bind(&SpidrPlugin::onDataEvent, this, std::placeholders::_1));

    setTaskName("Spidr");

    //_spidrSettingsAction.loadDefault();
}

void SpidrPlugin::onDataEvent(hdps::DataEvent* dataEvent)
{

    if (dataEvent->getDataset() == getInputDataset())
        _spidrSettingsAction.getDimensionSelectionAction().getPickerAction().setPointsDataset(dataEvent->getDataset<Points>());

}


void SpidrPlugin::startComputation()
{
    setTaskRunning();
    setTaskProgress(0.0f);
    setTaskDescription("Preparing data");

    _spidrSettingsAction.getGeneralSpidrSettingsAction().getNumberOfComputatedIterationsAction().reset();

    // Get the data
    // Use the source data to get the points 
    // Use the image data to get the image size
    qDebug() << "SpidrPlugin: Read data ";
    const auto inputImages = getInputDataset<Images>();
    const auto inputPoints = static_cast<hdps::Dataset<Points>>(inputImages->getParent());

    std::vector<float> attribute_data;              // Actual channel valures, only consider enabled dimensions
    std::vector<unsigned int> pointIDsGlobal;       // Global ID of each point in the image
    std::vector<unsigned int> backgroundIDsGlobal;  // ID of points which are not used during the t-SNE embedding - but will inform the feature extraction and distance calculation

    // Extract the enabled dimensions from the data
    std::vector<bool> enabledDimensions = _spidrSettingsAction.getDimensionSelectionAction().getPickerAction().getEnabledDimensions();
    std::vector<unsigned int> enabledDimensionsIndices;
    const auto numEnabledDimensions = count_if(enabledDimensions.begin(), enabledDimensions.end(), [](bool b) { return b; });

    // Populate selected data attributes
    attribute_data.resize((inputPoints->isFull() ? inputPoints->getNumPoints() : inputPoints->indices.size()) * numEnabledDimensions);

    for (unsigned int i = 0; i < inputPoints->getNumDimensions(); i++)
        if (enabledDimensions[i])
            enabledDimensionsIndices.push_back(i);

    inputPoints->populateDataForDimensions<std::vector<float>, std::vector<unsigned int>>(attribute_data, enabledDimensionsIndices);
    inputPoints->getGlobalIndices(pointIDsGlobal);

    // Image size
    ImgSize imgSize;
    imgSize.width = inputImages->getImageSize().width();
    imgSize.height = inputImages->getImageSize().height();

    // Background IDs
    hdps::Dataset<Points> backgroundDataset = static_cast<hdps::Dataset<Points>>(_spidrSettingsAction.getBackgroundSelectionAction().getBackgroundDataset());
    if (backgroundDataset.isValid())
    {
        if (!_spidrSettingsAction.getBackgroundSelectionAction().getIDsInData())  // use the global IDs of the background dataset (it is a subset of the inputPoints dataset)
        {
            backgroundDataset->getGlobalIndices(backgroundIDsGlobal);

            qDebug() << "SpidrPlugin: Use background IDs from dataset " << backgroundDataset->getGuiName() << " (using its global point IDs)" << backgroundIDsGlobal.size() << " background points";
        }
        else   // background dataset contains the background IDs
        {
            // Check of the dimensions and number of points make sense
            if ( backgroundDataset->getNumDimensions() == 1 && backgroundDataset->getNumPoints() < inputPoints->getNumPoints() )
            {
                backgroundDataset->visitFromBeginToEnd([&backgroundIDsGlobal](auto beginBackgroundDataset, auto endBackgroundDataset) {
                    backgroundIDsGlobal.insert(backgroundIDsGlobal.begin(), beginBackgroundDataset, endBackgroundDataset);
                });

                qDebug() << "SpidrPlugin: Use background IDs from dataset " << backgroundDataset->getGuiName() << " (using its data values): " << backgroundIDsGlobal.size() << " background points";
            }
        }
    }

    // complete Spidr parameters
    _spidrSettingsAction.getSpidrParameters()._numPoints = pointIDsGlobal.size();
    _spidrSettingsAction.getSpidrParameters()._numForegroundPoints = pointIDsGlobal.size() - backgroundIDsGlobal.size();
    _spidrSettingsAction.getSpidrParameters()._numDims = numEnabledDimensions;
    _spidrSettingsAction.getSpidrParameters()._imgSize = imgSize;

    // Setup worker classes with data and parameters
    qDebug() << "SpidrPlugin: Initialize settings";
    
    // set the data and all the parameters
    _spidrAnalysisWrapper.setup(attribute_data, pointIDsGlobal, QString("sp-emb_hdps"), backgroundIDsGlobal, _spidrSettingsAction.getSpidrParameters());

    // Start spatial analysis in worker thread
    _workerThreadSpidr = new QThread();
    _workerThreadtSNE = new QThread();

    _spidrAnalysisWrapper.moveToThread(_workerThreadSpidr);
    _tnseWrapper.moveToThread(_workerThreadtSNE);

    // delete threads after work is done
    connect(_workerThreadSpidr, &QThread::finished, _workerThreadSpidr, &QObject::deleteLater);
    connect(_workerThreadtSNE, &QThread::finished, _workerThreadtSNE, &QObject::deleteLater);

    // connect wrappers
    connect(&_spidrAnalysisWrapper, &SpidrAnalysisQtWrapper::finishedKnn, this, &SpidrPlugin::tsneComputation);
    connect(&_spidrAnalysisWrapper, &SpidrAnalysisQtWrapper::publishFeatures, this, &SpidrPlugin::onPublishFeatures);
    connect(this, &SpidrPlugin::startAnalysis, &_spidrAnalysisWrapper, &SpidrAnalysisQtWrapper::spatialAnalysis);
    connect(this, &SpidrPlugin::starttSNE, &_tnseWrapper, &TsneComputationQtWrapper::compute);


    qDebug() << "SpidrPlugin: Start Analysis";
    _spidrSettingsAction.getComputationAction().getRunningAction().setChecked(true);
    _workerThreadSpidr->start();
    emit startAnalysis();   // trigger computation in other thread
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

void SpidrPlugin::onFinishedEmbedding() {
    auto& embedding = getOutputDataset<Points>();

    std::vector<float> outputData = _tnseWrapper.output();
    std::vector<float> embWithBg;
    _spidrAnalysisWrapper.addBackgroundToEmbedding(embWithBg, outputData);

    assert(embWithBg.size() % 2 == 0);
    assert(embWithBg.size() == _spidrAnalysisWrapper.getNumImagePoints() * 2);

    qDebug() << "SpidrPlugin: Publishing final embedding";

    embedding->setData(embWithBg.data(), _spidrAnalysisWrapper.getNumImagePoints(), 2);
    _core->notifyDatasetChanged(getOutputDataset());

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

}

// =============================================================================
// Factory
// =============================================================================

AnalysisPlugin* SpidrPluginFactory::produce()
{
    return new SpidrPlugin(this);
}

hdps::DataTypes SpidrPluginFactory::supportedDataTypes() const
{
    DataTypes supportedTypes;
    supportedTypes.append(ImageType);
    return supportedTypes;
}


QIcon SpidrPluginFactory::getIcon() const
{
    const auto margin = 3;
    const auto pixmapSize = QSize(100, 100);
    const auto pixmapRect = QRect(QPoint(), pixmapSize).marginsRemoved(QMargins(margin, margin, margin, margin));
    const auto halfSize = pixmapRect.size() / 2;

    // Create pixmap
    QPixmap pixmap(pixmapSize);

    // Fill with a transparent background
    pixmap.fill(Qt::transparent);

    // Create a painter to draw in the pixmap
    QPainter painter(&pixmap);

    // Enable anti-aliasing
    painter.setRenderHint(QPainter::Antialiasing);

    // Get the text color from the application
    const auto textColor = QApplication::palette().text().color();

    // Configure painter
    painter.setPen(QPen(textColor, 1, Qt::SolidLine, Qt::SquareCap, Qt::SvgMiterJoin));
    painter.setFont(QFont("Arial", 38, 250));

    const auto textOption = QTextOption(Qt::AlignCenter);

    // Do the painting
    painter.drawText(QRect(pixmapRect.topLeft(), halfSize), "S", textOption);
    painter.drawText(QRect(QPoint(halfSize.width(), pixmapRect.top()), halfSize), "S", textOption);
    painter.drawText(QRect(QPoint(pixmapRect.left(), halfSize.height()), halfSize), "N", textOption);
    painter.drawText(QRect(QPoint(halfSize.width(), halfSize.height()), halfSize), "E", textOption);

    return QIcon(pixmap);
}