#include "SpidrPlugin.h"

#include "ImageData/Images.h"
#include "PointData/PointData.h"
#include "PointData/InfoAction.h"

#include "actions/PluginTriggerAction.h"
#include "util/Icon.h"
#include "util/StyledIcon.h"

#include "SpidrSettingsAction.h"
#include "SpidrAnalysisQtWrapper.h"
#include "TsneComputationQtWrapper.h"

#include <QtCore>
#include <QSize>
#include <QtDebug>
#include <QPainter>

#include <utility>      // std::as_const
#include <vector>       // std::vector
#include <algorithm>    // std::set_difference
#include <iterator>     // std::inserter

Q_PLUGIN_METADATA(IID "nl.tudelft.SpidrPlugin")

using namespace mv;
using namespace mv::gui;

// =============================================================================
// Analysis Plugin
// =============================================================================

SpidrPlugin::SpidrPlugin(const PluginFactory* factory) :
    AnalysisPlugin(factory),
    _spidrSettingsAction(std::make_unique<SpidrSettingsAction>(this)),
    _spidrAnalysisWrapper(std::make_unique<SpidrAnalysisQtWrapper>()),
    _tnseWrapper(std::make_unique<TsneComputationQtWrapper>()),
    _computationPreparationTask(this, "Preparing Spidr computation"),
    _workerThreadSpidr(nullptr), _workerThreadtSNE(nullptr)
{
    setObjectName("Spidr");

    _computationPreparationTask.setDescription("All operations prior to Spidr computation");
}

SpidrPlugin::~SpidrPlugin(void)
{
    stopComputation();
}

void SpidrPlugin::init()
{
    // Get both image data and its parent data set
    auto imagesDataset = getInputDataset<Images>();
    auto inputDataset = static_cast<mv::Dataset<Points>>(imagesDataset->getParent());
    
    // set the output data as derived from the parent data set (since the type differs from the input data set)
    setOutputDataset(mv::data().createDerivedDataset("sp-tsne_embedding", inputDataset, inputDataset));
    auto outputDataset = getOutputDataset<Points>();

    // Automaticallt select the output data in the GUI data hierarchy
    getOutputDataset()->getDataHierarchyItem().select();

    // Setup ManiVault progress reporting
    _computationPreparationTask.setParentTask(&outputDataset->getTask());

    // Set up output data
    std::vector<float> initialData;
    const size_t numEmbeddingDimensions = 2;
    initialData.resize(numEmbeddingDimensions * inputDataset->getNumPoints());
    outputDataset->setData(initialData.data(), inputDataset->getNumPoints(), numEmbeddingDimensions);
    outputDataset->getDataHierarchyItem().select();

    // Set up action connections
    outputDataset->addAction(_spidrSettingsAction->getGeneralSpidrSettingsAction());
    outputDataset->addAction(_spidrSettingsAction->getAdvancedTsneSettingsAction());
    outputDataset->addAction(_spidrSettingsAction->getDimensionSelectionAction());
    outputDataset->addAction(_spidrSettingsAction->getBackgroundSelectionAction());

    outputDataset->getDataHierarchyItem().select();

    // Do not show data info by default to give more space to other settings
    outputDataset->_infoAction->collapse();

    auto& computationAction = _spidrSettingsAction->getGeneralSpidrSettingsAction().getComputationAction();

    // 
    const auto updateComputationAction = [this, &computationAction]() {
        const auto isRunning = computationAction.getRunningAction().isChecked();

        computationAction.getStartComputationAction().setEnabled(!isRunning);
        computationAction.getStopComputationAction().setEnabled(isRunning);
    };

    // Embedding finished
    connect(_tnseWrapper.get(), &TsneComputationQtWrapper::finishedEmbedding, this, [this, &computationAction]() {
        onFinishedEmbedding();

        computationAction.getRunningAction().setChecked(false);

        _spidrSettingsAction->getGeneralSpidrSettingsAction().setReadOnly(false);
        _spidrSettingsAction->getAdvancedTsneSettingsAction().setReadOnly(false);

        _computationPreparationTask.setFinished();
        getOutputDataset()->getTask().setFinished();
    });

    // start computation
    connect(&computationAction.getStartComputationAction(), &TriggerAction::triggered, this, [this, &computationAction]() {
        _spidrSettingsAction->getGeneralSpidrSettingsAction().setReadOnly(true);
        _spidrSettingsAction->getAdvancedTsneSettingsAction().setReadOnly(true);

        startComputation();
        });

    // abort t-SNE
    connect(&computationAction.getStopComputationAction(), &TriggerAction::triggered, this, [this]() {
        qApp->processEvents();

        stopComputation();
        });

    // embedding changed
    connect(_tnseWrapper.get(), &TsneComputationQtWrapper::newEmbedding, this, [this]() {
        const std::vector<float>& outputData = _tnseWrapper->output();

        // Update the output points dataset with new data from the TSNE analysis
        getOutputDataset<Points>()->setData(outputData.data(), outputData.size() / 2, 2);

        _spidrSettingsAction->getGeneralSpidrSettingsAction().getNumberOfComputatedIterationsAction().setValue(_tnseWrapper->getNumCurrentIterations() - 1);

        QCoreApplication::processEvents();

        // Notify others that the embedding data changed
        events().notifyDatasetDataChanged(getOutputDataset());
        });


    _spidrSettingsAction->getDimensionSelectionAction().getPickerAction().setPointsDataset(inputDataset);

    connect(&computationAction.getRunningAction(), &ToggleAction::toggled, this, [this, &computationAction, updateComputationAction](bool toggled) {
        _spidrSettingsAction->getDimensionSelectionAction().setEnabled(!toggled);
        _spidrSettingsAction->getBackgroundSelectionAction().setEnabled(!toggled);

        updateComputationAction();
        });


    updateComputationAction();

    // Update dimension selection with new data
    connect(&inputDataset, &Dataset<Points>::dataChanged, this, [this, inputDataset]() {
        _spidrSettingsAction->getDimensionSelectionAction().getPickerAction().setPointsDataset(inputDataset);
        });


    // Update task description in GUI
    auto& datasetTask = getOutputDataset()->getTask();
    datasetTask.setName("Spidr Computation");
    datasetTask.setConfigurationFlag(Task::ConfigurationFlag::OverrideAggregateStatus);

    _tnseWrapper->setTask(&datasetTask);
    _spidrAnalysisWrapper->setTask(&datasetTask);
}

void SpidrPlugin::startComputation()
{
    getOutputDataset()->getTask().setRunning();

    _computationPreparationTask.setEnabled(true);
    _computationPreparationTask.setRunning();
    _computationPreparationTask.setProgress(0.0f);
    _computationPreparationTask.setDescription("Preparing data");

    // _spidrSettingsAction->getGeneralSpidrSettingsAction().getNumberOfComputatedIterationsAction().reset(); // deprecated

    // Get the data
    // Use the source data to get the points 
    // Use the image data to get the image size
    // if inputPoints is not a subset, inputPointsParent = inputPoints and contextIDsGlobal will remain empty
    qDebug() << "SpidrPlugin: Read data ";
    auto inputImages = getInputDataset<Images>();
    auto inputPoints = static_cast<mv::Dataset<Points>>(inputImages->getParent());
    auto inputPointsParent = static_cast<mv::Dataset<Points>>(inputImages->getParent());

    // we introduce all there global ID sets so that the user can work on subsets of data and still define background selections in those subsets

    std::vector<float> attribute_data;              // Actual channel valures, only consider enabled dimensions
    std::vector<unsigned int> pointIDsGlobal;       // Global ID of each point in the image
    std::vector<unsigned int> pointIDsFocus;        // Global ID of each point in the image selection
    std::vector<unsigned int> backgroundIDsGlobal;  // ID of points which are not used during the t-SNE embedding - but will inform the feature extraction and distance calculation
    std::vector<unsigned int> contextIDsGlobal;     // In case the dataset is a subset, these are the IDs of the original data that are NOT in the subset

    auto numPointsFocus = (inputPoints->isFull() ? inputPoints->getNumPoints() : inputPoints->indices.size());
    inputPoints->getGlobalIndices(pointIDsFocus);

    // traverse up the data hierarchy
    while (!inputPointsParent->isFull())
    {
        inputPointsParent = inputPointsParent->getParent();
    }

    // Extract the enabled dimensions from the data
    std::vector<bool> enabledDimensions = _spidrSettingsAction->getDimensionSelectionAction().getPickerAction().getEnabledDimensions();
    std::vector<unsigned int> enabledDimensionsIndices;
    const auto numEnabledDimensions = count_if(enabledDimensions.begin(), enabledDimensions.end(), [](bool b) { return b; });

    // Populate selected data attributes
    auto numPointsGlobal = (inputPointsParent->isFull() ? inputPointsParent->getNumPoints() : inputPointsParent->indices.size());
    attribute_data.resize(numPointsGlobal * numEnabledDimensions);

    for (unsigned int i = 0; i < inputPointsParent->getNumDimensions(); i++)
        if (enabledDimensions[i])
            enabledDimensionsIndices.push_back(i);

    inputPointsParent->populateDataForDimensions<std::vector<float>, std::vector<unsigned int>>(attribute_data, enabledDimensionsIndices);
    inputPointsParent->getGlobalIndices(pointIDsGlobal);

    // Image size
    ImgSize imgSize;
    imgSize.width = inputImages->getImageSize().width();
    imgSize.height = inputImages->getImageSize().height();

    // If the data is a subset: pointIDsGlobal != pointIDsFocus
    if (!inputPoints->isFull())
    {
        std::set_difference(pointIDsGlobal.begin(), pointIDsGlobal.end(), 
                            pointIDsFocus.begin(), pointIDsFocus.end(),
                            std::inserter(contextIDsGlobal, contextIDsGlobal.end()));
    }

    // Background IDs
    mv::Dataset<Points> backgroundDataset = static_cast<mv::Dataset<Points>>(_spidrSettingsAction->getBackgroundSelectionAction().getBackgroundDataset());
    if (backgroundDataset.isValid())
    {
        if (!_spidrSettingsAction->getBackgroundSelectionAction().getIDsInData())  // use the global IDs of the background dataset (it is a subset of the inputPoints dataset)
        {
            backgroundDataset->getGlobalIndices(backgroundIDsGlobal);

            qDebug() << "SpidrPlugin: Use background IDs from dataset " << backgroundDataset->getGuiName() << " (using its global point IDs)" << backgroundIDsGlobal.size() << " background points";
        }
        else   // background dataset contains the background IDs
        {
            // Check of the dimensions and number of points make sense
            if ( backgroundDataset->getNumDimensions() == 1 && backgroundDataset->getNumPoints() < inputPointsParent->getNumPoints() )
            {
                backgroundDataset->visitFromBeginToEnd([&backgroundIDsGlobal](auto beginBackgroundDataset, auto endBackgroundDataset) {
                    backgroundIDsGlobal.insert(backgroundIDsGlobal.begin(), beginBackgroundDataset, endBackgroundDataset);
                });

                qDebug() << "SpidrPlugin: Use background IDs from dataset " << backgroundDataset->getGuiName() << " (using its data values): " << backgroundIDsGlobal.size() << " background points";
            }
        }
    }

    // complete Spidr parameters
    auto& spidrParameters = _spidrSettingsAction->getSpidrParameters();
    spidrParameters._numPoints = pointIDsGlobal.size();
    spidrParameters._numForegroundPoints = pointIDsGlobal.size() - backgroundIDsGlobal.size();
    spidrParameters._numDims = numEnabledDimensions;
    spidrParameters._imgSize = imgSize;

    // Setup worker classes with data and parameters
    qDebug() << "SpidrPlugin: Initialize settings";
    
    // set the data and all the parameters
    _spidrAnalysisWrapper->setup(attribute_data, pointIDsGlobal, QString("sp-emb_hdps"), backgroundIDsGlobal, contextIDsGlobal, spidrParameters);

    // Start spatial analysis in worker thread
    _workerThreadSpidr = new QThread();
    _workerThreadtSNE = new QThread();

    _spidrAnalysisWrapper->moveToThread(_workerThreadSpidr);
    _tnseWrapper->moveBufferToThread(_workerThreadtSNE);
    _tnseWrapper->moveToThread(_workerThreadtSNE);

    // delete threads after work is done
    connect(_workerThreadSpidr, &QThread::finished, _workerThreadSpidr, &QObject::deleteLater);
    connect(_workerThreadtSNE, &QThread::finished, _workerThreadtSNE, &QObject::deleteLater);

    // connect wrappers
    connect(_spidrAnalysisWrapper.get(), &SpidrAnalysisQtWrapper::finishedKnn, this, &SpidrPlugin::tsneComputation);
    connect(this, &SpidrPlugin::startAnalysis,_spidrAnalysisWrapper.get(), &SpidrAnalysisQtWrapper::spatialAnalysis);
    connect(this, &SpidrPlugin::starttSNE, _tnseWrapper.get(), &TsneComputationQtWrapper::compute);


    qDebug() << "SpidrPlugin: Start Analysis";
    _spidrSettingsAction->getGeneralSpidrSettingsAction().getComputationAction().getRunningAction().setChecked(true);
    _workerThreadSpidr->start();
    emit startAnalysis();   // trigger computation in other thread
}

void SpidrPlugin::tsneComputation()
{
    // is called once knn computation is finished in _spidrAnalysisWrapper
    std::vector<int> _knnIds;
    std::vector<float> _knnDists;
    std::tie(_knnIds, _knnDists) = _spidrAnalysisWrapper->getKnn();
    _tnseWrapper->setup(_knnIds, _knnDists, _spidrAnalysisWrapper->getParameters()); // maybe I have to do this differently by sending a signal and getting hte values as a return...
    _workerThreadtSNE->start();
    emit starttSNE();
}

void SpidrPlugin::onFinishedEmbedding() {
    auto embedding = getOutputDataset<Points>();

    std::vector<float> embWithBg;
    _spidrAnalysisWrapper->addBackgroundToEmbedding(embWithBg, _tnseWrapper->outputRef());

    assert(embWithBg.size() % 2 == 0);
    assert(embWithBg.size() == _spidrAnalysisWrapper->getNumEmbPoints() * 2);

    qDebug() << "SpidrPlugin: Publishing final embedding";

    embedding->setData(embWithBg.data(), embWithBg.size() / 2, 2);
    events().notifyDatasetDataChanged(getOutputDataset());

    qDebug() << "SpidrPlugin: Done.";
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
        _tnseWrapper->stopGradientDescent();
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

SpidrPluginFactory::SpidrPluginFactory()
{
    setIcon(mv::util::StyledIcon(mv::gui::createPluginIcon("HSNE")));
}

AnalysisPlugin* SpidrPluginFactory::produce()
{
    return new SpidrPlugin(this);
}

PluginTriggerActions SpidrPluginFactory::getPluginTriggerActions(const mv::Datasets& datasets) const
{
    PluginTriggerActions pluginTriggerActions;

    const auto getPluginInstance = [this](const Dataset<Points>& dataset) -> SpidrPlugin* {
        return dynamic_cast<SpidrPlugin*>(plugins().requestPlugin(getKind(), { dataset }));
    };

    if (PluginFactory::areAllDatasetsOfTheSameType(datasets, ImageType)) {
        if (datasets.count() >= 1) {
            auto pluginTriggerAction = new PluginTriggerAction(const_cast<SpidrPluginFactory*>(this), this, "Spidr analysis", "Perform spatially informed t-SNE analysis", icon(), [this, getPluginInstance, datasets](PluginTriggerAction& pluginTriggerAction) -> void {
                for (const auto& dataset : datasets)
                    getPluginInstance(dataset);
            });

            pluginTriggerActions << pluginTriggerAction;
        }
    }

    return pluginTriggerActions;
}
