#pragma once

#include <AnalysisPlugin.h>

#include <memory>

#include <QtCore>
#include <QSize>

#include "SpidrAnalysisQt.h"
#include "TsneComputationQt.h"
#include "PointData.h"

#include "Application.h"  // form hdps

class SpidrSettingsWidget;

using namespace hdps::plugin;
// using namespace hdps::gui;


// =============================================================================
// Analysis Plugin
// =============================================================================

/*!
 *
 *
 */
class SpidrPlugin : public QObject, public AnalysisPlugin
{
    Q_OBJECT
public:
    SpidrPlugin();
    ~SpidrPlugin(void) override;

    void init() override;

    /** Returns the icon of this plugin */
    QIcon getIcon() const override {
        return hdps::Application::getIconFont("FontAwesome").getIcon("table");
    }

    void onDataEvent(hdps::DataEvent* dataEvent);

    hdps::gui::SettingsWidget* const getSettings() override;

    void startComputation();
    void stopComputation();

public slots:
    void dataSetPicked(const QString& name);
    void onNewEmbedding();
    void onFinishedEmbedding();

    void onPublishFeatures(const unsigned int dataFeatsSize);

private slots:
    void tsneComputation();

signals:
    void embeddingComputationStopped();
    void startAnalysis();
    void starttSNE();

private:

    /**
    * Takes a set of selected points and retrieves teh corresponding attributes in all enabled dimensions
    * @param dataName Name of data set as defined in hdps core
    * @param pointIDsGlobal  Will contain IDs of selected points in the data set
    * @param attribute_data  Will contain the attributes for all points, size: pointIDsGlobal.size() * numDimensions
    * @param numDims Will contain the number of dimensions
    * @param imgSize Will contain the size of the image (width and height)
    * @param backgroundIDsGlobal Will contain the global pixel IDs of a background that is ignored during t-SNE computation but taken into account for feature extraction
    */
    void retrieveData(QString dataName, std::vector<unsigned int>& pointIDsGlobal, std::vector<float>& attribute_data, unsigned int& numDims, ImgSize& imgSize, std::vector<unsigned int>& backgroundIDsGlobal);

    SpidrAnalysisQtWrapper* _spidrAnalysisWrapper;       /*!<> */
    TsneComputationQt* _tnseWrapper;                     /*!<> */
    std::unique_ptr<SpidrSettingsWidget> _settings;     /*!<> */
    QString _embeddingName;                             /*!<> */
    QThread workerThreadSpidr;                               /*!<> */
    QThread workerThreadtSNE;                               /*!<> */
};

// =============================================================================
// Factory
// =============================================================================

class SpidrPluginFactory : public AnalysisPluginFactory
{
    Q_INTERFACES(hdps::plugin::AnalysisPluginFactory hdps::plugin::PluginFactory)
        Q_OBJECT
        Q_PLUGIN_METADATA(IID   "nl.tudelft.SpidrPlugin"
            FILE  "SpidrPlugin.json")

public:
    SpidrPluginFactory(void) {}
    ~SpidrPluginFactory(void) override {}

    AnalysisPlugin* produce() override;
};
