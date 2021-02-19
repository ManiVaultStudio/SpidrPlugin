#pragma once

#include <AnalysisPlugin.h>

#include <memory>

#include <QtCore>
#include <QSize>

#include "SpidrAnalysisQt.h"
#include "PointData.h"

#include "Application.h" 

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

    void dataAdded(const QString name) Q_DECL_OVERRIDE;
    void dataChanged(const QString name) Q_DECL_OVERRIDE;
    void dataRemoved(const QString name) Q_DECL_OVERRIDE;
    void selectionChanged(const QString dataName) Q_DECL_OVERRIDE;
    hdps::DataTypes supportedDataTypes() const Q_DECL_OVERRIDE;

    hdps::gui::SettingsWidget* const getSettings() override;

    void startComputation();
    void stopComputation();

public slots:
    void dataSetPicked(const QString& name);
    void onNewEmbedding();
    void onFinishedEmbedding();

signals:
    void embeddingComputationStopped();

private:
    /*!
     * 
     * 
     */
    void initializeAnalysisSettings();

    /**
    * Takes a set of selected points and retrieves teh corresponding attributes in all enabled dimensions 
    * @param dataName Name of data set as defined in hdps core
    * @param imgSize Will contain the size of the image (width and height)
    * @param pointIDsGlobal Will contain IDs of selected points in the data set
    * @param numDimensions Will contain the number of enabled dimensions 
    * @param data Will contain the attributes for all points, size: pointIDsGlobal.size() * numDimensions
    */
    /*!
     * 
     * 
     * \param dataName
     * \param pointIDsGlobal
     * \param attribute_data
     * \param numDims
     * \param imgSize
     * \param backgroundIDsGlobal
     */
    void retrieveData(QString dataName, std::vector<unsigned int>& pointIDsGlobal, std::vector<float>& attribute_data, unsigned int& numDims, ImgSize& imgSize, std::vector<unsigned int>& backgroundIDsGlobal);

    SpidrAnalysisQt _spidrAnalysisQt;                       /*!<> */
    std::unique_ptr<SpidrSettingsWidget> _settings;     /*!<> */
    QString _embeddingName;                             /*!<> */
    QThread workerThread;                               /*!<> */
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
