#pragma once

#include <AnalysisPlugin.h>

#include "TsneAnalysis.h"
#include "FeatureExtraction.h"
#include "PointData.h"
class SpidrSettingsWidget;

using namespace hdps::plugin;
using namespace hdps::gui;

// =============================================================================
// View
// =============================================================================

class SpidrPlugin : public QObject, public AnalysisPlugin
{
    Q_OBJECT   
public:
    SpidrPlugin();
    ~SpidrPlugin(void) override;
    
    void init() override;

    void dataAdded(const QString name) Q_DECL_OVERRIDE;
    void dataChanged(const QString name) Q_DECL_OVERRIDE;
    void dataRemoved(const QString name) Q_DECL_OVERRIDE;
    void selectionChanged(const QString dataName) Q_DECL_OVERRIDE;
    hdps::DataTypes supportedDataTypes() const Q_DECL_OVERRIDE;

    SettingsWidget* const getSettings() override;

    void startComputation();
    void stopComputation();

public slots:
    void dataSetPicked(const QString& name);
    void onKnnAlgorithmPicked(const int index);
    void onDistanceMetricPicked(const int index);
    void onNewEmbedding();

private:
    void initializeTsne();
    /**
    * Takes a set of selected points and retrieves teh corresponding attributes in all enabled dimensions 
    * @param points Selected points in the data set
    * @param numDimensions Will contains the number of enabled dimensions 
    * @param data Will contain the attributes for all points, size: numPoints * numDimensions
    */
    void retrieveData(const Points points, unsigned int& numDimensions, std::vector<float>& data);

    TsneAnalysis _tsne;
    FeatureExtraction _featExtraction;
    std::unique_ptr<SpidrSettingsWidget> _settings;
    QString _embeddingName;
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
