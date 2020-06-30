#pragma once

#include <AnalysisPlugin.h>

#include "TsneAnalysis.h"
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

    TsneAnalysis _tsne;
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
