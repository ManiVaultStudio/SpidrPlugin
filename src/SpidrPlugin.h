#pragma once

#include <actions/WidgetAction.h>
#include <AnalysisPlugin.h>

#include "SpidrSettingsAction.h"

#include <memory>

#include "SpidrAnalysisQtWrapper.h"
#include "TsneComputationQtWrapper.h"

class SpidrSettingsWidget;

using namespace hdps::plugin;
using namespace hdps::gui;

// =============================================================================
// Analysis Plugin
// =============================================================================

/*!
 *
 *
 */
class SpidrPlugin : public AnalysisPlugin
{
    Q_OBJECT
public:
    SpidrPlugin(const PluginFactory* factory);
    ~SpidrPlugin(void) override;

    void init() override;

    void startComputation();
    void stopComputation();


public: // Action getters

    SpidrSettingsAction& getGeneralSpidrSettingsAction() { return _spidrSettingsAction; }


public slots:
    void onFinishedEmbedding();

private slots:
    void tsneComputation();

signals:
    void embeddingComputationStopped();
    void startAnalysis();
    void starttSNE();

private:

    SpidrAnalysisQtWrapper _spidrAnalysisWrapper;
    TsneComputationQtWrapper _tnseWrapper;

    SpidrSettingsAction         _spidrSettingsAction;           /** Spidr settings action */

    QString _embeddingName;                             /*!<> */
    QThread* _workerThreadSpidr;                        /*!<> */
    QThread* _workerThreadtSNE;                         /*!<> */

    //QString _inputSourceName;    // the input image name is available with getInputDatasetName()
    //QString _outputDataName;     // since the output has a different data type than the input (Points intead of Images) we have to create it separately 
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

    /** Returns the plugin icon */
    QIcon getIcon(const QColor& color = Qt::black) const override;

    AnalysisPlugin* produce() override;

    /**
     * Get plugin trigger actions given \p datasets
     * @param datasets Vector of input datasets
     * @return Vector of plugin trigger actions
     */
    PluginTriggerActions getPluginTriggerActions(const hdps::Datasets& datasets) const override;
};
