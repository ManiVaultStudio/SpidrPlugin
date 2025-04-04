#pragma once

#include <actions/WidgetAction.h>
#include <AnalysisPlugin.h>

#include <ForegroundTask.h>

#include <memory>

class SpidrSettingsAction;
class SpidrAnalysisQtWrapper;
class TsneComputationQtWrapper;

using namespace mv::plugin;
using namespace mv::gui;

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

public slots:
    void onFinishedEmbedding();

private slots:
    void tsneComputation();

signals:
    void embeddingComputationStopped();
    void startAnalysis();
    void starttSNE();

private:

    std::unique_ptr<SpidrAnalysisQtWrapper>     _spidrAnalysisWrapper;          /** Spidr feature and knn computation wrapper */
    std::unique_ptr<TsneComputationQtWrapper>   _tnseWrapper;                   /** t-sne computation wrapper */

    std::unique_ptr<SpidrSettingsAction>        _spidrSettingsAction;           /** Spidr settings action */

    QThread*                                    _workerThreadSpidr;             /** worker thread for spidr feature amd knn computation */
    QThread*                                    _workerThreadtSNE;              /** worker thread for t-SNE layout computation */

    mv::ForegroundTask                          _computationPreparationTask;    /** Task for reporting computation preparation progress */
};

// =============================================================================
// Factory
// =============================================================================

class SpidrPluginFactory : public AnalysisPluginFactory
{
    Q_INTERFACES(mv::plugin::AnalysisPluginFactory mv::plugin::PluginFactory)
        Q_OBJECT
        Q_PLUGIN_METADATA(IID   "nl.tudelft.SpidrPlugin"
                          FILE  "SpidrPlugin.json")

public:
    SpidrPluginFactory(void);
    ~SpidrPluginFactory(void) override {}

    AnalysisPlugin* produce() override;

    /**
     * Get plugin trigger actions given \p datasets
     * @param datasets Vector of input datasets
     * @return Vector of plugin trigger actions
     */
    PluginTriggerActions getPluginTriggerActions(const mv::Datasets& datasets) const override;
};
