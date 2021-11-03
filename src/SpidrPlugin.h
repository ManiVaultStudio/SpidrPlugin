#pragma once

#include <AnalysisPlugin.h>
#include "SpidrSettingsAction.h"
#include "DimensionSelectionAction.h"

#include <memory>

#include "SpidrAnalysisQtWrapper.h"
#include "TsneComputationQtWrapper.h"

#include "Application.h"  // form hdps

class SpidrSettingsWidget;

using namespace hdps::plugin;

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
    SpidrPlugin(const PluginFactory* factory);
    ~SpidrPlugin(void) override;

    void init() override;

    void onDataEvent(hdps::DataEvent* dataEvent);

    void startComputation();
    void stopComputation();

    SpidrSettingsAction& getGeneralSpidrSettingsAction() { return _spidrSettingsAction; }
    DimensionSelectionAction& getDimensionSelectionAction() { return _dimensionSelectionAction; }


public slots:
    void onFinishedEmbedding();

    void onPublishFeatures(const unsigned int dataFeatsSize);

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
    DimensionSelectionAction    _dimensionSelectionAction;      /** Dimension selection settings action */

    QString _embeddingName;                             /*!<> */
    QThread* _workerThreadSpidr;                        /*!<> */
    QThread* _workerThreadtSNE;                         /*!<> */

    QString _inputSourceName;    // the input image name is available with getInputDatasetName()
    QString _outputDataName;     // since the output has a different data type than the input (Points intead of Images) we have to create it separately 
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
    QIcon getIcon() const override;

    AnalysisPlugin* produce() override;

    hdps::DataTypes supportedDataTypes() const override;

};
