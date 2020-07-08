#pragma once

#include <AnalysisPlugin.h>

#include <QtCore>
#include <QSize>

#include "TsneAnalysis.h"
#include "DistanceCalculation.h"
#include "FeatureExtraction.h"
#include "PointData.h"
#include "KNNUtils.h"

class SpidrSettingsWidget;

using namespace hdps::plugin;
using namespace hdps::gui;

class Parameters {
public:
    Parameters() :
        _perplexity(30),
        _perplexity_multiplier(3),
        _aknn_algorithm(knn_library::KNN_HSNW),
        _aknn_metric(knn_distance_metric::KNN_METRIC_QF)
    {}

public:
    float               _perplexity;            //! Perplexity value in evert distribution.
    int                 _perplexity_multiplier; //! Multiplied by the perplexity gives the number of nearest neighbors used
    knn_library         _aknn_algorithm;
    knn_distance_metric _aknn_metric;
    unsigned int        _numHistBins;           // to be set in FeatureExtraction
    unsigned int        _nn;                    // number of nearest neighbors, determined by _perplexity*_perplexity_multiplier + 1; to be set in DistanceCalculation
    unsigned int        _numPoints;             // to be set in SpidrPlugin
    unsigned int        _numDims;               // to be set in SpidrPlugin
};


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
    void initializeTsneSettings();

    /**
    * Takes a set of selected points and retrieves teh corresponding attributes in all enabled dimensions 
    * @param dataName Name of data set as defined in hdps core
    * @param imgSize Will contain the size of the image (width and height)
    * @param pointIDsGlobal Will contain IDs of selected points in the data set
    * @param numDimensions Will contain the number of enabled dimensions 
    * @param data Will contain the attributes for all points, size: pointIDsGlobal.size() * numDimensions
    */
    void retrieveData(QString dataName, QSize& imgSize, std::vector<unsigned int>& pointIDsGlobal, std::vector<float>& data, Parameters& params);

    TsneAnalysis _tsne;
    DistanceCalculation _distCalc;
    FeatureExtraction _featExtraction;
    std::unique_ptr<SpidrSettingsWidget> _settings;
    QString _embeddingName;
    Parameters _params;
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
