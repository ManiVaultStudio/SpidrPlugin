#include "GeneralSpidrSettingsAction.h"
#include "SpidrSettingsAction.h"

#include <QLabel>
#include <QPushButton>
#include <QGridLayout>

#include "FeatureUtils.h"

#include "SpidrAnalysisParameters.h"  // get_feat_and_dist


using namespace hdps::gui;

GeneralSpidrSettingsAction::GeneralSpidrSettingsAction(SpidrSettingsAction& spidrSettingsAction) :
    GroupAction(&spidrSettingsAction, true),
    _spidrSettingsAction(spidrSettingsAction),
    _knnTypeAction(this, "KNN Type"),
    _distanceMetricAction(this, "Distance metric"),
    _kernelSize(this, "Neighborhood size"),
    _kernelWeight(this, "Neighborhood weighting"), 
    _histBinSizeAction(this),
    _numIterationsAction(this, "Number of iterations"),
    _perplexityAction(this, "Perplexity"),
    _computationAction(this),
    _resetAction(this, "Reset all")
{
    setText("Spidr");

    const auto& spidrParameters = _spidrSettingsAction.getSpidrParameters();

    _knnTypeAction.setDefaultWidgetFlags(OptionAction::ComboBox);
    _distanceMetricAction.setDefaultWidgetFlags(OptionAction::ComboBox);
    _kernelSize.setDefaultWidgetFlags(IntegralAction::SpinBox);
    _kernelWeight.setDefaultWidgetFlags(OptionAction::ComboBox);
    _numIterationsAction.setDefaultWidgetFlags(IntegralAction::SpinBox);
    _perplexityAction.setDefaultWidgetFlags(IntegralAction::SpinBox | IntegralAction::Slider);

    _knnTypeAction.initialize(QStringList({ "HNSW", "Exact kNN"}), "HNSW", "HNSW");
    // TODO: there must be a nice way to add the feat_dist here and later use it directly without a switch or if statement
    _distanceMetricAction.initialize(QStringList({ 
        "Texture Hist. (QF)",       // case 0
        "Texture Hist. (Hel)",      // case 1
        "Covmat & Means (Bat)",     // case 2
        "Covmat & Means (Fro)",     // case 3
        "Local Moran's I (L2)",     // case 4
        "Local Geary's C (L2)",     // case 5
        "Point Clound (Chamfer)",   // case 6
        "Point Clound (Hausdorff)", // case 7
        "Add XY Pos",               // case 9 
        "Add XY Pos (normed)"}),    // case 9 
        "Texture Hist. (QF)", "Texture Hist. (QF)");    // default

    _kernelWeight.initialize(QStringList({ "Uniform", "Gaussian" }), "Uniform", "Uniform");
    _kernelSize.initialize(1, 50, 1, 1);
    _numIterationsAction.initialize(1, 10000, 1000, 1000);
    _perplexityAction.initialize(2, 100, 30, 30);

    // set default values
    _spidrSettingsAction.getSpidrParameters().set_numNeighborsInEachDirection(_kernelSize.getValue());
    _spidrSettingsAction.getSpidrParameters()._numHistBins = _histBinSizeAction.getNumHistBinsAction().getValue();

    const auto updateKnnAlgorithm = [this]() -> void {
        knn_library knn_lib = knn_library::KNN_HNSW;

        switch (_knnTypeAction.getCurrentIndex()) {
        case 0:
            knn_lib = knn_library::KNN_HNSW;
            break;
        case 1:
            knn_lib = knn_library::KKN_EXACT;
            break;
        }

        _spidrSettingsAction.getSpidrParameters()._aknn_algorithm = knn_lib;
    };

    const auto updateDistanceMetric = [this]() -> void {
        feature_type feat = feature_type::TEXTURE_HIST_1D;
        distance_metric dist = distance_metric::METRIC_QF;

        switch (_distanceMetricAction.getCurrentIndex()) {
        case 0: // Texture Hist. (QF)
            std::tie(feat, dist) = get_feat_and_dist(feat_dist::HIST_QF);
            break;
        case 1: // Texture Hist. (Hel)
            std::tie(feat, dist) = get_feat_and_dist(feat_dist::HIST_HEL);
            break;
        case 2: // Covmat & Means (Bat)
            std::tie(feat, dist) = get_feat_and_dist(feat_dist::MVN_BHAT);
            break;
        case 3: // Covmat & Means (Fro)
            std::tie(feat, dist) = get_feat_and_dist(feat_dist::MVN_FRO);
            break;
        case 4: // Local Moran's I (L2)
            std::tie(feat, dist) = get_feat_and_dist(feat_dist::LMI_EUC);
            break;
        case 5: // Local Geary's C (L2)
            std::tie(feat, dist) = get_feat_and_dist(feat_dist::LGC_EUC);
            break;
        case 6: // Point Clound (Chamfer)
            std::tie(feat, dist) = get_feat_and_dist(feat_dist::PC_CHA);
            break;
        case 7: // Point Clound(Hausdorff)
            std::tie(feat, dist) = get_feat_and_dist(feat_dist::PC_HAU);
            break;
        case 8: // Add XY Pos (normed)
            std::tie(feat, dist) = get_feat_and_dist(feat_dist::PIXEL_LOCATION);
            break;
        case 9: // Add XY Pos (normed)
            std::tie(feat, dist) = get_feat_and_dist(feat_dist::PIXEL_LOCATION_NORM);
            break;
        }

        _spidrSettingsAction.getSpidrParameters()._featureType = feat;
        _spidrSettingsAction.getSpidrParameters()._aknn_metric = dist;
    };

    const auto updateKerneSize = [this]() -> void {
        _spidrSettingsAction.getSpidrParameters().set_numNeighborsInEachDirection(_kernelSize.getValue());
    };

    const auto updateKernelWeight = [this]() -> void {
        loc_Neigh_Weighting neigh_weight = loc_Neigh_Weighting::WEIGHT_UNIF;

        switch (_kernelWeight.getCurrentIndex()) {
        case 0:
            neigh_weight = loc_Neigh_Weighting::WEIGHT_UNIF;
            break;
        case 1:
            neigh_weight = loc_Neigh_Weighting::WEIGHT_GAUS;
            break;
        }

        _spidrSettingsAction.getSpidrParameters()._neighWeighting = neigh_weight;
    };

    const auto adjustHistBinNum = [this]() -> void {
        unsigned int num_bins = 5;

        auto kernelSize = _kernelSize.getValue();
        auto numLocNeighbors = (2 * kernelSize + 1) * (2 * kernelSize + 1);

        switch (_histBinSizeAction.getHistBinSizeHeur().getCurrentIndex()) {
        case 0: // Manual
            // do nothing
            return;
        case 1: // Rice
            num_bins = RiceBinSize(numLocNeighbors);
            break;
        case 2: // Sturges
            num_bins = SturgesBinSize(numLocNeighbors);
            break;
        case 3: // Sqrt
            num_bins = SqrtBinSize(numLocNeighbors);
            break;
        }

        _histBinSizeAction.getNumHistBinsAction().setValue(num_bins);

    };

    const auto updateHistBinNum = [this]() -> void {
        _spidrSettingsAction.getSpidrParameters()._numHistBins = _histBinSizeAction.getNumHistBinsAction().getValue();
    };

    const auto updateNumIterations = [this]() -> void {
        _spidrSettingsAction.getSpidrParameters()._numIterations = _numIterationsAction.getValue();
    };

    const auto updatePerplexity = [this]() -> void {
        _spidrSettingsAction.getSpidrParameters().set_perplexity(_perplexityAction.getValue());
    };

    const auto isResettable = [this]() -> bool {
        if (_knnTypeAction.isResettable())
            return true;

        if (_distanceMetricAction.isResettable())
            return true;

        if (_numIterationsAction.isResettable())
            return true;

        if (_histBinSizeAction.isResettable())
            return true;

        if (_kernelSize.isResettable())
            return true;

        if (_kernelWeight.isResettable())
            return true;

        if (_perplexityAction.isResettable())
            return true;

        return false;
    };

    const auto updateReset = [this, isResettable]() -> void {
        _resetAction.setEnabled(isResettable());
    };

    const auto updateReadOnly = [this]() -> void {
        const auto enable = !isReadOnly();

        _knnTypeAction.setEnabled(enable);
        _distanceMetricAction.setEnabled(enable);
        _numIterationsAction.setEnabled(enable);
        _perplexityAction.setEnabled(enable);
        _kernelSize.setEnabled(enable);
        _kernelWeight.setEnabled(enable);
        _histBinSizeAction.setEnabled(enable);
        _resetAction.setEnabled(enable);
    };

    connect(&_knnTypeAction, &OptionAction::currentIndexChanged, this, [this, updateDistanceMetric, updateReset](const std::int32_t& currentIndex) {
        updateDistanceMetric();
        updateReset();
    });

    connect(&_distanceMetricAction, &OptionAction::currentIndexChanged, this, [this, updateDistanceMetric, updateReset](const std::int32_t& currentIndex) {
        updateDistanceMetric();
        updateReset();
    });

    connect(&_kernelSize, &IntegralAction::valueChanged, this, [this, updateKerneSize, adjustHistBinNum, updateHistBinNum, updateReset](const std::int32_t& value) {
        updateKerneSize();
        adjustHistBinNum();
        updateHistBinNum();
        updateReset();
    });

    connect(&(_histBinSizeAction.getHistBinSizeHeur()), &OptionAction::currentIndexChanged, this, [this, adjustHistBinNum, updateHistBinNum, updateReset](const std::int32_t& currentIndex) {
        adjustHistBinNum();
        updateHistBinNum();
        updateReset();
        });

    connect(&(_histBinSizeAction.getNumHistBinsAction()), &IntegralAction::valueChanged, this, [this, updateHistBinNum, updateReset](const std::int32_t& value){
        updateHistBinNum();
        updateReset();
     });

    connect(&_kernelWeight, &OptionAction::currentIndexChanged, this, [this, updateKernelWeight, updateReset](const std::int32_t& value) {
        updateKernelWeight();
        updateReset();
        });

    connect(&_numIterationsAction, &IntegralAction::valueChanged, this, [this, updateNumIterations, updateReset](const std::int32_t& value) {
        updateNumIterations();
        updateReset();
        });

    connect(&_perplexityAction, &IntegralAction::valueChanged, this, [this, updatePerplexity, updateReset](const std::int32_t& value) {
        updatePerplexity();
        updateReset();
    });

    connect(&_resetAction, &TriggerAction::triggered, this, [this](const std::int32_t& value) {
        _knnTypeAction.reset();
        _distanceMetricAction.reset();
        _numIterationsAction.reset();
        _perplexityAction.reset();
        _histBinSizeAction.reset();
        _kernelSize.reset();
        _kernelWeight.reset();
    });

    connect(this, &GroupAction::readOnlyChanged, this, [this, updateReadOnly](const bool& readOnly) {
        updateReadOnly();
    });

    updateKnnAlgorithm();
    updateDistanceMetric();
    updateNumIterations();
    updatePerplexity();
    updateReset();
    updateReadOnly();
}
