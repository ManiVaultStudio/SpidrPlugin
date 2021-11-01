#include "GeneralSpidrSettingsAction.h"
#include "SpidrSettingsAction.h"

#include <QLabel>
#include <QPushButton>
#include <QGridLayout>

#include "FeatureUtils.h"


using namespace hdps::gui;

GeneralSpidrSettingsAction::GeneralSpidrSettingsAction(SpidrSettingsAction& spidrSettingsAction) :
    GroupAction(&spidrSettingsAction, true),
    _spidrSettingsAction(spidrSettingsAction),
    _knnTypeAction(this, "KNN Type"),
    _distanceMetricAction(this, "Distance metric"),
    _kernelSize(this, "Neighborhood size"),
    _kernelWeight(this, "Neighborhood weighting"), 
    _histBinSizeHeur(this, "Histo. Bin Heuristic"),
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
    _histBinSizeHeur.setDefaultWidgetFlags(OptionAction::ComboBox);
    _numIterationsAction.setDefaultWidgetFlags(IntegralAction::SpinBox);
    _perplexityAction.setDefaultWidgetFlags(IntegralAction::SpinBox | IntegralAction::Slider);

    _knnTypeAction.initialize(QStringList({ "HNSW", "Exact kNN"}), "HNSW", "HNSW");
    _distanceMetricAction.initialize(QStringList({ "Texture Hist. (QF)", "Texture Hist. (Hel)", "Covmat & Means (Bat)", "Covmat & Means (Fro)", "Local Moran's I (L2)",
                                                    "Local Geary's C (L2)", "Point Clound (Chamfer)", "Point Clound(Hausdorff)"}), "Texture Hist. (QF)", "Texture Hist. (QF)");
    _kernelWeight.initialize(QStringList({ "Uniform", "Gaussian" }), "Uniform", "Uniform");
    _histBinSizeHeur.initialize(QStringList({ "Manual", "Rice", "Sturges", "Sqrt" }), "Rice", "Rice");
    _kernelSize.initialize(1, 50, 1, 1);
    _numIterationsAction.initialize(1, 10000, 1000, 1000);
    _perplexityAction.initialize(2, 100, 30, 30);

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
            feat = feature_type::TEXTURE_HIST_1D;
            dist = distance_metric::METRIC_QF;
            break;
        case 1: // Texture Hist. (Hel)
            feat = feature_type::TEXTURE_HIST_1D;
            dist = distance_metric::METRIC_HEL;
            break;
        case 2: // Covmat & Means (Bat)
            feat = feature_type::MULTIVAR_NORM;
            dist = distance_metric::METRIC_BHATTACHARYYA;
            break;
        case 3: // Covmat & Means (Fro)
            feat = feature_type::MULTIVAR_NORM;
            dist = distance_metric::METRIC_FROBENIUS_CovMat;
            break;
        case 4: // Local Moran's I (L2)
            feat = feature_type::LOCALMORANSI;
            dist = distance_metric::METRIC_EUC;
            break;
        case 5: // Local Geary's C (L2)
            feat = feature_type::LOCALGEARYC;
            dist = distance_metric::METRIC_EUC;
            break;
        case 6: // Point Clound (Chamfer)
            feat = feature_type::PCLOUD;
            dist = distance_metric::METRIC_CHA;
            break;
        case 7: // Point Clound(Hausdorff)
            feat = feature_type::PCLOUD;
            dist = distance_metric::METRIC_HAU;
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

        switch (_knnTypeAction.getCurrentIndex()) {
        case 0:
            neigh_weight = loc_Neigh_Weighting::WEIGHT_UNIF;
            break;
        case 1:
            neigh_weight = loc_Neigh_Weighting::WEIGHT_GAUS;
            break;
        }

        _spidrSettingsAction.getSpidrParameters()._neighWeighting = neigh_weight;
    };

    const auto updateHistBinSizeHeur = [this]() -> void {
        unsigned int num_bins = 5;

        auto kernelSize = _kernelSize.getValue();
        auto numLocNeighbors = (2 * kernelSize + 1) * (2 * kernelSize + 1);

        switch (_knnTypeAction.getCurrentIndex()) {
        case 0: // Manual
            num_bins = 5;   // TODO expose to GUI
            qDebug() << "Manual hist bin number not implemented - default to 5";
            break;
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

        _spidrSettingsAction.getSpidrParameters()._numHistBins = num_bins;

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
