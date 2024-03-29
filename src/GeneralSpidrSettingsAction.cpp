#include "GeneralSpidrSettingsAction.h"
#include "SpidrSettingsAction.h"

#include <QLabel>
#include <QPushButton>
#include <QGridLayout>
#include <QVariant> 

#include "FeatureUtils.h"
#include "SpidrAnalysisParameters.h"  // get_feat_and_dist

using namespace mv::gui;

Q_DECLARE_METATYPE(feat_dist);      // in order to use QVariant::fromValue with custom type feat_dist 

GeneralSpidrSettingsAction::GeneralSpidrSettingsAction(SpidrSettingsAction& spidrSettingsAction) :
    GroupAction(&spidrSettingsAction, "GeneralSpidrSettingsAction", true),
    _spidrSettingsAction(spidrSettingsAction),
    _knnTypeAction(this, "KNN Type"),
    _distanceMetricAction(this, "Distance metric"),
    _kernelSize(this, "Neighborhood size"),
    _kernelWeight(this, "Neighborhood weighting"), 
    _histBinSizeAction(this),
    _numIterationsAction(this, "Number of iterations"),
    _numberOfComputatedIterationsAction(this, "Computed iterations", 0, 1000000000, 0),
    _perplexityAction(this, "Perplexity"),
    _pixelWeightAction(this, "Pixel Weight"),
    _computationAction(this)
{
    setText("Spidr");
    setObjectName("Spidr");

    addAction(&_knnTypeAction);
    addAction(&_distanceMetricAction);
    addAction(&_kernelSize);
    addAction(&_kernelWeight);
    addAction(&_histBinSizeAction);
    addAction(&_numIterationsAction);
    addAction(&_numberOfComputatedIterationsAction);
    addAction(&_perplexityAction);
    addAction(&_pixelWeightAction);
    addAction(&_computationAction);

    const auto& spidrParameters = _spidrSettingsAction.getSpidrParameters();

    _numberOfComputatedIterationsAction.setEnabled(false);

    _knnTypeAction.setDefaultWidgetFlags(OptionAction::ComboBox);
    _distanceMetricAction.setDefaultWidgetFlags(OptionAction::ComboBox);
    _kernelSize.setDefaultWidgetFlags(IntegralAction::SpinBox);
    _kernelWeight.setDefaultWidgetFlags(OptionAction::ComboBox);
    _numIterationsAction.setDefaultWidgetFlags(IntegralAction::SpinBox);
    _numberOfComputatedIterationsAction.setDefaultWidgetFlags(IntegralAction::LineEdit);
    _perplexityAction.setDefaultWidgetFlags(IntegralAction::SpinBox | IntegralAction::Slider);
    _pixelWeightAction.setDefaultWidgetFlags(IntegralAction::SpinBox | IntegralAction::Slider);

    _knnTypeAction.initialize(QStringList({ "HNSW", "Exact kNN"}), "HNSW");

     //Use an item model to add feat_dist enums to each drop down menu entry
    _distanceItemModel = std::make_shared<QStandardItemModel>(0, 1);

    _distanceItemList.append(std::make_shared<QStandardItem>("Texture Hist. (QF)"));
    _distanceItemList.last()->setData(QVariant::fromValue(feat_dist::HIST_QF));

    _distanceItemList.append(std::make_shared<QStandardItem>("Point Clound (Chamfer)"));
    _distanceItemList.last()->setData(QVariant::fromValue(feat_dist::PC_CHA));

    _distanceItemList.append(std::make_shared<QStandardItem>("Covmat & Means (Bat)"));
    _distanceItemList.last()->setData(QVariant::fromValue(feat_dist::MVN_BHAT));

    _distanceItemList.append(std::make_shared<QStandardItem>("Local Moran's I (L2)"));
    _distanceItemList.last()->setData(QVariant::fromValue(feat_dist::LMI_EUC));

    _distanceItemList.append(std::make_shared<QStandardItem>("XY Pos (euclid weighted)"));
    _distanceItemList.last()->setData(QVariant::fromValue(feat_dist::XY_EUCW));

    _distanceItemList.append(std::make_shared<QStandardItem>("Point Clound (Hausdorff)"));
    _distanceItemList.last()->setData(QVariant::fromValue(feat_dist::PC_HAU));

    _distanceItemList.append(std::make_shared<QStandardItem>("Texture Hist. (Hel)"));
    _distanceItemList.last()->setData(QVariant::fromValue(feat_dist::HIST_HEL));

    _distanceItemList.append(std::make_shared<QStandardItem>("XY Pos (normed)"));
    _distanceItemList.last()->setData(QVariant::fromValue(feat_dist::XYRNORM_EUC));

    // add all feat_dist entries
    for (auto& item : _distanceItemList)
        _distanceItemModel->appendRow(item.get());

    _distanceMetricAction.initialize(*_distanceItemModel, "Texture Hist. (QF)", "Texture Hist. (QF)");

    _kernelWeight.initialize(QStringList({ "Uniform", "Gaussian" }), "Uniform");
    _kernelSize.initialize(1, 50, 1);
    _numIterationsAction.initialize(1, 10000, 1000);
    _perplexityAction.initialize(2, 100, 30);
    _pixelWeightAction.initialize(0, 1, 0.5, 0.5);

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

        // Get data (feat_dist) from the itemModel attached to the drop down menu
        auto index = _distanceMetricAction.getCurrentIndex();
        auto model = dynamic_cast<const QStandardItemModel*>(_distanceMetricAction.getModel());
        auto data = model->item(index, 0)->data();

        // Set feature attribute and distance metric
        feat_dist seleted_feat_dist = data.value<feat_dist>();
        std::tie(feat, dist) = get_feat_and_dist(seleted_feat_dist);

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

    const auto updatePixelWeight = [this]() -> void {
        _spidrSettingsAction.getSpidrParameters()._pixelWeight = _pixelWeightAction.getValue();
    };


    //const auto isResettable = [this]() -> bool {
    //    if (_knnTypeAction.isResettable())
    //        return true;

    //    if (_distanceMetricAction.isResettable())
    //        return true;

    //    if (_numIterationsAction.isResettable())
    //        return true;

    //    if (_histBinSizeAction.isResettable())
    //        return true;

    //    if (_kernelSize.isResettable())
    //        return true;

    //    if (_kernelWeight.isResettable())
    //        return true;

    //    if (_perplexityAction.isResettable())
    //        return true;

    //    if (_pixelWeightAction.isResettable())
    //        return true;

    //    return false;
    //};

    // call this after updateDistanceMetric
    const auto updateEnabledSettings = [this]() -> void {

        // only change histograms bin for appropriate feature
        if (_spidrSettingsAction.getSpidrParameters()._featureType == feature_type::TEXTURE_HIST_1D)
            _histBinSizeAction.setEnabled(true);
        else
            _histBinSizeAction.setEnabled(false);


        // only pixel-attribute weight for appropriate feature 
        if (_spidrSettingsAction.getSpidrParameters()._featureType == feature_type::PIXEL_LOCATION)
            _pixelWeightAction.setEnabled(true);
        else
            _pixelWeightAction.setEnabled(false);
    };


    const auto updateReadOnly = [this]() -> void {
        const auto enable = !isReadOnly();

        _knnTypeAction.setEnabled(enable);
        _distanceMetricAction.setEnabled(enable);
        _numIterationsAction.setEnabled(enable);
        _perplexityAction.setEnabled(enable);
        _pixelWeightAction.setEnabled(enable);
        _kernelSize.setEnabled(enable);
        _kernelWeight.setEnabled(enable);
        _histBinSizeAction.setEnabled(enable);
    };

    connect(&_knnTypeAction, &OptionAction::currentIndexChanged, this, [this, updateDistanceMetric](const std::int32_t& currentIndex) {
        updateDistanceMetric();
    });

    connect(&_distanceMetricAction, &OptionAction::currentIndexChanged, this, [this, updateDistanceMetric, updateEnabledSettings](const std::int32_t& currentIndex) {
        updateDistanceMetric();
        updateEnabledSettings();
    });

    connect(&_kernelSize, &IntegralAction::valueChanged, this, [this, updateKerneSize, adjustHistBinNum, updateHistBinNum](const std::int32_t& value) {
        updateKerneSize();
        adjustHistBinNum();
        updateHistBinNum();
    });

    connect(&(_histBinSizeAction.getHistBinSizeHeur()), &OptionAction::currentIndexChanged, this, [this, adjustHistBinNum, updateHistBinNum](const std::int32_t& currentIndex) {
        adjustHistBinNum();
        updateHistBinNum();
        });

    connect(&(_histBinSizeAction.getNumHistBinsAction()), &IntegralAction::valueChanged, this, [this, updateHistBinNum](const std::int32_t& value){
        updateHistBinNum();
     });

    connect(&_kernelWeight, &OptionAction::currentIndexChanged, this, [this, updateKernelWeight](const std::int32_t& value) {
        updateKernelWeight();
        });

    connect(&_numIterationsAction, &IntegralAction::valueChanged, this, [this, updateNumIterations](const std::int32_t& value) {
        updateNumIterations();
        });

    connect(&_perplexityAction, &IntegralAction::valueChanged, this, [this, updatePerplexity](const std::int32_t& value) {
        updatePerplexity();
    });

    connect(&_pixelWeightAction, &DecimalAction::valueChanged, this, [this, updatePixelWeight](const std::int32_t& value) {
        updatePixelWeight();
    });

    connect(this, &GroupAction::readOnlyChanged, this, [this, updateReadOnly, updateEnabledSettings](const bool& readOnly) {
        updateReadOnly();
        updateEnabledSettings();
        });

    updateKnnAlgorithm();
    updateDistanceMetric();
    updateNumIterations();
    updatePerplexity();
    updatePixelWeight();
    updateReadOnly();
    updateEnabledSettings();
}

