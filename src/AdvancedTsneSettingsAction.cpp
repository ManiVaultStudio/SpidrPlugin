#include "AdvancedTsneSettingsAction.h"
#include "SpidrSettingsAction.h"

#include <QTableView>

using namespace mv::gui;

AdvancedTsneSettingsAction::AdvancedTsneSettingsAction(SpidrSettingsAction& tsneSettingsAction) :
    GroupAction(&tsneSettingsAction, "AdvancedTsneSettingsAction"),
    _spidrSettingsAction(tsneSettingsAction),
    _exaggerationAction(this, "Exaggeration"),
    _exponentialDecayAction(this, "Exponential decay"),
    _numTreesAction(this, "Number of trees"),
    _numChecksAction(this, "Number of checks")
{
    setText("Advanced TSNE");
    setObjectName("Advanced TSNE");

    addAction(&_exaggerationAction);
    addAction(&_exponentialDecayAction);
    addAction(&_numTreesAction);
    addAction(&_numChecksAction);

    auto& tsneParameters = _spidrSettingsAction.getSpidrParameters();

    _exaggerationAction.setDefaultWidgetFlags(IntegralAction::SpinBox);
    _exponentialDecayAction.setDefaultWidgetFlags(IntegralAction::SpinBox);
    _numTreesAction.setDefaultWidgetFlags(IntegralAction::SpinBox);
    _numChecksAction.setDefaultWidgetFlags(IntegralAction::SpinBox);

    _exaggerationAction.initialize(1, 10000, 250);
    _exponentialDecayAction.initialize(1, 10000, 70);
    _numTreesAction.initialize(1, 10000, 4);
    _numChecksAction.initialize(1, 10000, 1024);

    const auto updateExaggeration = [this, &tsneParameters]() -> void {
        tsneParameters._exaggeration =_exaggerationAction.getValue();
    };

    const auto updateExponentialDecay = [this, &tsneParameters]() -> void {
        tsneParameters._expDecay = _exponentialDecayAction.getValue();
    };

    const auto updateNumTrees = [this]() -> void {
        // TODO: implement
        //_spidrSettingsAction.getSpidrParameters().setNumTrees(_numTreesAction.getValue());
    };

    const auto updateNumChecks = [this]() -> void {
       // TODO: implement this
        //_spidrSettingsAction.getSpidrParameters().setNumChecks(_numChecksAction.getValue());
    };

    const auto updateReadOnly = [this]() -> void {
        const auto enable = !isReadOnly();

        _exaggerationAction.setEnabled(enable);
        _exponentialDecayAction.setEnabled(enable);
        _numTreesAction.setEnabled(enable);
        _numChecksAction.setEnabled(enable);
    };

    connect(&_exaggerationAction, &IntegralAction::valueChanged, this, [this, updateExaggeration](const std::int32_t& value) {
        updateExaggeration();
    });

    connect(&_exponentialDecayAction, &IntegralAction::valueChanged, this, [this, updateExponentialDecay](const std::int32_t& value) {
        updateExponentialDecay();
    });

    connect(&_numTreesAction, &IntegralAction::valueChanged, this, [this, updateNumTrees](const std::int32_t& value) {
        updateNumTrees();
    });

    connect(&_numChecksAction, &IntegralAction::valueChanged, this, [this, updateNumChecks](const std::int32_t& value) {
        updateNumChecks();
    });

    connect(this, &GroupAction::readOnlyChanged, this, [this, updateReadOnly](const bool& readOnly) {
        updateReadOnly();
    });

    updateExaggeration();
    updateExponentialDecay();
    updateNumTrees();
    updateNumChecks();
    updateReadOnly();
}
