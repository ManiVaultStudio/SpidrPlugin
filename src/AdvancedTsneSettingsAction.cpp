#include "AdvancedTsneSettingsAction.h"
#include "SpidrSettingsAction.h"

#include <QTableView>

using namespace hdps::gui;

AdvancedTsneSettingsAction::AdvancedTsneSettingsAction(SpidrSettingsAction& tsneSettingsAction) :
    GroupAction(&tsneSettingsAction),
    _spidrSettingsAction(tsneSettingsAction),
    _exaggerationAction(this, "Exaggeration"),
    _exponentialDecayAction(this, "Exponential decay"),
    _numTreesAction(this, "Number of trees"),
    _numChecksAction(this, "Number of checks"),
    _resetAction(this, "Reset all")
{
    setText("Advanced TSNE");

    const auto& tsneParameters = _spidrSettingsAction.getSpidrParameters();

    _exaggerationAction.setDefaultWidgetFlags(IntegralAction::SpinBox);
    _exponentialDecayAction.setDefaultWidgetFlags(IntegralAction::SpinBox);
    _numTreesAction.setDefaultWidgetFlags(IntegralAction::SpinBox);
    _numChecksAction.setDefaultWidgetFlags(IntegralAction::SpinBox);

    _exaggerationAction.initialize(1, 10000, 250, 250);
    _exponentialDecayAction.initialize(1, 10000, 70, 70);
    _numTreesAction.initialize(1, 10000, 4, 4);
    _numChecksAction.initialize(1, 10000, 1024, 1024);

    const auto updateExaggeration = [this]() -> void {
        _spidrSettingsAction.getSpidrParameters()._exaggeration =_exaggerationAction.getValue();
    };

    const auto updateExponentialDecay = [this]() -> void {
        _spidrSettingsAction.getSpidrParameters()._expDecay = _exponentialDecayAction.getValue();
    };

    const auto updateNumTrees = [this]() -> void {
        // TODO: implement
        //_spidrSettingsAction.getSpidrParameters().setNumTrees(_numTreesAction.getValue());
    };

    const auto updateNumChecks = [this]() -> void {
       // TODO: implement this
        //_spidrSettingsAction.getSpidrParameters().setNumChecks(_numChecksAction.getValue());
    };

    const auto isResettable = [this]() -> bool {
        if (_exaggerationAction.isResettable())
            return true;

        if (_exponentialDecayAction.isResettable())
            return true;

        if (_numTreesAction.isResettable())
            return true;

        if (_numChecksAction.isResettable())
            return true;

        return false;
    };

    const auto updateReset = [this, isResettable]() -> void {
        _resetAction.setEnabled(isResettable());
    };

    const auto updateReadOnly = [this, isResettable]() -> void {
        const auto enable = !isReadOnly();

        _exaggerationAction.setEnabled(enable);
        _exponentialDecayAction.setEnabled(enable);
        _numTreesAction.setEnabled(enable);
        _numChecksAction.setEnabled(enable);
        _resetAction.setEnabled(enable);
    };

    connect(&_exaggerationAction, &IntegralAction::valueChanged, this, [this, updateExaggeration, updateReset](const std::int32_t& value) {
        updateExaggeration();
        updateReset();
    });

    connect(&_exponentialDecayAction, &IntegralAction::valueChanged, this, [this, updateExponentialDecay, updateReset](const std::int32_t& value) {
        updateExponentialDecay();
        updateReset();
    });

    connect(&_numTreesAction, &IntegralAction::valueChanged, this, [this, updateNumTrees, updateReset](const std::int32_t& value) {
        updateNumTrees();
        updateReset();
    });

    connect(&_numChecksAction, &IntegralAction::valueChanged, this, [this, updateNumChecks, updateReset](const std::int32_t& value) {
        updateNumChecks();
        updateReset();
    });

    connect(&_resetAction, &TriggerAction::triggered, this, [this](const std::int32_t& value) {
        _exaggerationAction.reset();
        _exponentialDecayAction.reset();
        _numTreesAction.reset();
        _numChecksAction.reset();
    });

    connect(this, &GroupAction::readOnlyChanged, this, [this, updateReadOnly](const bool& readOnly) {
        updateReadOnly();
    });

    updateExaggeration();
    updateExponentialDecay();
    updateNumTrees();
    updateNumChecks();
    updateReset();
    updateReadOnly();
}
