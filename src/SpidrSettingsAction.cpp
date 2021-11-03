#include "SpidrSettingsAction.h"

using namespace hdps::gui;

SpidrSettingsAction::SpidrSettingsAction(QObject* parent) :
    GroupAction(parent),
    _spidrParameters(),
    _generalSpidrSettingsAction(*this),
    _advancedTsneSettingsAction(*this)
{
    setText("Spidr");

    const auto updateReadOnly = [this]() -> void {
        _generalSpidrSettingsAction.setReadOnly(isReadOnly());
        _advancedTsneSettingsAction.setReadOnly(isReadOnly());
    };

    connect(this, &GroupAction::readOnlyChanged, this, [this, updateReadOnly](const bool& readOnly) {
        updateReadOnly();
    });

    updateReadOnly();
}

QMenu* SpidrSettingsAction::getContextMenu(QWidget* parent /*= nullptr*/)
{
    auto menu = new QMenu(text(), parent);

    auto& computationAction = _generalSpidrSettingsAction.getComputationAction();

    menu->addAction(&computationAction.getStartComputationAction());
    //menu->addAction(&computationAction.getContinueComputationAction());
    menu->addAction(&computationAction.getStopComputationAction());

    return menu;
}