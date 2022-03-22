#include "SpidrComputationAction.h"
#include "SpidrSettingsAction.h"

#include <QHBoxLayout>
#include <QMenu>

using namespace hdps::gui;

SpidrComputationAction::SpidrComputationAction(QObject* parent) :
    WidgetAction(parent),
    _startComputationAction(this, "Start"),
    //_continueComputationAction(this, "Continue"),
    _stopComputationAction(this, "Stop"),
    _runningAction(this, "Running")
{
    setText("Computation");

    _startComputationAction.setToolTip("Start the tSNE computation");
    //_continueComputationAction.setToolTip("Continue with the tSNE computation");
    _stopComputationAction.setToolTip("Stop the current tSNE computation");
}

QMenu* SpidrComputationAction::getContextMenu(QWidget* parent /*= nullptr*/)
{
    auto menu = new QMenu(text(), parent);

    menu->addAction(&_startComputationAction);
    //menu->addAction(&_continueComputationAction);
    menu->addAction(&_stopComputationAction);

    return menu;
}

SpidrComputationAction::Widget::Widget(QWidget* parent, SpidrComputationAction* tsneComputationAction) :
    WidgetActionWidget(parent, tsneComputationAction)
{
    auto layout = new QHBoxLayout();

    layout->setMargin(0);

    layout->addWidget(tsneComputationAction->getStartComputationAction().createWidget(this));
    //layout->addWidget(tsneComputationAction->getContinueComputationAction().createWidget(this));
    layout->addWidget(tsneComputationAction->getStopComputationAction().createWidget(this));

    setLayout(layout);
}
