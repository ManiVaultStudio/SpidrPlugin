#pragma once

#include "actions/Actions.h"

using namespace hdps::gui;

class SpidrSettingsAction;
class QMenu;

/**
 * Spidr computation action class
 *
 * Actions class for starting/continuing/stopping the TSNE computation
 *
 * @author Thomas Kroes (based on TsnepidrComputationAction)
 */
class SpidrComputationAction : public WidgetAction
{
protected:

    /** Widget class for TSNE computation action */
    class Widget : public WidgetActionWidget {
    public:

        /**
         * Constructor
         * @param parent Pointer to parent widget
         * @param tsneComputationAction Pointer to TSNE computation action
         */
        Widget(QWidget* parent, SpidrComputationAction* tsneComputationAction);
    };

    /**
     * Get widget representation of the TSNE computation action
     * @param parent Pointer to parent widget
     * @param widgetFlags Widget flags for the configuration of the widget (type)
     */
    QWidget* getWidget(QWidget* parent, const std::int32_t& widgetFlags) override {
        return new Widget(parent, this);
    };

public:

    /**
     * Constructor
     * @param parent Pointer to parent object
     */
    SpidrComputationAction(QObject* parent);

    /**
     * Get the context menu for the action
     * @param parent Parent widget
     * @return Context menu
     */
    QMenu* getContextMenu(QWidget* parent = nullptr) override;

public: // Action getters

    TriggerAction& getStartComputationAction() { return _startComputationAction; }
    //TriggerAction& getContinueComputationAction() { return _continueComputationAction; }
    TriggerAction& getStopComputationAction() { return _stopComputationAction; }
    ToggleAction& getRunningAction() { return _runningAction; }

protected:
    TriggerAction           _startComputationAction;        /** Start computation action */
    //TriggerAction           _continueComputationAction;     /** Continue computation action */
    TriggerAction           _stopComputationAction;         /** Stop computation action */
    ToggleAction            _runningAction;                 /** Running action */

};