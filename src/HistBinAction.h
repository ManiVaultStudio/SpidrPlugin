#pragma once

#include "actions/Actions.h"

using namespace hdps::gui;

class SpidrSettingsAction;
class QMenu;

/**
 * Hist bin size selecion action class
 *
 */
class HistBinAction : public WidgetAction
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
        Widget(QWidget* parent, HistBinAction* tsneComputationAction);
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
    HistBinAction(QObject* parent);

    /**
     * Get the context menu for the action
     * @param parent Parent widget
     * @return Context menu
     */
    QMenu* getContextMenu(QWidget* parent = nullptr) override;

//    bool isResettable() const override;

    void reset() override;

private slots:
    void onHistBinNumHeurChanged();

public: // Action getters

    OptionAction& getHistBinSizeHeur() { return _histBinNumHeur; }
    IntegralAction& getNumHistBinsAction() { return _numHistBins; }

protected:
    OptionAction            _histBinNumHeur;
    IntegralAction          _numHistBins;
};