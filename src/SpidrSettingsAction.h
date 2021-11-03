#pragma once

#include "GeneralSpidrSettingsAction.h"
#include "AdvancedTsneSettingsAction.h"

#include "SpidrAnalysisParameters.h"

using namespace hdps::gui;

class QMenu;

class SpidrComputationAction;

/**
 * Spidr settings class
 *
 * Settings actions class for general/advanced HSNE/TSNE settings
 *
 * @author Thomas Kroes (based on TSNE settings class)
 */
class SpidrSettingsAction : public GroupAction
{
public:

    /**
     * Constructor
     * @param parent Pointer to parent object
     */
    SpidrSettingsAction(QObject* parent);

    /**
     * Get the context menu for the action
     * @param parent Parent widget
     * @return Context menu
     */
    QMenu* getContextMenu(QWidget* parent = nullptr) override;

public: // Action getters

    SpidrParameters& getSpidrParameters() { return _spidrParameters; }
    GeneralSpidrSettingsAction& getGeneralSpidrSettingsAction() { return _generalSpidrSettingsAction; }
    AdvancedTsneSettingsAction& getAdvancedTsneSettingsAction() { return _advancedTsneSettingsAction; }
    SpidrComputationAction& getComputationAction() { return _generalSpidrSettingsAction.getComputationAction(); }

protected:
    SpidrParameters                  _spidrParameters;                /** TSNE parameters */
    GeneralSpidrSettingsAction       _generalSpidrSettingsAction;     /** General tSNE settings action */
    AdvancedTsneSettingsAction       _advancedTsneSettingsAction;    /** Advanced tSNE settings action */

    friend class Widget;
};