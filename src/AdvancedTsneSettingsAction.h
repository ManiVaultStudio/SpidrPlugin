#pragma once

#include "actions/Actions.h"

using namespace hdps::gui;

class QMenu;
class SpidrSettingsAction;

/**
 * Advanced TSNE setting action class
 *
 * Action class for advanced TSNE settings
 *
 * @author Thomas Kroes
 */
class AdvancedTsneSettingsAction : public GroupAction
{
public:

    /**
     * Constructor
     * @param tsneSettingsAction Reference to TSNE settings action
     */
    AdvancedTsneSettingsAction(SpidrSettingsAction& spidrSettingsAction);

public: // Action getters
    
    SpidrSettingsAction& getSpidrSettingsAction() { return _spidrSettingsAction; };
    IntegralAction& getExaggerationAction() { return _exaggerationAction; };
    IntegralAction& getExponentialDecayAction() { return _exponentialDecayAction; };
    IntegralAction& getNumTreesAction() { return _numTreesAction; };
    IntegralAction& getNumChecksAction() { return _numChecksAction; };

protected:
    SpidrSettingsAction&     _spidrSettingsAction;        /** Pointer to parent tSNE settings action */
    IntegralAction          _exaggerationAction;        /** Exaggeration action */
    IntegralAction          _exponentialDecayAction;    /** Exponential decay action */
    IntegralAction          _numTreesAction;            /** Exponential decay action */
    IntegralAction          _numChecksAction;           /** Exponential decay action */

};
