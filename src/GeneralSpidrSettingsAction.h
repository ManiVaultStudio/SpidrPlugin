#pragma once

#include "actions/Actions.h"

#include "TsneComputationAction.h"

using namespace hdps::gui;

class QMenu;
class SpidrSettingsAction;

/**
 * General Spidr setting action class
 *
 * Based on actions class for general TSNE settings
 *
 * @author Thomas Kroes (GeneralTsneSettingsAction)
 */
class GeneralSpidrSettingsAction : public GroupAction
{
public:

    /**
     * Constructor
     * @param tsneSettingsAction Reference to TSNE settings action
     */
    GeneralSpidrSettingsAction(SpidrSettingsAction& spidrSettingsAction);

public: // Action getters

    SpidrSettingsAction& getSpidrSettingsAction() { return _spidrSettingsAction; };
    OptionAction& getKnnTypeAction() { return _knnTypeAction; };
    OptionAction& getDistanceMetricAction() { return _distanceMetricAction; };
    IntegralAction& getNumIterationsAction() { return _numIterationsAction; };
    IntegralAction& getPerplexityAction() { return _perplexityAction; };
    SpidrComputationAction& getComputationAction() { return _computationAction; }
    TriggerAction& getResetAction() { return _resetAction; };

protected:
    SpidrSettingsAction&     _spidrSettingsAction;            /** Reference to parent tSNE settings action */
    OptionAction            _knnTypeAction;                 /** KNN action */
    OptionAction            _distanceMetricAction;          /** Distance metric action */
    OptionAction            _kernelWeight;
    OptionAction            _histBinSizeHeur;
    IntegralAction          _kernelSize;
    IntegralAction          _numIterationsAction;           /** Number of iterations action */
    IntegralAction          _perplexityAction;              /** Perplexity action */
    SpidrComputationAction   _computationAction;             /** Computation action */
    TriggerAction           _resetAction;                   /** Reset all input to defaults */

    friend class Widget;
};
