#pragma once

#include "actions/Actions.h"

#include "SpidrComputationAction.h"
#include "HistBinAction.h"

#include <QStandardItemModel> 
#include <QStandardItem> 
#include <QList> 

#include <memory> 

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
    IntegralAction& getNumberOfComputatedIterationsAction() { return _numberOfComputatedIterationsAction; };
    IntegralAction& getPerplexityAction() { return _perplexityAction; };
    DecimalAction& getPixelWeightAction() { return _pixelWeightAction; };
    HistBinAction& getHistBinAction() { return _histBinSizeAction; }
    SpidrComputationAction& getComputationAction() { return _computationAction; }

protected:
    SpidrSettingsAction&     _spidrSettingsAction;                  /** Reference to parent tSNE settings action */
    OptionAction            _knnTypeAction;                         /** KNN action */
    OptionAction            _distanceMetricAction;                  /** Distance metric action */
    OptionAction            _kernelWeight;                          /** Local weighting of kernel (neighborhood pixel) */
    HistBinAction           _histBinSizeAction;                     /** Heuristic to determine number of histogram bins */
    IntegralAction          _kernelSize;                            /** Number of neighbors in each direction, i.e. kernelSize = 1 => 3x3 neighborhood */
    IntegralAction          _numIterationsAction;                   /** Number of iterations action */
    IntegralAction          _numberOfComputatedIterationsAction;    /** Number of computed iterations action */
    IntegralAction          _perplexityAction;                      /** Perplexity action */
    DecimalAction           _pixelWeightAction;                     /** Pixel weight action */
    SpidrComputationAction   _computationAction;                    /** Computation action */

    friend class Widget;

private:
    std::shared_ptr<QStandardItemModel> _distanceItemModel;
    QList<std::shared_ptr<QStandardItem>> _distanceItemList;

};
