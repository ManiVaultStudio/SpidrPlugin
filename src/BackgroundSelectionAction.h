#pragma once

#include <actions/GroupAction.h>
#include <actions/DatasetPickerAction.h>

#include <PointData/DimensionsPickerAction.h>

/**
 * Dataset picker action class
 *
 * Action class for data set selection, used as background that is to be excluded
 *
 * @author Alexander Vieth
 */
class BackgroundSelectionAction : public mv::gui::GroupAction
{
public:

    /**
     * Constructor
     * @param parent Pointer to parent object
     */
    BackgroundSelectionAction(QObject* parent);

public: // Action getters

    /** Get smart pointer to dataset (if any) */
    mv::Dataset<mv::DatasetImpl> getBackgroundDataset() {
        if (_datasetPickerAction.isEnabled())
            return _datasetPickerAction.getCurrentDataset();
        else
            return nullptr;
    }

    bool getIDsInData() { return _idsInDataAction.isChecked();  }

protected:
    mv::gui::DatasetPickerAction     _datasetPickerAction;    /** Dataset picker action */
    mv::gui::TriggerAction           _reloadDataSets;         /** Start computation action */
    mv::gui::TriggerAction           _enableDisable;          /** Start computation action */
    mv::gui::ToggleAction            _idsInDataAction;        /** Tick if data contains IDs for background (useful for loading backgorund IDs), otherwise assume the data is a subset and use the global data IDs */

};