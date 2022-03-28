#pragma once

#include <actions/GroupAction.h>

#include <DimensionsPickerAction.h>

/**
 * Dataset picker action class
 *
 * Action class for data set selection, used as background that is to be excluded
 *
 * @author Alexander Vieth
 */
class BackgroundSelectionAction : public GroupAction
{
public:

    /**
     * Constructor
     * @param parent Pointer to parent object
     */
    BackgroundSelectionAction(QObject* parent);

public: // Action getters

    /** Get smart pointer to dataset (if any) */
    hdps::Dataset<hdps::DatasetImpl> getBackgroundDataset() {
        return _datasetPickerAction.getCurrentDataset();
    }

    bool getIDsInData() { return _idsInDataAction.isChecked();  }

protected:
    DatasetPickerAction     _datasetPickerAction;    /** Dataset picker action */
    ToggleAction            _idsInDataAction;        /** Tick if data contains IDs for background (useful for loading backgorund IDs), otherwise assume the data is a subset and use the global data IDs */

    friend class Widget;
};