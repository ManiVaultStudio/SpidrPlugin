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

protected:
    DatasetPickerAction     _datasetPickerAction;    /** Dataset picker action */

    friend class Widget;
};