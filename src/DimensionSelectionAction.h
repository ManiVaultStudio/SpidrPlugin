#pragma once

#include <actions/GroupAction.h>

#include <PointData/DimensionsPickerAction.h>

/**
 * Dimension selection action class
 *
 * Action class for point data dimension selection
 *
 * @author Thomas Kroes
 */
class DimensionSelectionAction : public mv::gui::GroupAction
{
public:

    /**
     * Constructor
     * @param parent Pointer to parent object
     */
    DimensionSelectionAction(QObject* parent);

public: // Action getters

    DimensionsPickerAction& getPickerAction() { return _pickerAction; };

protected:
    DimensionsPickerAction  _pickerAction;    /** Dimension picker action */

};