#pragma once

#include <actions/GroupAction.h>

#include <DimensionsPickerAction.h>

/**
 * Dimension selection action class
 *
 * Action class for point data dimension selection
 *
 * @author Thomas Kroes
 */
class DimensionSelectionAction : public GroupAction
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

    friend class Widget;
};