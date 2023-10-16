#include "DimensionSelectionAction.h"

DimensionSelectionAction::DimensionSelectionAction(QObject* parent) :
    GroupAction(parent, "DimensionSelection", true),
    _pickerAction(this, "DimensionPicker")
{
    setText("Dimensions");

    addAction(&_pickerAction);
}