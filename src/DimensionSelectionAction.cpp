#include "DimensionSelectionAction.h"

DimensionSelectionAction::DimensionSelectionAction(QObject* parent) :
    GroupAction(parent, true),
    _pickerAction(this)
{
    setText("Dimensions");
}