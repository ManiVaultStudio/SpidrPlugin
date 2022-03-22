#include "DimensionSelectionAction.h"

DimensionSelectionAction::DimensionSelectionAction(QObject* parent) :
    GroupAction(parent),
    _pickerAction(this)
{
    setText("Dimensions");
}