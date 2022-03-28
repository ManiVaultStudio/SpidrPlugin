#include "BackgroundSelectionAction.h"

#include <PointData.h>

BackgroundSelectionAction::BackgroundSelectionAction(QObject* parent) :
    GroupAction(parent),
    _datasetPickerAction(this, "Background dataset")
{
    setText("Background dataset");

    // Get unique identifier and gui names from all point data sets in the core
    auto dataSets = hdps::Application::core()->requestAllDataSets(QVector<hdps::DataType> {PointType});

    // Assign found dataset(s)
    _datasetPickerAction.setDatasets(dataSets);
}