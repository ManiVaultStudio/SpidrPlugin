#include "BackgroundSelectionAction.h"

#include <PointData/PointData.h>


BackgroundSelectionAction::BackgroundSelectionAction(QObject* parent) :
    GroupAction(parent, "BackgroundSelectionAction"),
    _datasetPickerAction(this, "Background dataset"),
    _reloadDataSets(this, "Reload Datasets"),
    _enableDisable(this, "Enable/Disable"),
    _idsInDataAction(this, "IDs in data", false)
{
    setText("Background dataset");

    addAction(&_datasetPickerAction);
    addAction(&_reloadDataSets);
    addAction(&_enableDisable);
    addAction(&_idsInDataAction);

    auto setDatasets = [this]() ->void {
        // Get unique identifier and gui names from all point data sets in the core
        auto dataSets = mv::data().getAllDatasets( {PointType});

        // Assign found dataset(s)
        _datasetPickerAction.setDatasets(dataSets);
    };

    setDatasets();

    connect(&_reloadDataSets, &TriggerAction::triggered, this, [this, setDatasets]() {
        setDatasets();
        });

    connect(&_enableDisable, &TriggerAction::triggered, this, [this]() {
        _datasetPickerAction.setEnabled(!_datasetPickerAction.isEnabled());
        });

}