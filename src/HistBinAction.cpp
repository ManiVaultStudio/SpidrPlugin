#include "HistBinAction.h"
#include "SpidrSettingsAction.h"

#include <QHBoxLayout>
#include <QMenu>

using namespace hdps::gui;

HistBinAction::HistBinAction(QObject* parent) :
    WidgetAction(parent),
    _histBinNumHeur(this, "Histo. bin heuristic: Sqrt: ceil(sqrt(n)) \nSturges: ceil(log_2(n))+1 \nRice: ceil(2*pow(n, 1/3))"),
    _numHistBins(this, "Num bins")
{
    setText("Histogram bins");

    _histBinNumHeur.setToolTip("Select heuristic");
    _numHistBins.setToolTip("Number of bins in histogram");

    _histBinNumHeur.setDefaultWidgetFlags(OptionAction::ComboBox);
    _numHistBins.setDefaultWidgetFlags(IntegralAction::SpinBox);

    _histBinNumHeur.initialize(QStringList({ "Manual", "Rice", "Sturges", "Sqrt" }), "Rice", "Rice");
    _numHistBins.initialize(1, 10000, 5, 5);
    _numHistBins.setDisabled(true); // enable only for manual

    connect(&_histBinNumHeur, &OptionAction::currentIndexChanged, this, &HistBinAction::onHistBinNumHeurChanged);

}

QMenu* HistBinAction::getContextMenu(QWidget* parent /*= nullptr*/)
{
    auto menu = new QMenu(text(), parent);

    menu->addAction(&_histBinNumHeur);
    menu->addAction(&_numHistBins);

    return menu;
}

HistBinAction::Widget::Widget(QWidget* parent, HistBinAction* histBinAction) :
    WidgetActionWidget(parent, histBinAction)
{
    auto layout = new QHBoxLayout();

    layout->setMargin(0);

    layout->addWidget(histBinAction->getHistBinSizeHeur().createWidget(this));
    layout->addWidget(histBinAction->getNumHistBinsAction().createWidget(this));

    setLayout(layout);
}

void HistBinAction::onHistBinNumHeurChanged()
{

    if (_histBinNumHeur.getCurrentIndex() == 0) // Manual
    {
        _numHistBins.setEnabled(true);
    }
    else // Otherwise use heuristic, set value from GeneralSpidrSettingsAction
    {
        _numHistBins.setDisabled(true);
    }

}

//bool HistBinAction::isResettable() const {
//    return _histBinNumHeur.getCurrentIndex() != _histBinNumHeur.getDefaultIndex();
//}

void HistBinAction::reset() {
    _histBinNumHeur.reset();
}
