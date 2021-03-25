#pragma once

#include <widgets/SettingsWidget.h>

#include "PointData.h"

#include "DimensionSelectionWidget.h"

// Qt header files:
#include <QCheckBox>
#include <QComboBox>
#include <QGridLayout>
#include <QLineEdit>
#include <QSpinBox>
#include <QDoubleSpinBox>
#include <QSlider>
#include <QLabel>
#include <QObject>
#include <QPushButton>
#include <QString>

#include <memory> // For unique_ptr
#include <vector>

using namespace hdps::gui;

class SpidrPlugin;


/*!
 * Main settings widget
 */
class SpidrSettingsWidget : public SettingsWidget
{
    Q_OBJECT

public:
    // Explicitly delete its copy and move member functions.
    SpidrSettingsWidget(const SpidrSettingsWidget&) = delete;
    SpidrSettingsWidget(SpidrSettingsWidget&&) = delete;
    SpidrSettingsWidget& operator=(const SpidrSettingsWidget&) = delete;
    SpidrSettingsWidget& operator=(SpidrSettingsWidget&&) = delete;

    explicit SpidrSettingsWidget(SpidrPlugin&);

    QString getCurrentDataItem();
    void addDataItem(const QString name);
    void removeDataItem(const QString name);

    std::vector<bool> getEnabledDimensions();
    bool hasValidSettings();

    hdps::DimensionSelectionWidget& getDimensionSelectionWidget();

    void dataChanged(const Points& points);
    QString getEmbName();

private:
    void checkInputStyle(QLineEdit& input);

signals:
    void dataSetPicked(QString);

public slots:
    void computationStopped();
    void setEmbeddingName(QString embName);

private slots:
    void onStartToggled(bool pressed);
    void onKernelSizeChanged(const QString &kernelSizeField);
    void onHistBinSizeHeurPicked(int value);
    void onDistanceMetricPicked(int value);
    void numIterationsChanged(const QString &value);
    void perplexityChanged(const QString &value);
    void exaggerationChanged(const QString &value);
    void expDecayChanged(const QString &value);
    void numTreesChanged(const QString &value);
    void numChecksChanged(const QString &value);
    void thetaChanged(const QString& value);

public:
    QComboBox* _dataOptions;        //
    hdps::DimensionSelectionWidget _dimensionSelectionWidget;

    QComboBox dataOptions;
    QComboBox knnOptions;
    QComboBox distanceMetric;

    QComboBox kernelWeight;
    QSpinBox  kernelSize;
    QComboBox histBinSizeHeur;
    QSpinBox  histBinSize;

    QSlider   weightSpaAttrSlider;
    QDoubleSpinBox weightSpaAttrNum;

    QCheckBox publishFeaturesToCore;

    QLineEdit numIterations;
    QLineEdit perplexity;
    QLineEdit exaggeration;
    QLineEdit expDecay;
    QLineEdit numTrees;
    QLineEdit numChecks;
    QLineEdit theta;

    QComboBox* backgroundNameLine;
    QCheckBox backgroundFromData;   // if ticked take the data not the indices of the data as the background info (for loading background data sets)
    QCheckBox forceBackgroundFeatures;

    QLineEdit embeddingNameLine;
    QPushButton startButton;

private:
    SpidrPlugin& _analysisPlugin;
};
