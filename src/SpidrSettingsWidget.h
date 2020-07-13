#pragma once

#include <widgets/SettingsWidget.h>

#include "PointData.h"

#include "DimensionSelectionWidget.h"

// Qt header files:
#include <QCheckBox>
#include <QComboBox>
#include <QGridLayout>
#include <QLineEdit>
#include <QLabel>
#include <QObject>
#include <QPushButton>
#include <QString>

#include <memory> // For unique_ptr
#include <vector>

using namespace hdps::gui;

class SpidrPlugin;

/**
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

    std::vector<bool> getEnabledDimensions();
    bool hasValidSettings();

    QString currentData();
    void dataChanged(const Points& points);
private:
    void checkInputStyle(QLineEdit& input);

signals:
    void dataSetPicked(QString);
    void knnAlgorithmPicked(int);
    void distanceMetricPicked(int);
    void kernelWeightPicked(int);
    void histBinSizeHeurPicked(int);

public slots:
    void computationStopped();

private slots:
    void onStartToggled(bool pressed);
    void kernelSizeChanged(const QString &value);
    void histBinSizeChanged(const QString &value);
    void onHistBinSizeChanged(const QString &value);
    void onHistBinSizeHeurPicked(int value);
    void numIterationsChanged(const QString &value);
    void perplexityChanged(const QString &value);
    void exaggerationChanged(const QString &value);
    void expDecayChanged(const QString &value);
    void numTreesChanged(const QString &value);
    void numChecksChanged(const QString &value);
    void thetaChanged(const QString& value);

public:
    hdps::DimensionSelectionWidget _dimensionSelectionWidget;

    QComboBox dataOptions;
    QComboBox knnOptions;
    QComboBox distanceMetric;

    QComboBox kernelWeight;
    QLineEdit kernelSize;
    QComboBox histBinSizeHeur;
    QLineEdit histBinSize;

    QLineEdit numIterations;
    QLineEdit perplexity;
    QLineEdit exaggeration;
    QLineEdit expDecay;
    QLineEdit numTrees;
    QLineEdit numChecks;
    QLineEdit theta;

    QPushButton startButton;

private:
  SpidrPlugin& _analysisPlugin;
};
