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
#include <QLabel>
#include <QObject>
#include <QPushButton>
#include <QString>

#include <memory> // For unique_ptr
#include <vector>

using namespace hdps::gui;

class SpidrPlugin;

// Heuristic for setting the histogram bin size
enum class histBinSizeHeuristic : unsigned int
{
    MANUAL = 0,    /*!< Manually  adjust histogram bin size */
    SQRT = 1,      /*!< ceil(sqrt(n)), n = neighborhood size */
    STURGES = 2,   /*!< ceil(log_2(n))+1, n = neighborhood size */
    RICE = 3,      /*!< ceil(2*pow(n, 1/3)), n = neighborhood size */
};

/**
 * Main settings widget
 */
/*!
 * 
 * 
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
    QString getEmbName();

private:
    void checkInputStyle(QLineEdit& input);

signals:
    void dataSetPicked(QString);

public slots:
    void computationStopped();
    void setEmbName(QString embName);

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
    hdps::DimensionSelectionWidget _dimensionSelectionWidget;

    QComboBox dataOptions;
    QComboBox knnOptions;
    QComboBox distanceMetric;

    QComboBox kernelWeight;
    QSpinBox  kernelSize;
    QComboBox histBinSizeHeur;
    QSpinBox  histBinSize;

    QLineEdit numIterations;
    QLineEdit perplexity;
    QLineEdit exaggeration;
    QLineEdit expDecay;
    QLineEdit numTrees;
    QLineEdit numChecks;
    QLineEdit theta;

    QLineEdit embNameLine;
    QPushButton startButton;

private:
  SpidrPlugin& _analysisPlugin;
};
