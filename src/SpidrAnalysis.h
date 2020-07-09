#pragma once

#include "TsneComputation.h"
#include "DistanceCalculation.h"
#include "FeatureExtraction.h"

#include <QThread>

class SpidrAnalysis : public QThread
{
    Q_OBJECT
public:
    SpidrAnalysis();
    ~SpidrAnalysis() override;

private:
    void run() override;

    FeatureExtraction _featExtraction;
    DistanceCalculation _distCalc;
    TsneComputation _tsne;

};