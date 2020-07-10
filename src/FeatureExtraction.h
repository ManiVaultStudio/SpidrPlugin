#pragma once

#include <vector>
#include <QObject>
#include <QSize>


class Parameters;
enum loc_Neigh_Weighting;

class FeatureExtraction : public QObject
{
    Q_OBJECT
public:
    FeatureExtraction();
    ~FeatureExtraction(void) override;

    std::vector<float>* output();

    void setNeighborhoodSize(unsigned int size);    // TODO sets _numNeighbors
    void setNumHistBins(unsigned int size);         // TODO 
    //void setNumHistBins(heuristic heu);           // TODO introduce enum for options
    void setNeighborhoodWeighting(int weighting);   // TODO introduce enum for options

    /**
    * Setup feature extraction by introducing the data
    * @param data retrieved data from points
    * @param pointIds points.indices (global IDs)
    * @param numDimensions enabled dimensios
    * @param imgSize global image dimensions
    */
    void setupData(const std::vector<unsigned int>& pointIds, const std::vector<float>& attribute_data, Parameters& params);

    void run();

private:

    /**
    * Calculates histgram features
    */
    void computeHistogramFeatures();

    /**
    *  Init, i.e. identify min and max per dimension for histogramming
    *  Sets _minMaxVals according to _inputData
    */
    void initExtraction();

    void extractFeatures();

    std::vector<int> neighborhoodIndices(unsigned int pointInd);

    void calculateHistogram(unsigned int pointInd, std::vector<float> neighborValues);

    void weightNeighborhood(loc_Neigh_Weighting weighting);

signals:
 // TODO: add slots that change _neighborhoodSize, _numHistBins and _neighborhoodWeights when widgets emit signal

private:
    // Options 

    // Square neighborhood centered around an item with _neighborhoodSize neighbors to the left, right, top and buttom
    unsigned int _neighborhoodSize;
    // Number of neighbors including center
    unsigned int _numLocNeighbors;
    // Weightings of neighborhood kernel
    std::vector<float> _neighborhoodWeights;
    // Number of bins in each histogram
    unsigned int _numHistBins;
    // Extrema for each dimension/channel, i.e. [min_Ch0, max_Ch0, min_Ch1, max_Ch1, ...]
    std::vector<float> _minMaxVals;

    // Data
    // Input
    QSize _imgSize;
    unsigned int _numDims;
    unsigned int _numPoints;
    std::vector<unsigned int> _pointIds;
    std::vector<float> _attribute_data;

    // Output
    // Histogram features for each item. 
    // In case of 1D histograms for each data point there are _inputData.getNumDimensions() histograms with _numHistBins values, i.e. size _numPoints * _numDims * _numHistBins
    std::vector<float> _histogramFeatures;

};