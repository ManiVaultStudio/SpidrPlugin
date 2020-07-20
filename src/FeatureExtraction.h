#pragma once

#include <vector>
#include <QObject>
#include <QSize>

class Parameters;
enum class loc_Neigh_Weighting : unsigned int;

/*!
 * 
 * 
 */
class FeatureExtraction : public QObject
{
    Q_OBJECT
public:
    FeatureExtraction();
    ~FeatureExtraction(void) override;

    /*!
     * 
     * 
     * \return 
     */
    std::vector<float>* output();

    void setNumLocNeighbors(size_t size);
    void setNeighborhoodWeighting(loc_Neigh_Weighting weighting);
    void setNumHistBins(size_t size);
    //void setNumHistBins(heuristic heu);

    loc_Neigh_Weighting getNeighborhoodWeighting();

    /**
    * Setup feature extraction by introducing the data
    * @param data retrieved data from points
    * @param pointIds points.indices (global IDs)
    * @param numDimensions enabled dimensios
    * @param imgSize global image dimensions
    */
    /*!
     * 
     * 
     * \param pointIds
     * \param attribute_data
     * \param params
     */
    void setup(const std::vector<unsigned int>& pointIds, const std::vector<float>& attribute_data, const Parameters& params);

    /*!
     * 
     * 
     */
    void compute();

private:

    /**
    * Calculates histgram features
    */
    void computeHistogramFeatures();

    /*!
     *  Init, i.e. identify min and max per dimension for histogramming
     *  Sets _minMaxVals according to _inputData
    */
    void initExtraction();

    /*!
     * 
     * 
     */
    void extractFeatures();

    /*!
     * 
     * 
     * \param pointInd
     * \return 
     */
    std::vector<int> neighborhoodIndices(size_t pointInd);

    /*!
     * 
     * 
     * \param pointInd
     * \param neighborValues
     */
    void calculateHistogram(size_t pointInd, std::vector<float> neighborValues);

private:

    /*!
     * 
     * 
     * \param weighting
     */
    void weightNeighborhood(loc_Neigh_Weighting weighting);


    // Options 

    // Number of neighbors including center
    size_t       _locNeighbors;                     /*!<> */
    // Width of the kernel (2* _locNeighbors +1)
    size_t       _kernelWidth;                      /*!<> */
    // Square neighborhood centered around an item with _neighborhoodSize neighbors to the left, right, top and buttom
    size_t       _neighborhoodSize;                 /*!<> */
    // Weighting type of neighborhood kernel
    loc_Neigh_Weighting _neighborhoodWeighting;     /*!<> */
    // Weightings of neighborhood kernel
    std::vector<float> _neighborhoodWeights;        /*!<> */
    // Number of bins in each histogram
    size_t       _numHistBins;                      /*!<> */

    // Data
    // Input
    QSize _imgSize;                                 /*!<> */
    size_t       _numDims;                          /*!<> */
    size_t       _numPoints;                        /*!<> */
    std::vector<unsigned int> _pointIds;            /*!<> */
    std::vector<float> _attribute_data;             /*!<> */
    // Extrema for each dimension/channel, i.e. [min_Ch0, max_Ch0, min_Ch1, max_Ch1, ...]
    std::vector<float> _minMaxVals;                 /*!<> */

    // Output
    // Histogram features for each item. 
    // In case of 1D histograms for each data point there are _inputData.getNumDimensions() histograms with _numHistBins values, i.e. size _numPoints * _numDims * _numHistBins
    std::vector<float> _histogramFeatures;          /*!<> */

};