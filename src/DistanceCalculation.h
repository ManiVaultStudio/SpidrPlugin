#pragma once

#include <tuple>
#include <vector>

#include <QObject>
#include <QSize>

class Parameters;
enum class knn_library : size_t;
enum class knn_distance_metric : size_t;
enum class feature_type : unsigned int;
enum class loc_Neigh_Weighting : unsigned int;

/*!
 * 
 * 
 */
class DistanceCalculation : public QObject 
{
    Q_OBJECT
public:
    DistanceCalculation();
    ~DistanceCalculation(void) override;

    /*!
     * 
     * 
     */
    const std::tuple< std::vector<int>, std::vector<float>> output(); // tuple of indices and dists
    
    std::vector<int>* get_knn_indices();
    std::vector<float>* get_knn_distances_squared();

    void setKnnAlgorithm(knn_library knn);
    void setDistanceMetric(knn_distance_metric metric);

    /*!
     * 
     * 
     * \param histogramFeatures
     * \param params
     */
    void setup(std::vector<unsigned int>& pointIds, std::vector<float>& attribute_data, std::vector<float>* dataFeatures, Parameters& params);

    /*!
     * 
     * 
     */
    void compute();

private:

    void computekNN();

signals:
 // TODO: add slots that change _knn_lib and _knn_metric when widgets emit signal

private:
    // Options
    feature_type _featureType;                      /*!<> */
    knn_library _knn_lib;                           /*!<> */
    knn_distance_metric _knn_metric;                /*!<> */
    unsigned int _nn;                               /*!<> */
    size_t _neighborhoodSize;                       /*!< might be used for some distance metrics */
    loc_Neigh_Weighting _neighborhoodWeighting;     /*!< used when calculating distance directly from high-dim points (_featureType is no feature/PCOL) */

    // Data
    // Input
    size_t _numDims;                                /*!<> */
    size_t _numPoints;                              /*!<> */
    size_t _numHistBins;                            /*!<> */ // don't set this from the widget input. Instead you the value set in the feature extraction
    std::vector<float>* _dataFeatures;        /*!<> */
    std::vector<unsigned int>* _pointIds;     /*!<> */
    std::vector<float>* _attribute_data;      /*!<> */

    // Output
    std::vector<int> _knn_indices;                      /*!<> */
    std::vector<float> _knn_distances_squared;          /*!<> */
};