#pragma once

#include <tuple>
#include <vector>

#include <QObject>
#include <QSize>

class Parameters;
enum class knn_library : unsigned int;
enum class knn_distance_metric : unsigned int;

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
    void setup(std::vector<float>* histogramFeatures, Parameters& params);

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
    knn_library _knn_lib;                           /*!<> */
    knn_distance_metric _knn_metric;                /*!<> */
    unsigned int _nn;                               /*!<> */

    // Data
    // Input
    unsigned int _numDims;                          /*!<> */
    unsigned int _numPoints;                        /*!<> */
    unsigned int _numHistBins;                      // don't set this from the widget input. Instead you the value set in the feature extraction
    const std::vector<float>* _histogramFeatures;   /*!<> */

    // Output
    std::vector<int> _indices;                      /*!<> */
    std::vector<float> _distances_squared;          /*!<> */
};