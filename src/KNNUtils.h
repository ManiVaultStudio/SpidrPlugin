#pragma once
#include "hnswlib/hnswlib.h"

#include <cmath>     // std::sqrt
#include <vector>

enum knn_library
{
    KNN_HSNW = 0,
    //  KNN_FLANN = 1,
    //  KNN_KGRAPH = 2
};


enum knn_distance_metric
{
    KNN_METRIC_HEL = -1,
    KNN_METRIC_QF = 0,
    //	KNN_METRIC_EMD = 1
};

class Parameters {
public:
    Parameters() :
        _perplexity(30),
        _perplexity_multiplier(3),
        _num_trees(4),
        _num_checks(1024),
        _aknn_algorithm(KNN_HSNW),
        _aknn_metric(KNN_METRIC_QF)
    {}

public:
    float       _perplexity;            //! Perplexity value in evert distribution.
    int         _perplexity_multiplier; //! Multiplied by the perplexity gives the number of nearest neighbors used
    int         _num_trees;             //! Number of trees used int the AKNN
    int         _num_checks;            //! Number of checks used int the AKNN
    knn_library _aknn_algorithm;
    knn_distance_metric _aknn_metric;
};


namespace hnswlib {

    // ---------------
    // Quadratif form
    // ---------------

    static float
        QFSqr(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
        float *pVect1i = (float *)pVect1v;
        float *pVect1j = (float *)pVect1v;
        float *pVect2i = (float *)pVect2v;
        float *pVect2j = (float *)pVect2v;

        size_t qty = *((size_t *)qty_ptr);

        float res = 0;
        // add the histogram distance for each dimension
        for (size_t d = 0; d < qty; d++) {
            for (size_t i = 0; i < QFSpace::bin_; i++) {
                for (size_t j = 0; j < QFSpace::bin_; j++) {
                    float t = ::std::sqrt(*pVect1) - ::std::sqrt(*pVect2);
                    pVect2++;
                    res += t * t;
                    }
                pVect1++;
            }
        }
        return (res);
    }

    class QFSpace : public SpaceInterface<float> {

        DISTFUNC<float> fstdistfunc_;
        size_t data_size_;
        size_t dim_;

    public:
        static size_t bin_;
        static ::std::vector<size_t> A;

    public:
        QFSpace(size_t dim, size_t bin) {
            fstdistfunc_ = QFSqr;
            dim_ = dim;
            bin_ = bin;
            data_size_ = dim * bin * sizeof(float);
            A.resize(bin*bin);
            for (size_t i = 0; i < bin; i++) {
                for (size_t j = 0; j < bin; j++) {
                    size_t bin_diff = (i - j);
                    A[i*bin + j] = ::std::sqrt(1 + (bin_diff*bin_diff));
                }
            }
        }

        size_t get_data_size() {
            return data_size_;
        }

        DISTFUNC<float> get_dist_func() {
            return fstdistfunc_;
        }

        void *get_dist_func_param() {
            return &dim_;
        }

        ~QFSpace() {}
    };

    size_t QFSpace::bin_ = 0;

    // ---------------
    //    Hellinger
    // ---------------

    static float
        HelSqr(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
        float *pVect1 = (float *)pVect1v;
        float *pVect2 = (float *)pVect2v;
        size_t qty = *((size_t *)qty_ptr);

        float res = 0;
        // add the histogram distance for each dimension
        for (size_t d = 0; d < qty; d++) {
            for (size_t i = 0; i < HellingerSpace::bin_; i++) {
                float t = ::std::sqrt(*pVect1) - ::std::sqrt(*pVect2);
                pVect1++;
                pVect2++;
                res += t * t;
            }
        }
        return (res);
    }

    class HellingerSpace : public SpaceInterface<float> {

        DISTFUNC<float> fstdistfunc_;
        size_t data_size_;
        size_t dim_;

    public:
        static size_t bin_;

    public:
        HellingerSpace(size_t dim, size_t bin) {
            fstdistfunc_ = HelSqr;
            dim_ = dim;
            bin_ = bin;
            data_size_ = dim * bin * sizeof(float);
        }

        size_t get_data_size() {
            return data_size_;
        }

        DISTFUNC<float> get_dist_func() {
            return fstdistfunc_;
        }

        void *get_dist_func_param() {
            return &dim_;
        }

        ~HellingerSpace() {}
    };

    size_t HellingerSpace::bin_ = 0;

}
