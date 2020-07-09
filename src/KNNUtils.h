#pragma once
#include "hnswlib/hnswlib.h"

#include <cmath>     // std::sqrt, exp
#include <vector>

enum knn_library
{
    KNN_HNSW = 0
};


enum knn_distance_metric
{
    KNN_METRIC_QF = 0,
    //	KNN_METRIC_EMD = 1
    KNN_METRIC_HEL = 2,
};

enum ground_dist
{
    SIM_EUC = 0,
    SIM_EXP = 1
};

namespace hnswlib {


    // ---------------
    // Quadratic form
    // ---------------

    // data struct for distance calculation in QFSpace
    struct space_params_QF {
        size_t dim;
        size_t bin;
        ::std::vector<size_t> A;
    };

    static float
        QFSqr(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
        float* pVect1 = (float*)pVect1v;
        float* pVect2 = (float*)pVect2v;

        space_params_QF* sparam = (space_params_QF*)qty_ptr;

        float res = 0;
        // add the histogram distance for each dimension
        for (size_t d = 0; d < sparam->dim; d++) {
            for (size_t i = 0; i < sparam->bin; i++) {
                float t1 = *(pVect1 + i) - *(pVect2 + i);
                for (size_t j = 0; j < sparam->bin; j++) {
                    float t2 = *(pVect1 + j) - *(pVect2 + j);
                    res += *(sparam->A.data() + i * sparam->bin + j) * t1 * t2;
                }
            }
            // point to next dimension
            pVect1 += sparam->bin;
            pVect2 += sparam->bin;
        }

        return res;
    }

    // 1-D Histograms
    class QFSpace : public SpaceInterface<float> {

        DISTFUNC<float> fstdistfunc_;
        size_t data_size_;
        space_params_QF params_;

    public:
        // ground_weight might be set to (0.5 * sd of all data * ground_dist_max^2) as im doi:10.1006/cviu.2001.0934
        QFSpace(size_t dim, size_t bin, ground_dist ground_type = ground_dist::SIM_EUC, float ground_weight = 1) {
            fstdistfunc_ = QFSqr;
            data_size_ = dim * bin * sizeof(float);

            ::std::vector<size_t> A;
            A.resize(bin*bin);
            size_t ground_dist_max = bin - 1;

            int bin_diff = 0;

            size_t ground_dist_max_2 = ground_dist_max * ground_dist_max;
            size_t bin_diff_2 = 0;

            if (ground_type == ground_dist::SIM_EUC) {
                for (int i = 0; i < bin; i++) {
                    for (int j = 0; j < bin; j++) {
                        bin_diff = (i - j);
                        bin_diff_2 = bin_diff * bin_diff;
                        A[i * bin + j] = 1 - std::sqrt(float(bin_diff_2) / float(ground_dist_max_2));
                    }
                }
            }
            else if (ground_type == ground_dist::SIM_EXP) {
                for (size_t i = 0; i < bin; i++) {
                    for (size_t j = 0; j < bin; j++) {
                        bin_diff = (i - j);
                        bin_diff_2 = bin_diff * bin_diff;
                        A[i * bin + j] = ::std::exp(-1 * ground_weight * (float(bin_diff_2) / float(ground_dist_max_2)));
                    }
                }
            }
        
            params_ = { dim, bin, A};

        }

        size_t get_data_size() {
            return data_size_;
        }

        DISTFUNC<float> get_dist_func() {
            return fstdistfunc_;
        }

        void *get_dist_func_param() {
            return (void *) &params_;
        }

        ~QFSpace() {}
    };



    // ---------------
    //    Hellinger
    // ---------------

    // data struct for distance calculation in HellingerSpace
    struct space_params_Hel {
        size_t dim;
        size_t bin;
    };

    static float
        HelSqr(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
        float *pVect1 = (float *)pVect1v;
        float *pVect2 = (float *)pVect2v;
       
        space_params_Hel sparam = *((space_params_Hel*)qty_ptr);

        float res = 0;
        // add the histogram distance for each dimension
        for (size_t d = 0; d < sparam.dim; d++) {
            for (size_t i = 0; i < sparam.bin; i++) {
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

        space_params_Hel params_;

    public:
        HellingerSpace(size_t dim, size_t bin) {
            fstdistfunc_ = HelSqr;
            params_ = { dim, bin };
            data_size_ = dim * bin * sizeof(float);
        }

        size_t get_data_size() {
            return data_size_;
        }

        DISTFUNC<float> get_dist_func() {
            return fstdistfunc_;
        }

        void *get_dist_func_param() {
            return (void *) &params_;
        }

        ~HellingerSpace() {}
    };

}
