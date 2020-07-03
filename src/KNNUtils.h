#pragma once

enum knn_library
{
	KNN_FLANN = 0,
//	KNN_HSNW = 1,
//	KNN_KGRAPH = 2
};

enum knn_distance_metric
{
	KNN_METRIC_QF = 0,
//	KNN_METRIC_EMD = 1
};

template<class T>
struct QF
{
    typedef bool is_kdtree_distance;
    typedef T ElementType;
    typedef typename Accumulator<T>::Type ResultType;

    /*
    @param size Size of one point in the index (and database), see nn_index.h (842), usage kdtree_index.h (586)
    */
    template <typename Iterator1, typename Iterator2>
    ResultType operator()(Iterator1 a, Iterator2 b, size_t size, ResultType /*worst_dist*/ = -1) const
    {
        ResultType result = ResultType();
        ResultType diff;

        return result;
    }

    /*
    @param size Size of one point in the index (and database), see nn_index.h (842), usage kdtree_index.h (605)
    */
    template <typename U, typename V>
    inline ResultType accum_dist(const U& a, const V& b, int) const
    {
        return 0;
    }
};

template<class T>
struct EMD
{

};