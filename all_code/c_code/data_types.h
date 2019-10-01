#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <float.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_cblas.h>
#include <gsl/gsl_statistics_double.h>
#include <gsl/gsl_statistics_int.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_sort.h>

#ifndef DATA_TYPES_H
#define DATA_TYPES_H

#define DATA_ID(_data, _id) \
    (_data.set + _id * _data.row_stride)

#define DATA_ID_NS(_data, _id) \
    (_data.set + _id * _data.flen)

typedef struct
{
    long memory_len;
    void *memory_block;

    long len;
    double *point_set;

    long flen;
    double *set;
} flag_data;

// think about implementing those...
typedef struct
{
    long memory_len;
    void *memory_block;

    long len;
    long flen;
    double *set;
    long row_stride;
    long col_stride;
} data_workspc;

typedef struct
{
    long memory_len;
    void *memory_block;

    long len;
    long *set;
} data_feature_range;

typedef struct
{
    long memory_len;
    void *memory_block;

    long data_len;
    long *data_cluster_ids;

    long cluster_count;
    long *cluster_sizes;
} data_clustering;

#endif
