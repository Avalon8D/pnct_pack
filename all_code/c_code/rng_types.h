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

#ifndef RNG_TYPES_H
#define RNG_TYPES_H

typedef struct
{
    long memory_len;
    void *memory_block;

    long count;
    long *sample;
} pair_sample;

typedef struct
{
    long memory_len;
    void *memory_block;

    long sample_range;
    double *w_threshold_table;
    long *w_id_table;
} sample_weights_tables;

#endif