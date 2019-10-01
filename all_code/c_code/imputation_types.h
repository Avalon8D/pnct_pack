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
#include <gsl/gsl_rstat.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_sort.h>

#ifndef IMPUTATION_TYPES_H
#define IMPUTATION_TYPES_H

typedef struct
{
	long memory_len;
	void *memory_block;

	long cluster_count;
	long neighbour_count;
	double *dists_buffer;
} KNN_weights_buffer;

typedef struct
{
	long memory_len;
	void *memory_block;

	long len;
	long *ids;
} index_sample;

#endif