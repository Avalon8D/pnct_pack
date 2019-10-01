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

#ifndef CLUSTER_STATS_TYPES_H
#define CLUSTER_STATS_TYPES_H

typedef struct
{
	long memory_len;
	void *memory_block;

	long count;
	long flen;
	double *set_up;
	double *set_down;
} clustering_margins;

typedef struct
{
	long memory_len;
	void *memory_block;

	long count;
	long flen;
	double *set;
} clustering_centroids;

typedef struct
{
	long memory_len;
	void *memory_block;

	long cluster_count;
	long cluster_ranges_size;
	long *cluster_ranges;

	long data_len;
	long *clusters_data_ids;
} clustering_sample_space;

#endif