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

#ifndef PROCLUS_TYPES_H
#define PROCLUS_TYPES_H

typedef struct
{
	long count;
	long sample_size;
	long *sample;
	long *off_currents;
	long *currents;
	long bad_count;
	long *bad;
} medoid_workspc;

typedef struct
{
	long memory_len;
	void *memory_block;

	long data_flen;
	double *range;
	double *point;
	double *subspace_scores;

	size_t subspace_count;
	size_t *subspace_arg;

	double least_cluster_size;
	double *cluster_sizes;
} proclus_buffer_workspc;

typedef struct
{
	long memory_len;
	void *memory_block;

	long *data_cluster_ids;
	medoid_workspc medoid;
	proclus_buffer_workspc buffer;
} proclus_workspc;

typedef struct
{
	long memory_len;
	void *memory_block;

	long cluster_count;
	long subspace_count;
	long *subspace_ranges;
	long *subspaces;
} proclus_subspaces;

typedef struct
{
	long memory_len;
	void *memory_block;

	long count;
	long *ids;
} medoid_set;

#endif