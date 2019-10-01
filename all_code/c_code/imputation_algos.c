#include <data_types.h>
#include <cluster_stats_types.h>
#include <imputation_types.h>
#include <vector_util.c>
#include <rng_util.c>
#include <general_util.c>
#include <data_util.c>
#include <cluster_stats.c>

#ifndef IMPUTATION_ALGOS_C
#define IMPUTATION_ALGOS_C

#define IMPUTATION_ALGOS_VERBOSE

#define KNN_BUFFER_SIZE(cluster_count, neighbour_count) \
	(cluster_count * neighbour_count * sizeof (double))

#define INDEX_SAMPLE_SIZE(sample_len) \
	(sample_len * sizeof (long))

// classes flen is assumed to be divisible by class_count
// classes is assumed to contain only positive values
// classes_rates is assumed to be, at least,
// (clustering.cluster_count * classes.flen) in length
void data_clustering_classes_mean_distrib_rate
(data_workspc classes, long class_flen,
 data_clustering clustering, double *classes_rates,
 long normalize)
{
	long classes_len = classes.len;
	long classes_flen = classes.flen;
	long class_count = classes_flen / class_flen;
	printf ("%ld, %ld, %ld\n", classes_len, classes_flen, class_count);

	long cluster_count = clustering.cluster_count;
	vec_set_scal (classes_rates, 0, classes_flen * cluster_count, 1);

	{
		long *data_cluster_id = clustering.data_cluster_ids;
		double *classes_line = classes.set;

		long class_id;

		for (class_id = 0; class_id < classes_len; ++class_id,
		        ++data_cluster_id, classes_line += classes_flen)
		{
			mat_normalize1_cols (classes_line, class_count, class_flen, 1, 1);
			vec_add (classes_rates + *data_cluster_id * classes_flen, classes_line, classes_flen, 1);
		}
	}

	if (normalize)
	{
		long *cluster_size = clustering.cluster_sizes;
		double *class_rates = classes_rates;

		long cluster_id;

		for (cluster_id = 0; cluster_id < cluster_count; ++cluster_id,
		        ++cluster_size, class_rates += classes_flen)
		{
			vec_div_scal (class_rates, (double) *cluster_size, classes_flen, 1);
		}
	}
}

// samples sample_size points from index_sample and
// averages the following for each sample:
// generates a sample W ~ multi (class_count, sample_res, ws_vector)
// distributes the dim_id dimension of each data point
// to each of the class_count classes, with weight (W[i] / sample_res) for class i
// class_matrix is assumed to be class_count by data.flen, in row major order
// ws_sample is assumed to be of length class_count, supposed to hold a W sample
void data_sample_class_extrapolation_multinomial
(data_workspc data, double *class_matrix, long dim_id, long *index_sample,
 long sample_len, double *ws_vector, long class_count, long sample_size,
 unsigned sample_res, unsigned *ws_sample)
{
	integer_partial_shuffle (sample_len, sample_size, index_sample);

	long data_len = data.len;
	long data_flen = data.flen;

	vec_set_scal (class_matrix, 0, class_count * data_len, 1);

	{
		long *data_index = index_sample;
		double *class_col = class_matrix + dim_id;
		long sample_id;

		for (sample_id = 0; sample_id < sample_size;
		        ++sample_id, data_index++)
		{
			gsl_ran_multinomial (rand_gen, class_count, sample_res, ws_vector, ws_sample);

			double *class_bin = class_col;
			double *class_w = ws_vector;
			double data_bin = DATA_ID (data, *data_index)[dim_id];
			long class_id;

			for (class_id = 0; class_id < class_count;
			        ++class_id, class_bin += data_flen,
			        ++class_w)
			{
				*class_bin += data_bin * *class_w;
			}
		}
	}

	long sample_norm = 1 / (double) (sample_size * sample_res);
	vec_mul_scal (class_matrix + dim_id, sample_norm, class_count, data_flen);
}

#endif