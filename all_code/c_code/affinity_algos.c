#include <data_types.h>
#include <cluster_stats_types.h>
#include <affinity_types.h>
#include <general_util.c>
#include <vector_util.c>
#include <rng_util.c>
#include <cluster_stats.c>

#ifndef AFFINITY_ALGOS_C
#define AFFINITY_ALGOS_C

// #define AFFINITY_ALGOS_VERBOSE
#define LAPLACIAN_COMPONENTS_SIZE(data_len) \
	((1 + data_len) * data_len * sizeof (double))

void laplacian_components_alloc (laplacian_components *components, long data_len)
{
	long memory_len = LAPLACIAN_COMPONENTS_SIZE (data_len);
	void *memory_block = malloc (memory_len);

	components->memory_len = memory_len;
	components->memory_block = memory_block;

	components->data_len = data_len;
	components->vectors = memory_block;
	memory_block += data_len * data_len * sizeof (double);

	components->values = memory_block;
}

// returns largest affinity
long affinity_mean_distances (double *affinity_row, data_clustering clustering,
                              double *cluster_mean_distances)
{
	long data_len = clustering.data_len;
	long cluster_count = clustering.cluster_count;

	vec_set_scal (cluster_mean_distances, 0, data_len, 1);

	{
		long *data_cluster_id = clustering.data_cluster_ids;
		double *data_affinity_val = affinity_row;

		long data_id;

		for (data_id = 0; data_id < data_len; ++data_id,
		        data_cluster_id++, data_affinity_val++)
		{
			cluster_mean_distances[*data_cluster_id] += *data_affinity_val;
		}
	}

	vec_div_long (cluster_mean_distances, clustering.cluster_sizes, cluster_count, 1, 1);
	long amax_mean_affinity = 0;

	{
		double max_mean_affinity = -DBL_MAX;
		double *cluster_mean_distance = cluster_mean_distances;

		long cluster_id;

		for (cluster_id = 0; cluster_id < cluster_count;
		        ++cluster_count, cluster_mean_distance++)
		{
			long smaller = max_mean_affinity < *cluster_mean_distance;
			max_mean_affinity = (smaller ? *cluster_mean_distance : max_mean_affinity);
			amax_mean_affinity = (smaller ? cluster_id : amax_mean_affinity);
		}
	}

	return amax_mean_affinity;
}

// dist_buffer has length at least base_clustering.cluster_count
void affinity_assign_from_clusters (double *cross_affinity, data_clustering base_clustering,
                                    data_clustering resulting_clustering, double *dist_buffer)
{
	long cluster_count = base_clustering.cluster_count;
	long base_data_len = base_clustering.data_len;
	long data_len = resulting_clustering.data_len;

	long *cluster_sizes = resulting_clustering.cluster_sizes;
	memset (cluster_sizes, 0, cluster_count * sizeof (long));

	{
		double *cross_affinity_row = cross_affinity;
		long *data_cluster_ids = resulting_clustering.data_cluster_ids;
		long data_id;

		for (data_id = 0; data_id < data_len; ++data_id,
		        data_cluster_ids++, cross_affinity_row += base_data_len)
		{
			long cluster_id = affinity_mean_distances (cross_affinity_row, base_clustering, dist_buffer);
			*data_cluster_ids = cluster_id;
			cluster_sizes[cluster_id]++;
		}
	}
}

double min_dist2s_to_centroid (data_workspc data, clustering_centroids centroids,
                               long *data_ids, double *dist_buffer, long id_count,
                               long centroid_id)
{
	long id;
	double dist_sum = 0;
	double *centroid_point = DATA_ID_NS (centroids, centroid_id);

	for (id = 0; id < id_count; ++id)
	{
		double curr_dist = vec_sqr_dist2 (DATA_ID (data, data_ids[id]), centroid_point,
		                                  data.flen, data.col_stride, 1);

		curr_dist = (curr_dist < dist_buffer[id] ? curr_dist : dist_buffer[id]);
		dist_buffer[id] = curr_dist;

		dist_sum += curr_dist;
	}

	return dist_sum;
}

void centroids_fast_distw_data_sample (data_workspc data, clustering_centroids centroids,
                                       long initial_sample_size, long *data_id_base,
                                       double *dist_buffer)
{
	{
		long id;

		for (id = 0; id < data.len; ++id)
		{
			data_id_base[id] = id;
		}

		for (id = 0; id < initial_sample_size; ++id)
		{
			dist_buffer[id] = DBL_MAX;
		}

		integer_partial_shuffle (data.len, initial_sample_size, data_id_base);
	}

	long data_flen = centroids.flen;
	double *current_centroid = centroids.set;

	{
		long rand_arg = gsl_rng_uniform_int (rand_gen, initial_sample_size);
		vec_set (current_centroid, DATA_ID (data, data_id_base[rand_arg]), data_flen, 1, data.col_stride);
		--initial_sample_size;
		data_id_base[rand_arg] = data_id_base[initial_sample_size];
	}

	long current_centroid_id;
	long current_sample_size;
	long sample_size = centroids.count;

	for (current_sample_size = 0, current_centroid_id = 0;
	        current_sample_size < sample_size; current_centroid += data_flen,
	        ++current_centroid_id, ++current_sample_size)
	{
		double dist_sum = min_dist2s_to_centroid (data, centroids, data_id_base,
		                  dist_buffer, initial_sample_size,
		                  current_centroid_id);

		long rand_arg = rng_weighted_int_naive (initial_sample_size,
		                                        dist_buffer, dist_sum, 0);

		vec_set (current_centroid, DATA_ID (data, data_id_base[rand_arg]), data_flen, 1, data.col_stride);

		--initial_sample_size;
		data_id_base[rand_arg] = data_id_base[initial_sample_size];
		dist_buffer[rand_arg] = dist_buffer[initial_sample_size];
	}

	#ifdef AFFINITY_ALGOS_VERBOSE

	{
		printf ("\n");

		double *centroid_bin = centroids.set;

		long centroid_id;

		for (centroid_id = 0; centroid_id < sample_size;
		        ++centroid_id)
		{
			long bin_id;

			for (bin_id = 0; bin_id < data_flen;
			        ++centroid_bin, ++bin_id)
			{
				printf ("%.2E,", *centroid_bin);
			}

			printf ("\n");
		}
	}

	#endif
}

double centroids_sqr_score (data_workspc data, data_clustering clustering,
                            clustering_centroids centroids)
{
	long data_len = data.len;
	long data_flen = data.flen;

	double *data_point = data.set;
	long *data_cluster_id = clustering.data_cluster_ids;

	double score = 0;
	long data_id;

	for (data_id = 0; data_id < data_len;
	        ++data_cluster_id, data_point += data.row_stride,
	        ++data_id)
	{
		score += vec_sqr_dist2 (data_point, DATA_ID_NS (centroids, *data_cluster_id),
		                        data_flen, data.col_stride, 1);
	}

	return score / data_len;
}

double centroids_data_cluster_ids (data_workspc data, data_clustering clustering,
                                   clustering_centroids centroids)
{
	long data_len = data.len;
	long data_flen = data.flen;
	long centroid_count = centroids.count;

	double *data_point = data.set;
	long *data_cluster_id = clustering.data_cluster_ids;
	long *cluster_sizes = clustering.cluster_sizes;
	memset (cluster_sizes, 0, centroid_count * sizeof (long));

	double score = 0;
	long data_id;

	for (data_id = 0; data_id < data_len;
	        ++data_cluster_id, data_point += data.row_stride,
	        ++data_id)
	{
		double *centroid = centroids.set;
		double closest_dist = DBL_MAX;
		long closest_centroid = 0;
		long centroid_id;

		for (centroid_id = 0; centroid_id < centroid_count;
		        centroid += data_flen, ++centroid_id)
		{
			double dist_to_centroid = vec_sqr_dist2 (data_point, centroid,
			                          data_flen, data.col_stride, 1);

			long smaller = dist_to_centroid < closest_dist;
			closest_dist = (smaller ? dist_to_centroid : closest_dist);
			closest_centroid = (smaller ? centroid_id : closest_centroid);
		}

		*data_cluster_id = closest_centroid;
		cluster_sizes[closest_centroid]++;
		score += closest_dist;
	}

	return score / data_len;
}

void centroid_cluster_mean_eval (data_workspc data, data_clustering clustering,
                                 clustering_centroids centroids)
{
	long data_len = data.len;
	long data_flen = data.flen;
	long centroid_count = centroids.count;

	double *data_point = data.set;
	long *data_cluster_id = clustering.data_cluster_ids;

	{
		double *centroid = centroids.set;
		long centroid_id;

		for (centroid_id = 0; centroid_id < centroid_count;
		        centroid += data_flen, ++centroid_id)
		{
			vec_set_scal (centroid, 0, data_flen, 1);
		}
	}

	{
		long data_id;

		for (data_id = 0; data_id < data_len;
		        data_point += data.row_stride, ++data_cluster_id,
		        ++data_id)
		{
			vec_add (DATA_ID_NS (centroids, *data_cluster_id),
			         data_point, data_flen, 1, data.col_stride);
		}
	}

	{
		double *centroid = centroids.set;
		long *cluster_sizes = clustering.cluster_sizes;
		long centroid_id;

		for (centroid_id = 0; centroid_id < centroid_count;
		        centroid += data_flen, ++cluster_sizes,
		        ++centroid_id)
		{
			long cluster_size = *cluster_sizes;

			if (cluster_size)
			{
				vec_div_scal (centroid, (double) cluster_size,
				              data_flen, 1);
			}

			else
			{
				long rand_arg = gsl_rng_uniform_int (rand_gen, data_len);
				memcpy (centroid, DATA_ID (data, rand_arg), data_flen * sizeof (double));
			}
		}
	}
}

void kmeans_iteration (data_workspc data, data_clustering clustering,
                       clustering_centroids centroids,
                       long trial_max)
{
	do
	{
		centroids_data_cluster_ids (data, clustering, centroids);
		centroid_cluster_mean_eval (data, clustering, centroids);
	}
	while (trial_max--);

	#ifdef AFFINITY_ALGOS_VERBOSE

	{
		printf ("\n");

		long centroid_count = centroids.count;
		long data_flen = data.flen;

		double *centroid_bin = centroids.set;
		long centroid_id;

		for (centroid_id = 0; centroid_id < centroid_count;
		        ++centroid_id)
		{
			long feature_id;

			for (feature_id = 0; feature_id < data_flen;
			        ++centroid_bin, ++feature_id)
			{
				printf ("%.2E,", *centroid_bin);
			}

			printf ("\n");
		}
	}

	{
		printf ("\n");

		long data_len = data.len;
		long *data_cluster_id = clustering.data_cluster_ids;

		long data_id;

		for (data_id = 0; data_id < data_len;
		        ++data_cluster_id, ++data_id)
		{
			printf ("%ld,", *data_cluster_id);
		}

		printf ("\n");
	}

	#endif
}

void kmeans_one_shot (data_workspc data, data_clustering clustering,
                      long trial_max)
{
	clustering_centroids centroids;
	clustering_centroids_alloc (&centroids, clustering.cluster_count,
	                            data.flen);

	double *dist_buffer = malloc (clustering.cluster_count * sizeof (double));

	centroids_fast_distw_data_sample (data, centroids, clustering.cluster_count,
	                                  clustering.data_cluster_ids,
	                                  dist_buffer);

	kmeans_iteration (data, clustering, centroids,
	                  trial_max);

	free (centroids.memory_block);
}

// reassigns points with negative cluster labels
// does not reevaluate cluster centroids
long kmeans_reassign_small (data_workspc data, data_clustering clustering,
                            clustering_centroids centroids, double small_fraction)
{
	long data_len = data.len;
	long data_flen = data.flen;
	long centroid_count = centroids.count;

	{
		long small_thresh = small_fraction * data_len;
		small_thresh = max (1, small_thresh);
		small_thresh = min (small_thresh, data_len);

		long *cluster_size = clustering.cluster_sizes;
		long centroid_id;

		for (centroid_id = 0; centroid_id < centroid_count;
		        ++centroid_id, ++cluster_size)
		{
			long size = *cluster_size;
			*cluster_size = (size <= small_thresh ? -1 : size);
		}
	}

	double score = 0;

	{
		double *data_point = data.set;
		long *data_cluster_id = clustering.data_cluster_ids;
		long *cluster_sizes = clustering.cluster_sizes;

		long data_id;

		for (data_id = 0; data_id < data_len;
		        ++data_cluster_id, data_point += data.row_stride,
		        ++data_id)
		{
			if (cluster_sizes[*data_cluster_id] != -1)
			{
				continue;
			}

			double *centroid = centroids.set;
			long *cluster_size = cluster_sizes;

			double closest_dist = DBL_MAX;
			long closest_centroid = 0;
			long centroid_id;

			for (centroid_id = 0; centroid_id < centroid_count;
			        centroid += data_flen, ++cluster_size, ++centroid_id)
			{
				if (*cluster_size == -1)
				{
					continue;
				}

				double dist_to_centroid = vec_sqr_dist2 (data_point, centroid, data_flen, data.col_stride, 1);

				long smaller = dist_to_centroid < closest_dist;
				closest_dist = (smaller ? dist_to_centroid : closest_dist);
				closest_centroid = (smaller ? centroid_id : closest_centroid);
			}

			*data_cluster_id = closest_centroid;
			cluster_sizes[closest_centroid]++;
			score += closest_dist;
		}
	}

	return score / data_len;
}

// remakes clusterings with potentially empty cluster labels (within cluster count)
// into a clustering with potentially less clusters

long clustering_normalize (long *data_cluster_ids, long *cluster_sizes,
                           long data_len, long cluster_count)
{
	long *used_cluster_labels = calloc (cluster_count, sizeof (long));
	memset (cluster_sizes, 0, cluster_count * sizeof (long));

	{
		long *data_cluster_id = data_cluster_ids;

		long data_id;

		for (data_id = 0; data_id < data_len;
		        ++data_id, ++data_cluster_id)
		{
			cluster_sizes[*data_cluster_id]++;
			used_cluster_labels[*data_cluster_id] = 1;
		}
	}

	long normal_cluster_count;

	{
		long *used_cluster_label = used_cluster_labels;
		long *cluster_size = cluster_sizes;

		long cluster_id;

		for (cluster_id = 0, normal_cluster_count = 0;
		        cluster_id < cluster_count; ++cluster_id,
		        ++used_cluster_label, ++cluster_size)
		{
			for (; cluster_id < cluster_count && *used_cluster_label;
			        ++cluster_id, ++used_cluster_label, ++cluster_size)
			{
				*used_cluster_label = normal_cluster_count;
				cluster_sizes[normal_cluster_count] = *cluster_size;
				++normal_cluster_count;
			}
		}
	}

	{
		long *data_cluster_id = data_cluster_ids;

		long data_id;

		for (data_id = 0; data_id < data_len;
		        ++data_id, ++data_cluster_id)
		{
			*data_cluster_id = used_cluster_labels[*data_cluster_id];
		}
	}

	return normal_cluster_count;
}

long kmeans_no_small_one_shot (data_workspc data, data_clustering clustering,
                               long trial_max, double small_fraction)
{
	clustering_centroids centroids;
	clustering_centroids_alloc (&centroids, clustering.cluster_count,
	                            data.flen);

	double *dist_buffer = malloc (clustering.cluster_count * sizeof (double));

	centroids_fast_distw_data_sample (data, centroids, clustering.cluster_count,
	                                  clustering.data_cluster_ids,
	                                  dist_buffer);

	kmeans_iteration (data, clustering, centroids, trial_max);
	kmeans_reassign_small (data, clustering, centroids, small_fraction);

	free (centroids.memory_block);

	return clustering_normalize (clustering.data_cluster_ids, clustering.cluster_sizes,
	                             clustering.data_len, clustering.cluster_count);
}

void spectral_clustering_form_laplacian (double *affinity_matrix, double *laplacian, long data_len)
{
	memcpy (laplacian, affinity_matrix, data_len * data_len * sizeof (double));

	{
		double *affinity_matrix_row = affinity_matrix;
		double *laplacian_mat_col = laplacian;
		double *laplacian_row_bin = laplacian;

		long mat_id_row;

		for (mat_id_row = 0; mat_id_row < data_len;
		        affinity_matrix_row += data_len,
		        ++laplacian_mat_col, ++mat_id_row)
		{
			double inv_row_norm1 = 1 / sqrt (vec_sum (affinity_matrix_row, data_len, 1));

			long mat_id_col;
			double *laplacian_col_bin = laplacian_mat_col;

			for (mat_id_col = 0; mat_id_col < data_len;
			        laplacian_col_bin += data_len,
			        ++laplacian_row_bin, ++mat_id_col)
			{
				*laplacian_row_bin = - *laplacian_row_bin * inv_row_norm1;
				*laplacian_col_bin *= inv_row_norm1;
			}
		}
	}

	{
		double *laplacian_diag = laplacian;
		long diag_id;

		for (diag_id = 0; diag_id < data_len;
		        laplacian_diag += (data_len + 1),
		        ++diag_id)
		{
			*laplacian_diag += 1.0;
		}
	}

	#ifdef AFFINITY_ALGOS_VERBOSE

	{
		printf ("\n");

		double *laplacian_mat_bin = laplacian;
		long mat_id_row;

		for (mat_id_row = 0; mat_id_row < data_len;
		        ++mat_id_row)
		{
			long mat_id_col;

			for (mat_id_col = 0; mat_id_col < data_len;
			        ++laplacian_mat_bin, ++mat_id_col)
			{
				printf ("%.2E,", *laplacian_mat_bin);
			}

			printf ("\n");
		}
	}

	#endif
}

void spectral_clustering_form_laplacian_inplace (double *affinity_matrix, long data_len, double *rows_inorm)
{
	{
		double *affinity_matrix_row = affinity_matrix;
		double *row_inorm = rows_inorm;
		long mat_id_row;

		for (mat_id_row = 0; mat_id_row < data_len;
		        affinity_matrix_row += data_len,
		        ++row_inorm, ++mat_id_row)
		{
			*row_inorm = 1 / sqrt (vec_sum (affinity_matrix_row, data_len, 1));
		}
	}

	{
		double *laplacian_mat_col = affinity_matrix;
		double *laplacian_row_bin = affinity_matrix;
		double *row_inorm = rows_inorm;

		long mat_id_row;

		for (mat_id_row = 0; mat_id_row < data_len;
		        ++row_inorm, ++laplacian_mat_col, ++mat_id_row)
		{
			double row_inorm_val = *row_inorm;
			double *laplacian_col_bin = laplacian_mat_col;
			long mat_id_col;

			for (mat_id_col = 0; mat_id_col < data_len;
			        laplacian_col_bin += data_len,
			        ++laplacian_row_bin, ++mat_id_col)
			{
				*laplacian_row_bin = - *laplacian_row_bin * row_inorm_val;
				*laplacian_col_bin *= row_inorm_val;
			}
		}
	}

	{
		double *laplacian_diag = affinity_matrix;
		long diag_id;

		for (diag_id = 0; diag_id < data_len;
		        laplacian_diag += (data_len + 1),
		        ++diag_id)
		{
			*laplacian_diag += 1.0;
		}
	}

	#ifdef AFFINITY_ALGOS_VERBOSE

	{
		printf ("\n");

		double *laplacian_mat_bin = laplacian;
		long mat_id_row;

		for (mat_id_row = 0; mat_id_row < data_len;
		        ++mat_id_row)
		{
			long mat_id_col;

			for (mat_id_col = 0; mat_id_col < data_len;
			        ++laplacian_mat_bin, ++mat_id_col)
			{
				printf ("%.2E,", *laplacian_mat_bin);
			}

			printf ("\n");
		}
	}

	#endif
}

void normalized_affinity_matrix_inplace (double *affinity_matrix, long data_len, double *rows_inorm)
{
	{
		double *affinity_matrix_row = affinity_matrix;
		double *row_inorm = rows_inorm;
		long mat_id_row;

		for (mat_id_row = 0; mat_id_row < data_len;
		        affinity_matrix_row += data_len,
		        ++row_inorm, ++mat_id_row)
		{
			*row_inorm = 1 / sqrt (vec_sum (affinity_matrix_row, data_len, 1));
		}
	}

	{
		double *n_affinity_mat_col = affinity_matrix;
		double *n_affinity_row_bin = affinity_matrix;
		double *row_inorm = rows_inorm;

		long mat_id_row;

		for (mat_id_row = 0; mat_id_row < data_len;
		        ++row_inorm, ++n_affinity_mat_col, ++mat_id_row)
		{
			double row_inorm_val = *row_inorm;
			double *n_affinity_col_bin = n_affinity_mat_col;
			long mat_id_col;

			for (mat_id_col = 0; mat_id_col < data_len;
			        n_affinity_col_bin += data_len,
			        ++n_affinity_row_bin, ++mat_id_col)
			{
				*n_affinity_row_bin = *n_affinity_row_bin * row_inorm_val;
				*n_affinity_col_bin *= row_inorm_val;
			}
		}
	}

	#ifdef AFFINITY_ALGOS_VERBOSE

	{
		printf ("\n");

		double *n_affinity_mat_bin = laplacian;
		long mat_id_row;

		for (mat_id_row = 0; mat_id_row < data_len;
		        ++mat_id_row)
		{
			long mat_id_col;

			for (mat_id_col = 0; mat_id_col < data_len;
			        ++n_affinity_mat_bin, ++mat_id_col)
			{
				printf ("%.2E,", *n_affinity_mat_bin);
			}

			printf ("\n");
		}
	}

	#endif
}

void spectral_clustering_form_components (double *affinity_matrix, laplacian_components components)
{
	long data_len = components.data_len;

	double *components_values = components.values;
	double *components_vectors = components.vectors;

	spectral_clustering_form_laplacian (affinity_matrix, components_vectors,
	                                    data_len);

	SPD_spectral_decomp (components_vectors, components_values, data_len);

	#ifdef AFFINITY_ALGOS_VERBOSE

	{
		printf ("\n");

		long value_id;

		for (value_id = 0; value_id < data_len;
		        ++components_values, ++value_id)
		{
			printf ("%.2E,", *components_values);
		}

		printf ("\n");
	}

	{
		printf ("\n");

		long mat_id_row;

		for (mat_id_row = 0; mat_id_row < data_len;
		        ++mat_id_row)
		{
			long mat_id_col;

			for (mat_id_col = 0; mat_id_col < data_len;
			        ++components_vectors, ++mat_id_col)
			{
				printf ("%.2E,", *components_vectors);
			}

			printf ("\n");
		}
	}

	#endif
}

#endif