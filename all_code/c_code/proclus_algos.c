#include <proclus_components_trials.c>
#include <proclus_components_clustering.c>
#include <data_util.c>
#include <general_util.c>

#ifndef PROCLUS_ALGOS_C
#define PROCLUS_ALGOS_C

// #define PROCLUS_ALGOS_VERBOSE

#define MEDOID_WORKSPC_SIZE(cluster_count, sample_size) \
	(sample_size * sizeof(long) + cluster_count * sizeof(long))

#define BUFFER_WORKSPC_SIZE(data_flen, cluster_count, subspace_count) \
	(cluster_count * data_flen * sizeof(double) + \
	 cluster_count * subspace_count * sizeof (size_t) + \
	 2 * cluster_count * sizeof (double) + \
	 data_flen * sizeof (double))

#define WORKSPC_SIZE(data_len, data_flen, cluster_count, subspace_count, sample_size) \
	(MEDOID_WORKSPC_SIZE(cluster_count, sample_size) + \
	 BUFFER_WORKSPC_SIZE(data_flen, cluster_count, subspace_count) + \
	 data_len * sizeof (long))

#define SUBSPACES_SIZE(subspace_count, cluster_count) \
	((cluster_count + 1) * sizeof (long) + \
	 cluster_count * subspace_count * sizeof (long))

#define MEDOID_SET_SIZE(medoid_count) \
	(medoid_count * sizeof (long))

void proclus_workspc_alloc (proclus_workspc *proclus, long data_len, long data_flen,
                            long cluster_count, size_t subspace_count, long sample_size,
                            long least_cluster_size)
{
	long memory_len = WORKSPC_SIZE (data_len, data_flen, cluster_count,
	                                subspace_count, sample_size);
	void *memory_block = malloc (memory_len);

	proclus->memory_len = memory_len;
	proclus->memory_block = memory_block;

	proclus->data_cluster_ids = memory_block;
	memory_block += data_len * sizeof (long);

	proclus->medoid.count = cluster_count;
	proclus->medoid.sample_size = sample_size;

	proclus->medoid.sample = memory_block;
	memory_block += proclus->medoid.sample_size * sizeof (long);

	proclus->medoid.currents = proclus->medoid.sample;
	proclus->medoid.off_currents = proclus->medoid.sample + cluster_count;

	proclus->medoid.bad_count = 0;
	proclus->medoid.bad = memory_block;
	memory_block += cluster_count * sizeof (long);

	proclus->buffer.data_flen = data_flen;
	proclus->buffer.range = memory_block;
	memory_block += cluster_count * sizeof (double);

	proclus->buffer.point = memory_block;
	memory_block += data_flen * sizeof (double);

	proclus->buffer.subspace_count = cluster_count * subspace_count;
	proclus->buffer.subspace_arg = memory_block;
	memory_block += cluster_count * subspace_count * sizeof (size_t);

	proclus->buffer.subspace_scores = memory_block;
	memory_block += cluster_count * data_flen * sizeof (double);

	proclus->buffer.least_cluster_size = least_cluster_size;
	proclus->buffer.cluster_sizes = memory_block;
}

void proclus_buffer_workspc_alloc (proclus_buffer_workspc *buffer, long data_flen,
                                   long cluster_count, long subspace_count)
{
	long memory_len = BUFFER_WORKSPC_SIZE (data_flen, cluster_count, subspace_count);
	void *memory_block = malloc (memory_len);

	buffer->data_flen = data_flen;
	buffer->range = memory_block;
	memory_block += cluster_count * sizeof (double);

	buffer->point = memory_block;
	memory_block += data_flen * sizeof (double);

	buffer->subspace_count = cluster_count * subspace_count;
	buffer->subspace_arg = memory_block;
	memory_block += cluster_count * subspace_count * sizeof (size_t);

	buffer->subspace_scores = memory_block;
	memory_block += cluster_count * data_flen * sizeof (double);

	buffer->least_cluster_size = 0;
	buffer->cluster_sizes = memory_block;
}

void proclus_subspaces_alloc (proclus_subspaces *subspcs, long subspace_count,
                              long cluster_count)
{
	long memory_len = SUBSPACES_SIZE (subspace_count, cluster_count);
	void *memory_block = malloc (memory_len);

	subspcs->memory_len = memory_len;
	subspcs->memory_block = memory_block;

	subspcs->cluster_count = cluster_count;
	subspcs->subspace_count = subspace_count;
	subspcs->subspace_ranges = memory_block;
	memory_block += (cluster_count + 1) * sizeof (long);

	subspcs->subspaces = memory_block;
}

void proclus_subspaces_mem_load (proclus_subspaces *subspcs, long *subspaces,
                                 long subspace_count, long *subspace_ranges,
                                 long cluster_count)
{
	subspcs->memory_len = 0;
	subspcs->memory_block = NULL;

	subspcs->cluster_count = cluster_count;
	subspcs->subspace_count = subspace_count;
	subspcs->subspace_ranges = subspace_ranges;
	subspcs->subspaces = subspaces;
}

void medoid_set_alloc (medoid_set *medoid, long medoid_count)
{
	long memory_len = MEDOID_SET_SIZE (medoid_count);
	void *memory_block = malloc (memory_len);

	medoid->memory_len = memory_len;
	medoid->memory_block = memory_block;

	medoid->count = medoid_count;
	medoid->ids = memory_block;
}

void medoid_set_mem_load (medoid_set *medoid, long *medoid_ids, long medoid_count)
{
	medoid->memory_len = 0;
	medoid->memory_block = NULL;

	medoid->count = medoid_count;
	medoid->ids = medoid_ids;
}

void proclus_trials_dist1_seg_score (data_workspc data, proclus_workspc proclus,
                                     proclus_subspaces subspcs, long *best_data_cluster_ids,
                                     long *medoid_bests, long trial_max, long non_improv_max,
                                     data_feature_range *features)
{
	double best_score = DBL_MAX;
	long cluster_count = proclus.medoid.count;
	long *medoid_currents = proclus.medoid.currents;
	long *data_cluster_ids = proclus.data_cluster_ids;

	long non_improv_cnt = 0;

	while (trial_max--)
	{
		proclus_medoid_subspaces_scores (data, proclus, features);
		proclus_medoid_subspaces (subspcs, proclus.buffer);
		proclus_data_cluster_ids (data, subspcs, proclus);

		double score = proclus_dist1_seg_score (data, subspcs, proclus);

		++non_improv_cnt;

		if (score < best_score)
		{
			best_score = score;
			memcpy (medoid_bests, medoid_currents, cluster_count * sizeof (long));
			memcpy (best_data_cluster_ids, data_cluster_ids, data.len * sizeof (long));
			non_improv_cnt = 0;
		}

		if (non_improv_cnt == non_improv_max)
		{
			break;
		}

		proclus_eval_bad_medoids (&proclus);
		proclus_resample_bad_medoids (proclus);
	}

	#ifdef PROCLUS_ALGOS_VERBOSE

	{
		printf ("\n%.2E\n", best_score);
	}

	{
		printf ("\n");

		long medoid_id;

		for (medoid_id = 0; medoid_id < cluster_count;
		        ++medoid_id, ++medoid_bests)
		{
			printf ("%ld,", *medoid_bests);
		}

		printf ("\n");
	}

	{
		printf ("\n");

		long data_id;

		for (data_id = 0; data_id < data.len;
		        ++data_id, ++best_data_cluster_ids)
		{
			printf ("%ld,", *best_data_cluster_ids);
		}

		printf ("\n");
	}

	#endif
}

void proclus_trials_interintra_dist1_seg_score
(data_workspc data, proclus_subspaces subspcs,
 proclus_workspc proclus, pair_sample pairs,
 long *best_data_cluster_ids, long *medoid_bests,
 long trial_max, long non_improv_max,
 data_feature_range *features)
{
	double best_score = DBL_MAX;
	long cluster_count = proclus.medoid.count;
	long *medoid_currents = proclus.medoid.currents;
	long *data_cluster_ids = proclus.data_cluster_ids;

	long non_improv_cnt = 0;

	while (trial_max--)
	{
		proclus_medoid_subspaces_scores (data, proclus, features);
		proclus_medoid_subspaces (subspcs, proclus.buffer);
		proclus_data_cluster_ids (data, subspcs, proclus);

		double score = proclus_interintra_dist1_seg_score (data, subspcs, proclus, pairs);
		++non_improv_cnt;

		if (score < best_score)
		{
			best_score = score;
			memcpy (medoid_bests, medoid_currents, cluster_count * sizeof (long));
			memcpy (best_data_cluster_ids, data_cluster_ids, data.len * sizeof (long));
			non_improv_cnt = 0;
		}

		if (non_improv_cnt == non_improv_max)
		{
			break;
		}

		proclus_eval_bad_medoids (&proclus);
		proclus_resample_bad_medoids (proclus);
	}

	#ifdef PROCLUS_ALGOS_VERBOSE

	{
		printf ("\n%.2E\n", best_score);
	}

	{
		printf ("\n");

		long medoid_id;

		for (medoid_id = 0; medoid_id < cluster_count;
		        ++medoid_id, ++medoid_bests)
		{
			printf ("%ld,", *medoid_bests);
		}

		printf ("\n");
	}

	{
		printf ("\n");

		long data_id;

		for (data_id = 0; data_id < data.len;
		        ++data_id, ++best_data_cluster_ids)
		{
			printf ("%ld,", *best_data_cluster_ids);
		}

		printf ("\n");
	}

	#endif
}

void data_clustering_refine (data_workspc data, data_clustering clustering,
                             medoid_set medoid, proclus_subspaces subspcs,
                             proclus_buffer_workspc buffer, data_feature_range *features)
{
	data_clustering_subspaces_scores (data, clustering, medoid, buffer, features);
	data_clustering_subspaces (subspcs, buffer);
	data_clustering_cluster_ids (data, clustering, medoid, subspcs);
}

void proclus_clustering_single_assign
(data_workspc data, data_clustering clustering, medoid_set medoid,
 proclus_subspaces subspcs, proclus_buffer_workspc buffer,
 data_feature_range *features)
{
	data_clustering_medoid_subspaces_scores (data, medoid, buffer, features);
	data_clustering_subspaces (subspcs, buffer);
	data_clustering_cluster_ids (data, clustering, medoid, subspcs);

	data_clustering_subspaces_scores (data, clustering, medoid, buffer, features);
	data_clustering_subspaces (subspcs, buffer);
	data_clustering_cluster_ids (data, clustering, medoid, subspcs);
}

// if weighted > 1, does furthest to closest, then does distance biased to sample medoids
// weighted, then, is taken to be the sample size from the furthest to closest sample to
// the distance biased one
// if prufer is non zero, then pair sample is generated from gen_prufer_pair sample
// then inter_intra_sample_size implies (data.len - 1) * inter_intra_sample_size pairs sampled
// this mode guarantees that there will always be pairs within clusters and between cluster
// which is important for the inter_intra scoring function to have a definite value

void proclus_clustering_pieces_alloc
(proclus_workspc *proclus, pair_sample *pairs,
 long data_len, long data_flen, long cluster_count,
 long subspace_count, long inter_intra_sample_size,
 long medoid_sample_size, long least_cluster_size,
 long prufer)
{
	proclus_workspc_alloc (proclus, data_len, data_flen, cluster_count,
	                       subspace_count, medoid_sample_size, least_cluster_size);

	if (prufer)
	{
		gen_prufer_pair_sample (pairs, inter_intra_sample_size, data_len);
	}

	else
	{
		gen_pair_sample (pairs, inter_intra_sample_size, data_len);
	}
}

void proclus_clustering_data_alloc
(data_clustering *clustering, medoid_set *medoid,
 proclus_subspaces *subspcs,
 long data_len, long cluster_count, long subspace_count)
{
	data_clustering_alloc (clustering, data_len, cluster_count);
	medoid_set_alloc (medoid, cluster_count);
	proclus_subspaces_alloc (subspcs, subspace_count, cluster_count);
}

void proclus_clustering_workspc_init
(data_workspc data, proclus_workspc proclus,
 long initial_sample_size,
 data_feature_range *features, long weighted)
{
	initial_sample_size = min (data.len, initial_sample_size);
	double *dist_buffer = malloc (initial_sample_size * sizeof (double));

	if (weighted == 1)
	{
		proclus_fast_weighted_data_sample (data, proclus.medoid.sample, proclus.medoid.sample_size,
		                                   initial_sample_size, proclus.data_cluster_ids, dist_buffer,
		                                   features, 0);
	}

	else
		if (weighted > 1)

		{
			long weighted_sample_size = weighted;
			long *weighted_sample = malloc (weighted_sample_size * sizeof (long));

			proclus_fast_weighted_data_sample (data, weighted_sample, weighted_sample_size,
			                                   initial_sample_size, proclus.data_cluster_ids,
			                                   dist_buffer, features, 0);

			proclus_fast_refined_data_sample (data, proclus.medoid.sample, proclus.medoid.sample_size,
			                                  weighted_sample_size, weighted_sample, dist_buffer,
			                                  features, 1);
		}

		else
		{
			proclus_fast_refined_data_sample (data, proclus.medoid.sample, proclus.medoid.sample_size,
			                                  initial_sample_size, proclus.data_cluster_ids, dist_buffer,
			                                  features, 0);
		}
}

// assumes all workspaces have been initialized, and merely executes the trials

void proclus_clustering_base
(data_workspc data, data_clustering clustering,
 medoid_set medoid, proclus_subspaces subspcs,
 proclus_workspc proclus, pair_sample pairs,
 long trial_max, long non_improv_max,
 data_feature_range *features)
{
	proclus_trials_interintra_dist1_seg_score (data, subspcs, proclus, pairs, clustering.data_cluster_ids,
	        medoid.ids, trial_max, non_improv_max, features);

	data_clustering_refine (data, clustering, medoid, subspcs,
	                        proclus.buffer, features);
}

void proclus_clustering_one_shot
(data_workspc data, data_clustering clustering,
 medoid_set medoid, proclus_subspaces subspcs,
 long trial_max, long non_improv_max,
 long inter_intra_sample_size, long medoid_sample_size,
 long initial_sample_size, long least_cluster_size,
 data_feature_range *features, long weighted, long prufer)
{
	proclus_workspc proclus;
	pair_sample pairs;

	proclus_clustering_pieces_alloc (&proclus, &pairs, data.len, data.flen,
	                                 clustering.cluster_count, subspcs.subspace_count,
	                                 inter_intra_sample_size, medoid_sample_size,
	                                 least_cluster_size, prufer);

	proclus_clustering_workspc_init (data, proclus, initial_sample_size, features, weighted);

	proclus_clustering_base (data, clustering, medoid, subspcs, proclus,
	                         pairs, trial_max, non_improv_max, features);

	free (proclus.memory_block);
	free (pairs.memory_block);
}

// implement score likelihood based sampling of bootstrap parameters

// fills a data_len by data_len matrix with the
// average occurrence of pair (i,j) of points,
// in the same cluster
// sampler receive the clustering score of the previous run and sampling_args respectively
// first call it is initialized as max double value, long bootstrap_max
// cluster_sampler and subspace_sampler must return values between 2 and their respective max
// cluster_avg_frac is to be used as the fraction of the average cluster size (data_len / cluster_count)
// to be considered as a small cluster size for the proclus algorithm

void proclus_clustering_bootstrap_base
(data_workspc data, double *affinity_matrix,
 data_clustering clustering, medoid_set medoid,
 proclus_subspaces subspcs, proclus_workspc proclus,
 pair_sample pairs, long (*cluster_sampler) (double, void *),
 long (*subspace_sampler) (double, void *),
 long trial_max, long non_improv_max,
 long bootstrap_max, double cluster_avg_frac,
 data_feature_range *features, void *sampling_args)
{
	double score_sum = 0;
	double curr_score = DBL_MAX;

	while (bootstrap_max--)
	{
		long cluster_count = cluster_sampler (curr_score, sampling_args);
		long subspace_count = subspace_sampler (curr_score, sampling_args);

		clustering.cluster_count = cluster_count;
		medoid.count = cluster_count;
		subspcs.cluster_count = cluster_count;

		subspcs.subspace_count = subspace_count;
		proclus.buffer.subspace_count = cluster_count * subspace_count;

		proclus.medoid.count = cluster_count;
		proclus.medoid.off_currents = proclus.medoid.sample + cluster_count;
		proclus.buffer.least_cluster_size = (cluster_avg_frac * data.len) / (double) cluster_count;

		proclus_clustering_base (data, clustering, medoid, subspcs, proclus,
		                         pairs, trial_max, non_improv_max, features);

		curr_score = data_clustering_interintra_dist1_seg_score (data, clustering, subspcs, pairs);

		score_sum += curr_score;

		double_mat_wsum_matches (affinity_matrix, clustering.data_cluster_ids, data.len, curr_score);
	}

	vec_div_scal (affinity_matrix, score_sum, data.len * data.len, 1);
}

// if data_subset is not null, it is used instead
// to sample the medoids used to cluster
// subset_original_ids is then supposed to be
// what are the corresponding ids
// in the original data provided
// if NULL, ids are no changing,
// which amounts to assume that all ids
// are within data.len
void proclus_clustering_bootstrap
(data_workspc data, double *affinity_matrix,
 long max_cluster_count, long max_subspace_count,
 long (*cluster_sampler) (double, void *),
 long (*subspace_sampler) (double, void *),
 long trial_max, long non_improv_max,
 long bootstrap_max, double cluster_avg_frac,
 long inter_intra_sample_size, long medoid_sample_size,
 long initial_sample_size,
 data_feature_range *features, long weighted,
 long prufer,
 void *sampling_args, data_workspc *data_subset,
 long *subset_original_ids)
{
	data_clustering clustering;
	medoid_set medoid;
	proclus_subspaces subspcs;

	proclus_clustering_data_alloc (&clustering, &medoid, &subspcs,
	                               data.len, max_cluster_count,
	                               max_subspace_count);

	proclus_workspc proclus;
	pair_sample pairs;

	proclus_clustering_pieces_alloc (&proclus, &pairs, data.len, data.flen,
	                                 max_cluster_count, max_subspace_count,
	                                 inter_intra_sample_size, medoid_sample_size,
	                                 0, prufer);

	if (data_subset)
	{
		proclus_clustering_workspc_init (*data_subset, proclus, initial_sample_size, features, weighted);

		if (subset_original_ids)
		{
			long medoid_id;
			long *medoid_sample = proclus.medoid.sample;
			long sample_size = proclus.medoid.sample_size;

			for (medoid_id = 0; medoid_id < sample_size; ++medoid_id, ++medoid_sample)
			{
				*medoid_sample = subset_original_ids[*medoid_sample];
			}
		}
	}

	else
	{
		proclus_clustering_workspc_init (data, proclus, initial_sample_size, features, weighted);
	}

	proclus_clustering_bootstrap_base (data, affinity_matrix, clustering,
	                                   medoid, subspcs, proclus, pairs,
	                                   cluster_sampler, subspace_sampler,
	                                   trial_max, non_improv_max,
	                                   bootstrap_max, cluster_avg_frac,
	                                   features, sampling_args);

	free (clustering.memory_block);
	free (medoid.memory_block);
	free (subspcs.memory_block);
	free (proclus.memory_block);
	free (pairs.memory_block);
}

// devise bootstrap process in order to find rates of different points
// belonging to each of preformed clusters
// ideas:
// - sample medoids from the preformed clusters
//	 taking uniformly from each cluster,
//	 and forming refined sample,
//	 with at least one member from each cluster
// - form then affinity with cluster
//   based on incidence with a medoid
//   belonging to that cluster
// - then, use those as likelihoods of
//   being in a given cluster. For bootstrap
//   purposes, for instance

#endif