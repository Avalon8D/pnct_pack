# the easy wrappings for useful functions
# remember to always init rand
# from os import sys

from numba import jit, vectorize
from numpy import ceil, dot
from scipy.sparse.linalg import eigsh

from all_code.python_interface_types import *

c_cluster_lib.clustering_normalize.restype = c_long
c_cluster_lib.sorted_vec_percentile.restype = c_double
c_cluster_lib.init_rand(POINTER(c_uint)(c_uint(1793414758)))


class BASE_CLUSTERING_ALGOS():
    def proclus_pieces_alloc(
            data_len, data_flen, cluster_count, subspace_count,
            inter_intra_sample_size, medoid_sample_size,
            least_cluster_size, prufer=1
    ):
        c_proclus = PROCLUS_WORKSPC()
        c_pairs = PAIR_SAMPLE()

        c_cluster_lib.proclus_clustering_pieces_alloc(
            byref(c_proclus), byref(c_pairs), data_len, data_flen,
            cluster_count, subspace_count, inter_intra_sample_size,
            medoid_sample_size, least_cluster_size, prufer
        )

        return c_proclus, c_pairs

    def proclus_data_alloc(data_len, cluster_count, subspace_count):
        c_clustering = DATA_CLUSTERING()
        c_medoid = MEDOID_SET()
        c_subspcs = PROCLUS_SUBSPACES()

        c_cluster_lib.proclus_clustering_data_alloc(
            byref(c_clustering), byref(c_medoid), byref(c_subspcs),
            data_len, cluster_count, subspace_count
        )

        return c_clustering, c_medoid, c_subspcs

    def proclus_workspc_init(
            c_data, c_proclus, initial_sample_size,
            c_features_p=NULL_FEATURE_RANGE,
            weighted=1
    ):

        c_cluster_lib.proclus_clustering_workspc_init(
            c_data, c_proclus, initial_sample_size, c_features_p, weighted
        )

    # assumes all workspaces have been initialized, and merely executes the trials

    def proclus_base(
            c_data, c_clustering, c_medoid, c_subspcs,
            c_proclus, c_pairs, trial_max, non_improv_max,
            c_features_p=NULL_FEATURE_RANGE
    ):

        c_cluster_lib.proclus_clustering_base(
            c_data, c_clustering, c_medoid, c_subspcs,
            c_proclus, c_pairs, trial_max, non_improv_max,
            c_features_p
        )

    # if cluster_count and subspace_count are both given, 
    # then every data, but c_data is ignored and new are allocated
    # the returned data will be the allocated data in that case
    # that is, providing the counts makes the function allocate
    # the clustering data for the run

    def proclus_one_shot(
            c_data, c_clustering, c_medoid, c_subspcs,
            trial_max, non_improv_max, inter_intra_sample_size,
            medoid_sample_size, initial_sample_size, least_cluster_size,
            c_features_p=NULL_FEATURE_RANGE, weighted=1, prufer=1,
            cluster_count=None, subspace_count=None
    ):

        if cluster_count is not None and subspace_count is not None:
            c_clustering, c_medoid, c_subspcs = BASE_CLUSTERING_ALGOS.proclus_data_alloc(
                c_data.len, cluster_count, subspace_count
            )

        c_cluster_lib.proclus_clustering_one_shot(
            c_data, c_clustering, c_medoid, c_subspcs,
            trial_max, non_improv_max, inter_intra_sample_size,
            medoid_sample_size, initial_sample_size, least_cluster_size,
            c_features_p, weighted, prufer
        )

        return c_clustering, c_medoid, c_subspcs

    # assumes all workspaces have been initialized, and merely executes the trials
    # assumes c_affinity_matrix to be c_data.len * c_data.len in double word length
    # C contiguous memory block

    def proclus_bootstrap_base(
            c_data, c_affinity_matrix, c_clustering,
            c_medoid, c_subspcs, c_proclus, c_pairs,
            c_cluster_sampler, c_subspace_sampler,
            trial_max, non_improv_max, bootstrap_max,
            cluster_avg_frac,
            c_features_p=NULL_FEATURE_RANGE,
            sampling_args=NULL_void):

        c_cluster_lib.proclus_clustering_bootstrap_base(
            c_data, c_affinity_matrix, c_clustering,
            c_medoid, c_subspcs, c_proclus, c_pairs,
            c_cluster_sampler, c_subspace_sampler,
            c_long(trial_max), c_long(non_improv_max),
            c_long(bootstrap_max),
            c_double(cluster_avg_frac),
            c_features_p, sampling_args
        )

    # if affinity_matrix is None, allocates a numpy array off appropriate shape
    # in that case, this array is returned by the function, otherwise, the same provided
    # if affinity_matrix is provided, it is assumed to be a numpy array of appropriate shape
    # which then is converted to a C pointer
    # does not check array type, i.e. converts blindly
    # if affinity_matrix is not contiguous, it is reallocated

    def proclus_bootstrap(
            c_data, max_cluster_count, max_subspace_count,
            c_cluster_sampler, c_subspace_sampler,
            trial_max, non_improv_max,
            bootstrap_max, cluster_avg_frac,
            inter_intra_sample_size, medoid_sample_size,
            initial_sample_size, affinity_matrix=None,
            c_features_p=NULL_FEATURE_RANGE, weighted=1,
            prufer=1,
            sampling_args=NULL_void, c_data_subset_p=NULL_DATA_WORKSPC,
            c_subset_original_ids=NULL_long
    ):

        if affinity_matrix is None:
            affinity_matrix = numpy.empty((c_data.len, c_data.len), dtype=float)

        c_affinity_matrix, affinity_matrix = C_NUMPY_UTIL.keep_if_C_behaved(affinity_matrix)
        affinity_matrix[...] = 0

        c_cluster_lib.proclus_clustering_bootstrap(
            c_data, c_affinity_matrix,
            c_long(max_cluster_count), c_long(max_subspace_count),
            c_cluster_sampler, c_subspace_sampler,
            c_long(trial_max), c_long(non_improv_max),
            c_long(bootstrap_max), c_double(cluster_avg_frac),
            c_long(inter_intra_sample_size), c_long(medoid_sample_size),
            c_long(initial_sample_size),
            c_features_p, c_long(weighted), c_long(prufer),
            sampling_args, c_data_subset_p, c_subset_original_ids
        )

        return affinity_matrix

    def kmeans_items_alloc(c_data, cluster_count):
        data_len, data_flen = c_data.len, c_data.flen
        c_clustering = DATA_CLUSTERING.alloc(data_len, cluster_count)
        c_centroids = CLUSTERING_CENTROIDS.alloc(cluster_count, data_flen)

        dist_numpy_buffer = numpy.empty(c_clustering.cluster_count, dtype=c_double)
        dist_buffer = dist_numpy_buffer.ctypes.data_as(c_double_p)
        c_cluster_lib.centroids_fast_distw_data_sample(
            c_data, c_centroids, c_centroids.count,
            c_clustering.data_cluster_ids, dist_buffer
        )

        del dist_numpy_buffer

        return c_clustering, c_centroids

    # assumes centroids already sampled
    def kmeans_intd_iteration(c_data, c_clustering, c_centroids, trial_max):
        c_cluster_lib.kmeans_iteration(c_data, c_clustering, c_centroids, trial_max)

    def kmeans_iteration(c_data, cluster_count, trial_max):
        data_len, data_flen = c_data.len, c_data.flen
        c_clustering, c_centroids = BASE_CLUSTERING_ALGOS.kmeans_items_alloc(c_data, cluster_count)

        c_cluster_lib.kmeans_iteration(c_data, c_clustering, c_centroids, trial_max)

        return c_clustering, c_centroids

    # assumes centroids already sampled
    # eliminates clusters smaller than small_fraction * data_len, 
    # perhaps diminishing original cluster_count on c_clustering
    def kmeans_initd_reassign_small(c_data, c_clustering, c_centroids, trial_max, small_fraction):
        c_cluster_lib.kmeans_iteration(c_data, c_clustering, c_centroids, trial_max)
        c_cluster_lib.kmeans_reassign_small(c_data, c_clustering, c_centroids, c_double(small_fraction))

        c_clustering.cluster_count = c_cluster_lib.clustering_normalize(
            c_clustering.data_cluster_ids,
            c_clustering.cluster_sizes,
            c_clustering.data_len,
            c_clustering.cluster_count
        )

        c_centroids.count = c_clustering.cluster_count
        c_cluster_lib.centroid_cluster_mean_eval(c_data, c_clustering, c_centroids)

    def kmeans_reassign_small(c_data, cluster_count, trial_max, small_fraction):
        data_len, data_flen = c_data.len, c_data.flen
        c_clustering, c_centroids = BASE_CLUSTERING_ALGOS.kmeans_items_alloc(c_data, cluster_count)

        BASE_CLUSTERING_ALGOS.kmeans_initd_reassign_small(
            c_data, c_clustering, c_centroids, trial_max, small_fraction
        )

        return c_clustering, c_centroids


class AFFINITY_ALGOS():

    @jit
    def affinity_mat_lambda(affinity_matrix):
        return lambda i, j: affinity_matrix[i, j]

    @jit
    def n_affinity_intr_dim(n_affinity_matrix):
        return int(ceil(n_affinity_matrix.trace()))

    def affinity_mat_laplacian(affinity_matrix, inplace=True):
        data_len = affinity_matrix.shape[0]

        if inplace:
            c_laplacian_matrix, laplacian_matrix = C_NUMPY_UTIL.keep_if_C_behaved(affinity_matrix)

        else:
            laplacian_matrix = affinity_matrix.copy()
            c_laplacian_matrix = laplacian_matrix.ctypes.data_as(c_double_p)

        rows_inorm_buffer = numpy.empty(data_len, dtype=c_double)
        c_rows_inorm_buffer = rows_inorm_buffer.ctypes.data_as(c_double_p)
        c_cluster_lib.spectral_clustering_form_laplacian_inplace(
            c_laplacian_matrix, data_len,
            c_rows_inorm_buffer
        )

        del rows_inorm_buffer

        return laplacian_matrix

    def normalized_affinity_matrix(affinity_matrix, inplace=True):
        data_len = affinity_matrix.shape[0]

        if inplace:
            c_n_affinity_matrix, n_affinity_matrix = C_NUMPY_UTIL.keep_if_C_behaved(affinity_matrix)

        else:
            n_affinity_matrix = affinity_matrix.copy()
            c_n_affinity_matrix = n_affinity_matrix.ctypes.data_as(c_double_p)

        rows_inorm_buffer = numpy.empty(data_len, dtype=c_double)
        c_rows_inorm_buffer = rows_inorm_buffer.ctypes.data_as(c_double_p)
        c_cluster_lib.normalized_affinity_matrix_inplace(
            c_n_affinity_matrix, data_len, c_rows_inorm_buffer
        )

        del rows_inorm_buffer

        return n_affinity_matrix

    def affinity_spectral_decomp(laplacian_matrix):
        data_len = laplacian_matrix.shape[0]
        eigen_vectors = numpy.ascontiguousarray(laplacian_matrix, dtype=float)
        eigen_values = numpy.empty(data_len)

        c_eigen_vectors = eigen_vectors.ctypes.data_as(c_double_p)
        c_eigen_values = eigen_values.ctypes.data_as(c_double_p)

        c_cluster_lib.SPD_spectral_decomp(c_eigen_vectors, c_eigen_values, eigen_vectors.shape[0])

        return eigen_vectors, eigen_values

    def affinity_laplacian_spectral_decomp(affinity_matrix):
        eigen_vectors, affinity_matrix = AFFINITY_ALGOS.affinity_mat_laplacian(affinity_matrix)
        data_len = affinity_matrix.shape[0]
        eigen_values = numpy.empty(data_len)

        c_eigen_vectors = eigen_vectors.ctypes.data_as(c_double_p)
        c_eigen_values = eigen_values.ctypes.data_as(c_double_p)

        c_cluster_lib.SPD_spectral_decomp(c_eigen_vectors, c_eigen_values, eigen_vectors.shape[0])

        return eigen_vectors, eigen_values

    # if small thresh is not None, kmeans with reassign small clusters is ran, 
    # with small_cluster_thresh as threshold
    @jit
    def affinity_laplacian_eigen_kmeans(
            affinity_matrix, trial_max,
            cluster_count=None, small_cluster_thresh=None,
            inplace=True, maxiter_lanc=500, max_rank=-1
    ):
        n_affinity_matrix = AFFINITY_ALGOS.normalized_affinity_matrix(affinity_matrix, inplace)

        intr_dim = AFFINITY_ALGOS.n_affinity_intr_dim(n_affinity_matrix)
        intr_dim = intr_dim if max_rank < 0 else min(intr_dim, max_rank)
        eigen_values, eigen_vectors = eigsh(
            n_affinity_matrix, k=intr_dim * 3,
            which='LM', maxiter=maxiter_lanc
        )
        data_len = eigen_vectors.shape[1]

        c_eigen_data = DATA_WORKSPC.create_as_mask_of_matrix(eigen_vectors[:, -intr_dim - 1:-1])
        c_cluster_lib.mat_normalize2_rows(
            c_eigen_data.set, c_eigen_data.len, c_eigen_data.flen,
            c_eigen_data.row_stride, c_eigen_data.col_stride, 0
        )

        if cluster_count is None:
            cluster_count = int(c_eigen_data.flen)

        if small_cluster_thresh is None:
            c_clustering, c_centroids = BASE_CLUSTERING_ALGOS.kmeans_iteration(
                c_eigen_data, cluster_count, trial_max
            )

        else:
            c_clustering, c_centroids = BASE_CLUSTERING_ALGOS.kmeans_reassign_small(
                c_eigen_data, cluster_count, trial_max,
                float(small_cluster_thresh)
            )

        c_centroids.kill_memory_block()
        # del eigen_vectors

        return c_clustering, eigen_values


class CLUSTER_STATS():
    def margins_items_alloc(c_clustering, data_flen):
        c_margins = CLUSTERING_MARGINS.alloc(c_clustering.cluster_count, data_flen)
        c_sample_space = CLUSTERING_SAMPLE_SPACE.alloc_from_clustering(c_clustering)

        return c_margins, c_sample_space

    # assumes initialized data
    def margins_initd_base_assym_interquantile(
            c_data, c_margins, c_sample_space, mass_up, mass_down, scale_up, scale_down
    ):
        c_cluster_lib.clustering_assym_interquantile_margins(
            c_data, c_margins, c_sample_space,
            c_double(mass_up), c_double(mass_down),
            c_double(scale_up), c_double(scale_down)
        )

    # black box margins
    def margins_assym_interquantile(
            c_data, c_clustering, mass_up, mass_down, scale_up, scale_down
    ):
        c_margins, c_sample_space = CLUSTER_STATS.margins_items_alloc(c_clustering, c_data.flen)

        c_cluster_lib.clustering_assym_interquantile_margins(
            c_data, c_margins, c_sample_space,
            c_double(mass_up), c_double(mass_down),
            c_double(scale_up), c_double(scale_down)
        )

        return c_margins, c_sample_space

    def margins_initd_base_assym_interinterquantile(
            c_data, c_margins, c_sample_space, mass_up, mass_down,
            sub_mass_up, sub_mass_down, scale_up, scale_down
    ):
        c_cluster_lib.clustering_assym_interinterquantile_margins(
            c_data, c_margins, c_sample_space,
            c_double(mass_up), c_double(mass_down),
            c_double(sub_mass_up), c_double(sub_mass_down),
            c_double(scale_up), c_double(scale_down)
        )

    def margins_assym_interinterquantile(
            c_data, c_clustering, mass_up, mass_down,
            sub_mass_up, sub_mass_down, scale_up, scale_down
    ):
        c_margins, c_sample_space = CLUSTER_STATS.margins_items_alloc(c_clustering, c_data.flen)

        c_cluster_lib.clustering_assym_interinterquantile_margins(
            c_data, c_margins, c_sample_space,
            c_double(mass_up), c_double(mass_down),
            c_double(sub_mass_up), c_double(sub_mass_down),
            c_double(scale_up), c_double(scale_down)
        )

        return c_margins, c_sample_space

    def margins_initd_base_assym_bounded_interinterquantile(
            c_data, c_margins, c_sample_space, mass_up, mass_down,
            sub_mass_up, sub_mass_down, scale_up, scale_down,
            bound_up, bound_down
    ):
        c_cluster_lib.clustering_assym_bounded_interinterquantile_margins(
            c_data, c_margins, c_sample_space,
            c_double(mass_up), c_double(mass_down),
            c_double(sub_mass_up), c_double(sub_mass_down),
            c_double(scale_up), c_double(scale_down),
            c_double(bound_up), c_double(bound_down)
        )

    def margins_assym_bounded_interinterquantile(
            c_data, c_clustering, mass_up, mass_down,
            sub_mass_up, sub_mass_down, scale_up, scale_down,
            bound_up, bound_down
    ):
        c_margins, c_sample_space = CLUSTER_STATS.margins_items_alloc(c_clustering, c_data.flen)

        c_cluster_lib.clustering_assym_bounded_interinterquantile_margins(
            c_data, c_margins, c_sample_space,
            c_double(mass_up), c_double(mass_down),
            c_double(sub_mass_up), c_double(sub_mass_down),
            c_double(scale_up), c_double(scale_down),
            c_double(bound_up), c_double(bound_down)
        )

        return c_margins, c_sample_space

    def data_margins_initd_assym_bounded_interinterquantile(
            c_data, c_margins, c_data_ids, data_count,
            mass_up, mass_down, sub_mass_up, sub_mass_down,
            scale_up, scale_down, bound_up, bound_down,
            set_ids=1
    ):

        c_cluster_lib.data_assym_bounded_interinterquantile_margins(
            c_data, c_margins, c_data_ids, c_long(data_count),
            c_double(mass_up), c_double(mass_down),
            c_double(sub_mass_up), c_double(sub_mass_down),
            c_double(scale_up), c_double(scale_down),
            c_double(bound_up), c_double(bound_down),
            c_long(set_ids)
        )

    def data_margins_assym_bounded_interinterquantile(
            c_data, mass_up, mass_down, sub_mass_up, sub_mass_down,
            scale_up, scale_down, bound_up, bound_down,
            data_ids_vector=None, set_ids=1
    ):

        data_count = c_data.len

        if data_ids_vector is None:
            data_ids_vector = numpy.empty(c_data.len, dtype=c_long)
        else:
            data_count = data_ids_vector.shape[0]

        c_data_ids = data_ids_vector.ctypes.data_as(c_long_p)
        c_margins = CLUSTERING_MARGINS.alloc(1, c_data.flen)

        CLUSTER_STATS.data_margins_initd_assym_bounded_interinterquantile(
            c_data, c_margins, c_data_ids, data_count,
            mass_up, mass_down, sub_mass_up, sub_mass_down,
            scale_up, scale_down, bound_up, bound_down
        )

        return c_margins

        # if clustering_vector is None, it is assumed that there is only 1 set of margins

    # and the data_matrix is globally classified under it
    # one means below low margin, two means above, three means both, zero, none
    # that is, in signed mode
    # otherwise, when nonzero is true, 0 indicates a bin beyond a margin, 1 otherwise
    nonzero_bin_class = vectorize()(jit()(
        lambda data_bin, up_bin, down_bin: 0 if (data_bin > up_bin) or (data_bin < down_bin) else 1)
    )
    signed_bin_class = vectorize()(jit()(
        lambda data_bin, up_bin, down_bin: (2 if data_bin < down_bin else 1)
        if data_bin > up_bin
        else (-1 if data_bin < down_bin else 0)
    ))

    @jit
    def beyond_margins_flags(data_matrix, margins_matrix, clustering_vector=None, nonzero=False):
        flags_matrix = numpy.empty(data_matrix.shape)

        margins_up = margins_matrix[0]
        margins_down = margins_matrix[1]

        bin_class = CLUSTER_STATS.nonzero_bin_class if nonzero else CLUSTER_STATS.signed_bin_class

        if clustering_vector is None:
            data_margin_up = margins_up[0]
            data_margin_down = margins_down[0]

            for data_line, flags_line in zip(data_matrix[:], flags_matrix[:]):
                bin_class(data_line, data_margin_up, data_margin_down, out=flags_line)

        else:

            for data_cluster_id, data_line, flags_line in zip(
                    clustering_vector, data_matrix[:], flags_matrix[:]
            ):
                data_margin_up = margins_up[data_cluster_id]
                data_margin_down = margins_down[data_cluster_id]
                bin_class(data_line, data_margin_up, data_margin_down, out=flags_line)

        return flags_matrix

    @jit
    def weighted_point_flags(flags_matrix, point_flags_weights):
        data_len = flags_matrix.shape[0]

        point_flags_vector = numpy.empty(data_len)

        for data_id, flags_row in enumerate(flags_matrix[:]):
            dot(point_flags_weights, flags_row, out=point_flags_vector[data_id])

        return point_flags_vector

    # if clustering_vector is None, it is assumed that there is only 1 set of margins
    # and the data_matrix is globally classified under it
    # one means below low margin, two means above, three means both, zero, none
    # that is, in signed mode
    # otherwise, when nonzero is true, 0 indicates a bin beyond a margin, 1 otherwise
    @jit
    def beyond_margins_point_flags(
            data_matrix, margins_matrix, point_flags_weights,
            clustering_vector=None, nonzero=False
    ):
        point_flags_vector = numpy.empty(data_matrix.shape[0])
        flags_line_buffer = numpy.empty(data_matrix.shape[1])

        margins_up = margins_matrix[0]
        margins_down = margins_matrix[1]

        bin_class = CLUSTER_STATS.nonzero_bin_class if nonzero else CLUSTER_STATS.signed_bin_class

        if clustering_vector is None:
            data_margin_up = margins_up[0]
            data_margin_down = margins_down[0]

            for point_flag, data_line in zip(
                    point_flags_vector.reshape(-1, 1)[:], data_matrix[:]
            ):
                point_flag = bin_class(
                    data_line, data_margin_up,
                    data_margin_down,
                    out=flags_line_buffer
                ).sum(axis=-1, out=point_flag)


        else:
            for point_flag, data_line, data_cluster_id in zip(
                    point_flags_vector.reshape(-1, 1)[:],
                    data_matrix[:], clustering_vector
            ):
                data_margin_up = margins_up[data_cluster_id]
                data_margin_down = margins_down[data_cluster_id]

                bin_class(data_line, data_margin_up, data_margin_down, out=flags_line_buffer)
                flags_line_buffer *= point_flags_weights
                flags_line_buffer.sum(axis=-1, out=point_flag)

        return point_flags_vector

    def centroids_items_alloc(c_clustering, data_flen):
        c_centroids = CLUSTERING_CENTROIDS.alloc(c_clustering.cluster_count, data_flen)
        c_sample_space = CLUSTERING_SAMPLE_SPACE.alloc_from_clustering(c_clustering)

        return c_centroids, c_sample_space

    def centroids_initd_base_percentile(c_data, c_clustering, c_centroids, c_sample_space, percentile):
        c_cluster_lib.clustering_percentile_centroids(
            c_data, c_clustering, c_centroids, c_sample_space, c_double(percentile)
        )

    def centroids_percentile(c_data, c_clustering, percentile):
        c_centroids, c_sample_space = CLUSTER_STATS.centroids_items_alloc(c_clustering, c_data.flen)

        c_cluster_lib.clustering_percentile_centroids(
            c_data, c_centroids, c_sample_space, c_double(percentile)
        )

        return c_centroids, c_sample_space
