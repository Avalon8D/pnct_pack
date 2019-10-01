#include <data_types.h>
#include <proclus_types.h>
#include <general_util.c>
#include <vector_util.c>
#include <rng_util.c>
#include <data_util.c>

#ifndef PROCLUS_COMPONENTS_CLUSTERING_C
#define PROCLUS_COMPONENTS_CLUSTERING_C

// #define PROCLUS_COMPONENTS_CLUSTERING_VERBOSE

void data_clustering_medoid_subspaces_scores
(data_workspc data, medoid_set medoid,
 proclus_buffer_workspc buffer,
 data_feature_range *features)
{
    long medoid_count = medoid.count;
    double *range = buffer.range;
    long *medoid_ids = medoid.ids;

    if (features)
    {
        long features_len = features->len;
        long *features_set = features->set;
        long id_a;

        for (id_a = 0; id_a < medoid_count; ++id_a)
        {
            range[id_a] = DBL_MAX;
        }

        for (id_a = 0; id_a < medoid_count; ++id_a)
        {
            long id_b;

            for (id_b = id_a + 1; id_b < medoid_count; ++id_b)
            {
                double dist = vec_fdist1 (DATA_ID (data, medoid_ids[id_a]),
                                          DATA_ID (data, medoid_ids[id_b]),
                                          features_set, features_len,
                                          data.col_stride, data.col_stride);

                range[id_a] = min (dist, range[id_a]);
                range[id_b] = min (dist, range[id_b]);
            }
        }
    }

    else
    {
        long id_a;

        for (id_a = 0; id_a < medoid_count; ++id_a)
        {
            range[id_a] = DBL_MAX;
        }

        for (id_a = 0; id_a < medoid_count; ++id_a)
        {
            long id_b;

            for (id_b = id_a + 1; id_b < medoid_count; ++id_b)
            {
                double dist = vec_dist1 (DATA_ID (data, medoid_ids[id_a]),
                                         DATA_ID (data, medoid_ids[id_b]),
                                         data.flen, data.col_stride,
                                         data.col_stride);

                range[id_a] = min (dist, range[id_a]);
                range[id_b] = min (dist, range[id_b]);
            }
        }
    }

    double *subspace_scores = buffer.subspace_scores;
    double *cluster_sizes = buffer.cluster_sizes;
    double *point = buffer.point;
    memset (subspace_scores, 0, medoid_count * data.flen * sizeof (double));
    memset (cluster_sizes, 0, medoid_count * sizeof (double));

    if (features)
    {
        long features_len = features->len;
        long *features_set = features->set;

        long data_id;
        double *data_row = data.set;

        for (data_id = 0; data_id < data.len;
                ++data_id, data_row += data.row_stride)
        {
            long medoid_id;
            double *subspace_scores_row = subspace_scores;

            for (medoid_id = 0; medoid_id < medoid_count;
                    ++medoid_id, subspace_scores_row += data.flen)
            {
                vec_abs_dev (data_row, DATA_ID (data, medoid_ids[medoid_id]),
                             point, data.flen, data.col_stride, data.col_stride);

                if (data_vec_index_int_sum (point, 1, features_set, features_len) < range[medoid_id])
                {
                    data_vec_index_int_add (subspace_scores_row, 1, point, 1,
                                            features_set, features_len);

                    cluster_sizes[medoid_id]++;
                }
            }
        }
    }

    else
    {
        long data_id;
        double *data_row = data.set;

        for (data_id = 0; data_id < data.len;
                ++data_id, data_row += data.row_stride)
        {
            long medoid_id;
            double *subspace_scores_row = subspace_scores;

            for (medoid_id = 0; medoid_id < medoid_count;
                    ++medoid_id, subspace_scores_row += data.flen)
            {
                vec_abs_dev (data_row, DATA_ID (data, medoid_ids[medoid_id]),
                             point, data.flen, data.col_stride, data.col_stride);

                if (vec_sum (point, data.flen, 1) < range[medoid_id])
                {
                    vec_add (subspace_scores_row, point,
                             data.flen, 1, 1);

                    cluster_sizes[medoid_id]++;
                }
            }
        }
    }

    if (features)
    {
        long features_len = features->len;
        long *features_set = features->set;

        long id;
        long range_id;

        for (range_id = 0, id = 0; range_id < features_len;
                ++features_set, ++range_id, ++id)
        {
            long current_feature = *features_set;

            for (; id < current_feature; ++id)
            {
                vec_set_scal (subspace_scores + id, DBL_MAX,
                              medoid_count, data.flen);
            }

            vec_div (subspace_scores + id, cluster_sizes,
                     medoid_count, data.flen, 1);
        }

        for (; id < data.flen; ++id)
        {
            vec_set_scal (subspace_scores + id, DBL_MAX,
                          medoid_count, data.flen);
        }
    }

    else
    {
        long id;

        for (id = 0; id < data.flen; ++id)
        {
            vec_div (subspace_scores + id, cluster_sizes,
                     medoid_count, data.flen, 1);
        }
    }

    #ifdef PROCLUS_COMPONENTS_CLUSTERING_VERBOSE

    {
        printf ("\n");

        long row_id;

        for (row_id = 0; row_id < medoid_count; ++row_id)
        {
            long col_id;

            for (col_id = 0; col_id < data.flen;
                    ++col_id, ++subspace_scores)
            {
                printf ("%.2E,", *subspace_scores);
            }

            printf ("\n");
        }
    }

    #endif
}

void data_clustering_subspaces_scores
(data_workspc data, data_clustering clustering, medoid_set medoid,
 proclus_buffer_workspc buffer, data_feature_range *features)
{
    long medoid_count = medoid.count;
    double *range = buffer.range;
    long *medoid_ids = medoid.ids;

    if (features)
    {
        long features_len = features->len;
        long *features_set = features->set;

        long id_a;

        for (id_a = 0; id_a < medoid_count; ++id_a)
        {
            range[id_a] = DBL_MAX;
        }

        for (id_a = 0; id_a < medoid_count; ++id_a)
        {
            long id_b;

            for (id_b = id_a + 1; id_b < medoid_count; ++id_b)
            {
                double dist = vec_fdist1 (DATA_ID (data, medoid_ids[id_a]),
                                          DATA_ID (data, medoid_ids[id_b]),
                                          features_set, features_len,
                                          data.col_stride, data.col_stride);

                range[id_a] = min (dist, range[id_a]);
                range[id_b] = min (dist, range[id_b]);
            }
        }
    }

    else
    {
        long id_a;

        for (id_a = 0; id_a < medoid_count; ++id_a)
        {
            range[id_a] = DBL_MAX;
        }

        for (id_a = 0; id_a < medoid_count; ++id_a)
        {
            long id_b;

            for (id_b = id_a + 1; id_b < medoid_count; ++id_b)
            {
                double dist = vec_dist1 (DATA_ID (data, medoid_ids[id_a]),
                                         DATA_ID (data, medoid_ids[id_b]),
                                         data.flen, data.col_stride,
                                         data.col_stride);

                range[id_a] = min (dist, range[id_a]);
                range[id_b] = min (dist, range[id_b]);
            }
        }
    }

    long *data_cluster_ids = clustering.data_cluster_ids;

    double *point = buffer.point;
    double *cluster_sizes = buffer.cluster_sizes;
    double *subspace_scores = buffer.subspace_scores;
    memset (subspace_scores, 0, medoid_count * data.flen * sizeof (double));

    if (features)
    {
        long features_len = features->len;
        long *features_set = features->set;

        long data_id;
        double *data_row = data.set;

        for (data_id = 0; data_id < data.len;
                ++data_id, data_row += data.row_stride)
        {
            long medoid_id = data_cluster_ids[data_id];
            double *subspace_scores_row = subspace_scores +
                                          medoid_id * data.flen;

            vec_abs_dev (data_row, DATA_ID (data, medoid_ids[medoid_id]),
                         point, data.flen, data.col_stride, data.col_stride);

            if (data_vec_index_int_sum (point, 1, features_set, features_len) < range[medoid_id])
            {
                data_vec_index_int_add (subspace_scores_row, 1, point, 1,
                                        features_set, features_len);

                cluster_sizes[medoid_id]++;
            }
        }
    }

    else
    {
        long data_id;
        double *data_row = data.set;

        for (data_id = 0; data_id < data.len;
                ++data_id, data_row += data.row_stride)
        {
            long medoid_id = data_cluster_ids[data_id];
            double *subspace_scores_row = subspace_scores +
                                          medoid_id * data.flen;

            vec_abs_dev (data_row, DATA_ID (data, medoid_ids[medoid_id]),
                         point, data.flen, data.col_stride, data.col_stride);

            vec_add (subspace_scores_row, point,
                     data.flen, 1, 1);

            cluster_sizes[medoid_id]++;
        }
    }

    if (features)
    {
        long id;
        long range_id;

        long features_len = features->len;
        long *features_set = features->set;

        for (range_id = 0, id = 0; range_id < features_len;
                ++features_set, ++range_id, ++id)
        {
            long current_feature = *features_set;

            for (; id < current_feature; ++id)
            {
                vec_set_scal (subspace_scores + id, DBL_MAX,
                              medoid_count, data.flen);
            }

            vec_div (subspace_scores + id, cluster_sizes,
                     medoid_count, data.flen, 1);
        }

        for (; id < data.flen; ++id)
        {
            vec_set_scal (subspace_scores + id, DBL_MAX,
                          medoid_count, data.flen);
        }
    }

    else
    {
        long id;

        for (id = 0; id < data.flen; ++id)
        {
            vec_div (subspace_scores + id, cluster_sizes,
                     medoid_count, data.flen, 1);
        }
    }

    #ifdef PROCLUS_COMPONENTS_CLUSTERING_VERBOSE

    {
        printf ("\n");

        long row_id;

        for (row_id = 0; row_id < medoid_count; ++row_id)
        {
            long col_id;

            for (col_id = 0; col_id < data.flen;
                    ++col_id, ++subspace_scores)
            {
                printf ("%.2E,", *subspace_scores);
            }

            printf ("\n");
        }
    }

    #endif
}

void data_clustering_subspaces (proclus_subspaces subspcs, proclus_buffer_workspc buffer)
{
    long data_flen = buffer.data_flen;
    long medoid_count = subspcs.cluster_count;
    size_t subspace_count = buffer.subspace_count;
    double *subspace_scores = buffer.subspace_scores;

    long *subspace_ranges = subspcs.subspace_ranges;
    long *medoid_subspaces = subspcs.subspaces;

    size_t *subspace_arg = buffer.subspace_arg;
    memset (medoid_subspaces, 0, subspace_count * sizeof (long));


    {
        long medoid_id;
        double *subspace_scores_row = subspace_scores;
        size_t *subspace_arg_row = subspace_arg;

        for (medoid_id = 0; medoid_id < medoid_count;
                ++medoid_id, subspace_scores_row += data_flen,
                subspace_arg_row += 2)
        {
            gsl_sort_smallest_index (subspace_arg_row, 2,
                                     subspace_scores_row,
                                     1, data_flen);

            subspace_scores_row[subspace_arg_row[0]] = DBL_MAX;
            subspace_scores_row[subspace_arg_row[1]] = DBL_MAX;

            subspace_count -= 2;
        }
    }

    if (!subspace_count)
    {
        {
            long medoid_id;
            long *subspace_range = subspace_ranges;
            subspace_ranges[0] = 0;
            size_t *subspace_arg_row = subspace_arg;
            long *medoid_subspaces_row = medoid_subspaces;

            for (medoid_id = 0; medoid_id < medoid_count;
                    ++medoid_id, ++subspace_range, subspace_arg_row += 2,
                    medoid_subspaces_row += 2)
            {
                medoid_subspaces_row[0] = subspace_arg_row[0];
                medoid_subspaces_row[1] = subspace_arg_row[1];
                subspace_range[1] = subspace_range[0] + 2;
            }
        }

        #ifdef PROCLUS_COMPONENTS_CLUSTERING_VERBOSE

        {
            printf ("\n");

            long range_id;
            long *subspace_range = subspace_ranges;

            for (range_id = 0; range_id < medoid_count;
                    ++range_id, ++subspace_range)
            {
                long range_beg = subspace_range[0];
                long range_len = subspace_range[1] - subspace_range[0];

                printf ("%ld, %ld\n", range_beg, range_len);

                long *medoid_subspaces_row = medoid_subspaces + range_beg;
                long subspace_id;

                for (subspace_id = 0; subspace_id < range_len;
                        ++subspace_id, ++medoid_subspaces_row)
                {
                    printf ("%ld,", *medoid_subspaces_row);
                }

                printf ("\n");
            }
        }

        #endif

        return;
    }

    {
        long medoid_id;
        size_t *subspace_arg_row = subspace_arg;

        for (medoid_id = 0; medoid_id < medoid_count;
                ++medoid_id, subspace_arg_row += 2)
        {
            long abs_incr = medoid_id * data_flen;
            subspace_arg_row[0] += abs_incr;
            subspace_arg_row[1] += abs_incr;
        }
    }

    {
        gsl_sort_smallest_index (subspace_arg + 2 * medoid_count, subspace_count,
                                 subspace_scores, 1, medoid_count * data_flen);

        subspace_count += 2 * medoid_count;
        qsort_r (subspace_arg, subspace_count, sizeof (size_t), long_quo_rem_cmp, & (data_flen));
    }

    {
        size_t arg_id;
        long *medoid_subspaces_row = medoid_subspaces;
        size_t *subspace_arg_row = subspace_arg;

        long *subspace_range = subspace_ranges;
        long max_subspace_medoid = -1;

        for (arg_id = 0; arg_id < subspace_count;
                ++arg_id, ++subspace_arg_row,
                ++medoid_subspaces_row)
        {
            long subspace_id = *subspace_arg_row;
            long subspace_medoid = subspace_id / data_flen;

            *medoid_subspaces_row = subspace_id - data_flen * subspace_medoid;

            if (subspace_medoid > max_subspace_medoid)
            {
                *subspace_range = arg_id;
                ++subspace_range;
                max_subspace_medoid = subspace_medoid;
            }
        }

        *subspace_range = subspace_count;
    }

    #ifdef PROCLUS_COMPONENTS_CLUSTERING_VERBOSE

    {
        printf ("\n");

        long range_id;
        long *subspace_range = subspace_ranges;

        for (range_id = 0; range_id < medoid_count;
                ++range_id, ++subspace_range)
        {
            long range_beg = subspace_range[0];
            long range_len = subspace_range[1] - subspace_range[0];

            printf ("%ld, %ld\n", range_beg, range_len);

            long *medoid_subspaces_row = medoid_subspaces + range_beg;
            long subspace_id;

            for (subspace_id = 0; subspace_id < range_len;
                    ++subspace_id, ++medoid_subspaces_row)
            {
                printf ("%ld,", *medoid_subspaces_row);
            }

            printf ("\n");
        }
    }

    #endif
}

void data_clustering_cluster_ids
(data_workspc data, data_clustering clustering,
 medoid_set medoid, proclus_subspaces subspcs)
{
    long *data_cluster_ids = clustering.data_cluster_ids;
    long *subspace_ranges = subspcs.subspace_ranges;
    long *medoid_subspaces = subspcs.subspaces;

    long medoid_count = clustering.cluster_count;
    long *medoid_ids = medoid.ids;
    long *cluster_sizes = clustering.cluster_sizes;
    memset (cluster_sizes, 0, medoid_count * sizeof (double));

    {
        long data_id;
        double *data_row = data.set;

        for (data_id = 0; data_id < data.len;
                ++data_id, data_row += data.row_stride)
        {
            // guarantees that every cluster has size 1, at least
            // necessary in case distances between any medoids are zero
            {
                long *curr_medoid = medoid_ids;
                long is_medoid = -1;
                long medoid_id;

                for (medoid_id = 0; medoid_id < medoid_count;
                        ++medoid_id, ++curr_medoid)
                {
                    is_medoid = (*curr_medoid == data_id ? medoid_id : is_medoid);
                }

                if (is_medoid >= 0)
                {
                    data_cluster_ids[data_id] = is_medoid;
                    cluster_sizes[is_medoid]++;
                    continue;
                }
            }

            long closest_medoid = 0;
            double closest_dist = DBL_MAX;
            long *subspace_range = subspace_ranges;
            long medoid_id;

            for (medoid_id = 0; medoid_id < medoid_count;
                    ++medoid_id, ++subspace_range)
            {
                long range_beg = subspace_range[0];
                long range_len = subspace_range[1] - subspace_range[0];
                double dist_to_medoid = vec_fdist1 (DATA_ID (data, medoid_ids[medoid_id]),
                                                    data_row, medoid_subspaces + range_beg,
                                                    range_len, data.col_stride, data.col_stride) / range_len;

                long smaller = dist_to_medoid < closest_dist;
                closest_dist = (smaller ? dist_to_medoid : closest_dist);
                closest_medoid = (smaller ? medoid_id : closest_medoid);
            }

            data_cluster_ids[data_id] = closest_medoid;
            cluster_sizes[closest_medoid]++;
        }
    }

    #ifdef PROCLUS_COMPONENTS_CLUSTERING_VERBOSE

    {
        printf ("\n");

        long medoid_id;

        for (medoid_id = 0; medoid_id < medoid_count;
                ++medoid_id, ++cluster_sizes)
        {
            printf ("%ld,", *cluster_sizes);
        }

        printf ("\n");
    }

    {
        printf ("\n");

        long data_id;

        for (data_id = 0; data_id < data.len;
                ++data_id, ++data_cluster_ids)
        {
            printf ("%ld,", *data_cluster_ids);
        }

        printf ("\n");
    }

    #endif
}

// Does not reorder subspaces! Recommended when, afterwards, one re-evaluates the subspaces
long data_clustering_medoid_normalize (data_clustering clustering, medoid_set medoid, double small_fraction)
{
    long small_thresh = small_fraction * clustering.data_len;
    small_thresh = max (1, small_thresh);
    small_thresh = min (small_thresh, clustering.data_len);

    long *cluster_size = clustering.cluster_sizes;
    long *medoid_ids = medoid.ids;

    long cluster_count = clustering.cluster_count;
    long good_cluster_count = 0;
    long medoid_id = 0;

    do
    {
        for (; medoid_id < cluster_count && *cluster_size <= small_thresh;
                ++cluster_size, ++medoid_id);

        if (medoid_id < cluster_count)
        {
            long swap_id = medoid_ids[good_cluster_count];
            medoid_ids[good_cluster_count] = medoid_ids[medoid_id];
            medoid_ids[medoid_id] = swap_id;

            ++medoid_id;
            ++cluster_size;
            ++good_cluster_count;
        }

        else
        {
            break;
        }
    }
    while (1);

    return good_cluster_count;
}

double data_clustering_dist1_seg_score
(data_workspc data, data_clustering clustering,
 medoid_set medoid, proclus_subspaces subspcs)
{
    long *data_cluster_ids = clustering.data_cluster_ids;
    long *subspace_ranges = subspcs.subspace_ranges;
    long *medoid_subspaces = subspcs.subspaces;

    long *medoid_ids = medoid.ids;

    long data_id;
    double *data_row = data.set;

    double seg_score = 0;

    for (data_id = 0; data_id < data.len;
            ++data_id, data_row += data.row_stride)
    {
        long data_medoid = data_cluster_ids[data_id];

        long range_beg = subspace_ranges[data_medoid];
        long range_len = subspace_ranges[data_medoid + 1] - subspace_ranges[data_medoid];
        seg_score += vec_fdist1 (DATA_ID (data, medoid_ids[data_medoid]),
                                 data_row, medoid_subspaces + range_beg,
                                 range_len, data.col_stride, data.col_stride) / range_len;
    }

    return seg_score / data.len;
}

double data_clustering_interintra_dist1_seg_score
(data_workspc data, data_clustering clustering,
 proclus_subspaces subspcs, pair_sample pairs)
{
    long *data_cluster_ids = clustering.data_cluster_ids;
    long *subspace_ranges = subspcs.subspace_ranges;
    long *medoid_subspaces = subspcs.subspaces;

    long pair_id;
    long *pair = pairs.sample;

    double inter_seg_score = 0;
    double intra_seg_score = 0;

    double inter_count = 0;
    double intra_count = 0;

    for (pair_id = 0; pair_id < pairs.count;
            ++pair_id, pair += 2)
    {
        long range_beg = subspace_ranges[data_cluster_ids[pair[0]]];
        long range_len = subspace_ranges[data_cluster_ids[pair[0]] + 1] - subspace_ranges[data_cluster_ids[pair[0]]];

        double pair_dist = vec_fdist1 (DATA_ID (data, pair[0]), DATA_ID (data, pair[1]),
                                       medoid_subspaces + range_beg, range_len, data.col_stride,
                                       data.col_stride);

        long same_cluster = (data_cluster_ids[pair[0]] == data_cluster_ids[pair[1]] ? 1 : 0);

        intra_seg_score += (same_cluster ? pair_dist : 0);
        intra_count += same_cluster;

        inter_seg_score += (same_cluster ? 0 : pair_dist);
        inter_count += (same_cluster ? 0 : 1);
    }

    #ifdef PROCLUS_COMPONENTS_CLUSTERING_VERBOSE

    {
        printf ("\n");

        printf ("%ld,%.2E,", (long) intra_count, intra_seg_score);
        printf ("%ld,%.2E,", (long) inter_count, inter_seg_score);
        printf ("%.2E,\n", (intra_seg_score / intra_count) /
                (inter_seg_score / inter_count));
    }


    #endif

    return (intra_seg_score / intra_count) /
           (inter_seg_score / inter_count);
}

#endif
