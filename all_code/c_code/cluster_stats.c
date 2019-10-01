#include <cluster_stats_types.h>
#include <data_types.h>
#include <general_util.c>
#include <vector_util.c>
#include <rng_util.c>
#include <data_util.c>

#ifndef CLUSTER_STATS_C
#define CLUSTER_STATS_C

// #define CLUSTER_STATS_VERBOSE

#define MARGINS_SIZE(cluster_count, data_flen) \
    (2 * cluster_count * data_flen * sizeof (double))

#define CENTROIDS_SIZE(cluster_count, data_flen) \
    (cluster_count * data_flen * sizeof (double))

#define SAMPLE_SPACE_SIZE(data_len, cluster_count) \
    (data_len * sizeof (long) + (cluster_count + 1) * sizeof (long))

void clustering_margins_alloc (clustering_margins *margins, long cluster_count,
                               long data_flen)
{
    long memory_len = MARGINS_SIZE (cluster_count, data_flen);
    void *memory_block = malloc (memory_len);

    margins->memory_len = memory_len;
    margins->memory_block = memory_block;

    margins->count = cluster_count;
    margins->flen = data_flen;

    margins->set_up = memory_block;
    memory_block += cluster_count * data_flen * sizeof (double);

    margins->set_down = memory_block;
}

void clustering_margins_mem_load (clustering_margins *margins, double *margins_set,
                                  long cluster_count, long data_flen)
{
    margins->memory_len = 0;
    margins->memory_block = NULL;

    margins->count = cluster_count;
    margins->flen = data_flen;

    margins->set_up = margins_set;
    margins_set += cluster_count * data_flen * sizeof (double);

    margins->set_down = margins_set;
}

void clustering_centroids_alloc (clustering_centroids *centroids, long cluster_count,
                                 long data_flen)
{
    long memory_len = CENTROIDS_SIZE (cluster_count, data_flen);
    void *memory_block = malloc (memory_len);

    centroids->memory_len = memory_len;
    centroids->memory_block = memory_block;

    centroids->count = cluster_count;
    centroids->flen = data_flen;

    centroids->set = memory_block;
}

void sample_space_alloc (clustering_sample_space *sample_space, long data_len, long cluster_count)
{
    long memory_len = SAMPLE_SPACE_SIZE (data_len, cluster_count);
    void *memory_block = malloc (memory_len);

    sample_space->memory_len = memory_len;
    sample_space->memory_block = memory_block;

    sample_space->cluster_count = cluster_count;
    sample_space->cluster_ranges_size = cluster_count + 1;
    sample_space->cluster_ranges = memory_block;

    memory_block += (cluster_count + 1) * sizeof (long);

    sample_space->data_len = data_len;
    sample_space->clusters_data_ids = memory_block;
}

void sample_space_alloc_from (clustering_sample_space *sample_space, data_clustering clustering)
{
    long memory_len = SAMPLE_SPACE_SIZE (clustering.data_len, clustering.cluster_count);
    void *memory_block = malloc (memory_len);

    sample_space->memory_len = memory_len;
    sample_space->memory_block = memory_block;

    sample_space->cluster_count = clustering.cluster_count;
    sample_space->cluster_ranges_size = clustering.cluster_count + 1;
    sample_space->cluster_ranges = memory_block;

    {
        long cluster_id;
        long *cluster_ranges = memory_block;
        cluster_ranges[0] = 0;

        for (cluster_id = 1; cluster_id <= clustering.cluster_count;
                ++cluster_id)
        {
            cluster_ranges[cluster_id] = clustering.cluster_sizes[cluster_id - 1] +
                                         cluster_ranges[cluster_id - 1];
        }
    }

    memory_block += (clustering.cluster_count + 1) * sizeof (long);

    sample_space->data_len = clustering.data_len;
    sample_space->clusters_data_ids = memory_block;

    {
        long data_id;
        long *clusters_data_ids = memory_block;
        long *data_cluster_ids = clustering.data_cluster_ids;
        long *cluster_ranges = sample_space->cluster_ranges;

        long *part_sizes = calloc (clustering.cluster_count, sizeof (double));

        for (data_id = 0; data_id < clustering.data_len;
                ++data_id, ++data_cluster_ids)
        {
            long cluster_id = *data_cluster_ids;

            clusters_data_ids[cluster_ranges[cluster_id] + part_sizes[cluster_id]] = data_id;
            part_sizes[cluster_id]++;
        }
    }
}

void sample_space_set (clustering_sample_space sample_space, data_clustering clustering)
{
    {
        long cluster_id;
        long *cluster_ranges = sample_space.cluster_ranges;
        cluster_ranges[0] = 0;

        for (cluster_id = 1; cluster_id <= clustering.cluster_count;
                ++cluster_id)
        {
            cluster_ranges[cluster_id] = clustering.cluster_sizes[cluster_id - 1] +
                                         cluster_ranges[cluster_id - 1];
        }
    }

    {
        long data_id;
        long *clusters_data_ids = sample_space.clusters_data_ids;
        long *data_cluster_ids = clustering.data_cluster_ids;
        long *cluster_ranges = sample_space.cluster_ranges;

        long *part_sizes = calloc (clustering.cluster_count, sizeof (double));

        for (data_id = 0; data_id < clustering.data_len;
                ++data_id, ++data_cluster_ids)
        {
            long cluster_id = *data_cluster_ids;

            clusters_data_ids[cluster_ranges[cluster_id] + part_sizes[cluster_id]] = data_id;
            part_sizes[cluster_id]++;
        }
    }
}

// sorts each set of clusters data ids from he order given by de values over a single feature
void clustering_sample_space_feature_argsort (data_workspc data, clustering_sample_space sample_space, long feature_id)
{
    double *data_set = data.set;
    long cluster_count = sample_space.cluster_count;

    struct
    {
        long stride;
        double *vals;
    } sorting_args = {data.row_stride, data_set + feature_id * data.col_stride};

    long *clusters_data_ids = sample_space.clusters_data_ids;
    long *cluster_range = sample_space.cluster_ranges;
    long cluster_id;

    for (cluster_id = 0; cluster_id < cluster_count;
            ++cluster_range, ++cluster_id)
    {
        long cluster_size = cluster_range[1] - cluster_range[0];

        if (cluster_size < 2)
        {
            continue;
        }

        qsort_r (clusters_data_ids + cluster_range[0], cluster_size, sizeof (double),
                 double_arg_cmp, &sorting_args);
    }
}

// returns the value (or interpolation of) at a given
// percentage margin of the whole data
double clustering_cluster_argsorted_feature_quantile
(data_workspc data, clustering_sample_space sample_space,
 long cluster_id, long feature_id, double percentage)
{
    long *cluster_range = sample_space.cluster_ranges + cluster_id;
    long *clusters_data_ids = sample_space.clusters_data_ids + cluster_range[0];
    double *data_features = data.set + feature_id * data.col_stride;

    return argsorted_vec_percentile (data_features, data.row_stride, clusters_data_ids,
                                     cluster_range[1] - cluster_range[0], percentage);
}

void clustering_interquantile_margins
(data_workspc data, clustering_margins margins,
 clustering_sample_space sample_space,
 double mass, double scale_up, double scale_down)
{
    long data_flen = data.flen;
    long data_sflen = data.flen * data.col_stride;
    long data_col_stride = data.col_stride;
    double *data_set = data.set;

    long cluster_count = sample_space.cluster_count;
    long *clusters_data_ids = sample_space.clusters_data_ids;
    long *cluster_ranges = sample_space.cluster_ranges;

    double down_quantile = (1 - mass) / 2;
    double up_quantile = 1 - down_quantile;

    struct
    {
        long stride;
        double *vals;
    } sorting_args = {data.row_stride, NULL};

    {
        long *cluster_range = cluster_ranges;
        double *margin_up = margins.set_up;
        double *margin_down = margins.set_down;
        long cluster_id;

        for (cluster_id = 0; cluster_id < cluster_count;
                ++cluster_range, ++cluster_id)
        {
            long cluster_size = cluster_range[1] - cluster_range[0];

            if (!cluster_size)
            {
                continue;
            }

            if (cluster_size == 1)
            {
                vec_set_scal (margin_up, 0, data_flen, 1);
                margin_up += data_flen;
                vec_set_scal (margin_down, 0, data_flen, 1);
                margin_up += data_flen;

                continue;
            }

            long cluster_last_id = cluster_size - 1;

            long range_up_id;
            double range_up_w;

            {
                double quantile_id = up_quantile * cluster_last_id;
                range_up_id = quantile_id;
                range_up_w = range_up_id + 1 - quantile_id;
            }

            long range_down_id;
            double range_down_w;

            {
                double quantile_id = down_quantile * cluster_last_id;
                range_down_id = quantile_id;
                range_down_w = range_down_id + 1 - quantile_id;
            }

            long *cluster_data_ids = clusters_data_ids + cluster_range[0];
            sorting_args.vals = data_set;
            long feature_id;

            for (feature_id = 0; feature_id < data_sflen;
                    sorting_args.vals += data_col_stride, ++margin_up,
                    ++margin_down, feature_id += data_col_stride)
            {
                qsort_r (cluster_data_ids, cluster_size, sizeof (double),
                         double_arg_cmp, &sorting_args);

                double quantile_up = range_up_w * DATA_ID (data, cluster_data_ids[range_up_id])[feature_id] +
                                     (1 - range_up_w) * DATA_ID (data, cluster_data_ids[range_up_id + 1])
                                     [feature_id];

                double quantile_down = range_down_w * DATA_ID (data, cluster_data_ids[range_down_id])[feature_id] +
                                       (1 - range_down_w) * DATA_ID (data, cluster_data_ids[range_down_id + 1])
                                       [feature_id];

                double interquantile_range = quantile_up - quantile_down;

                *margin_up = quantile_up + scale_up * interquantile_range;
                *margin_down = quantile_down - scale_down * interquantile_range;
            }
        }
    }

    #ifdef CLUSTER_STATS_VERBOSE

    {
        printf ("\n");

        double *margin_up = margins.set_up;
        double *margin_down = margins.set_down;

        long cluster_id;

        for (cluster_id = 0; cluster_id < cluster_count;
                ++cluster_id)
        {
            long feature_id;

            for (feature_id = 0; feature_id < data_flen;
                    ++margin_up, ++feature_id)
            {
                printf ("%.2E,", *margin_up);
            }

            printf ("\n");

            for (feature_id = 0; feature_id < data_flen;
                    ++margin_down, ++feature_id)
            {
                printf ("%.2E,", *margin_down);
            }

            printf ("\n");
        }
    }

    #endif
}

void clustering_std_margins (data_workspc data, clustering_margins margins,
                             clustering_sample_space sample_space,
                             double std_scale)
{
    long data_flen = data.flen;
    double *data_set = data.set;
    long data_col_stride = data.col_stride;
    long data_row_stride = data.row_stride;

    long cluster_count = sample_space.cluster_count;
    long *clusters_data_ids = sample_space.clusters_data_ids;
    long *cluster_ranges = sample_space.cluster_ranges;

    {
        long *cluster_range = cluster_ranges;
        double *margin_up = margins.set_up;
        double *margin_down = margins.set_down;
        long cluster_id;

        for (cluster_id = 0; cluster_id < cluster_count;
                ++cluster_ranges, ++cluster_id)
        {
            long cluster_size = cluster_range[1] - cluster_range[0];

            if (!cluster_size)
            {
                continue;
            }

            long *cluster_data_ids = clusters_data_ids + cluster_range[0];
            double *data_features = data_set;
            long feature_id;

            for (feature_id = 0; feature_id < data_flen;
                    data_features += data_col_stride, ++margin_up, ++margin_down,
                    ++feature_id)
            {
                double mean;
                double mean_sqr_sum = data_vec_index_int_sum_sqr_sum (data_features, data_row_stride,
                                      cluster_data_ids, cluster_size,
                                      &mean) / cluster_size;
                mean /= cluster_size;

                double cluster_std = sqrt (mean_sqr_sum - mean * mean);
                cluster_std *= std_scale;

                *margin_up = mean + cluster_std;
                *margin_down = mean - cluster_std;
            }
        }
    }

    #ifdef CLUSTER_STATS_VERBOSE

    {
        printf ("\n");

        double *margin_up = margins.set_up;
        double *margin_down = margins.set_down;

        long cluster_id;

        for (cluster_id = 0; cluster_id < cluster_count;
                ++cluster_id)
        {
            long feature_id;

            for (feature_id = 0; feature_id < data_flen;
                    ++margin_up, ++feature_id)
            {
                printf ("%.2E,", *margin_up);
            }

            printf ("\n");

            for (feature_id = 0; feature_id < data_flen;
                    ++margin_down, ++feature_id)
            {
                printf ("%.2E,", *margin_down);
            }

            printf ("\n");
        }
    }

    #endif
}

void clustering_assym_interquantile_margins
(data_workspc data, clustering_margins margins,
 clustering_sample_space sample_space,
 double mass_up, double mass_down,
 double scale_up, double scale_down)
{
    long data_flen = data.flen;
    long data_sflen = data.flen * data.col_stride;
    long data_col_stride = data.col_stride;
    double *data_set = data.set;

    long cluster_count = sample_space.cluster_count;
    long *clusters_data_ids = sample_space.clusters_data_ids;
    long *cluster_ranges = sample_space.cluster_ranges;

    double up_quantile = 0.5 + 0.5 * mass_up;
    double down_quantile = 0.5 - 0.5 * mass_down;

    struct
    {
        long stride;
        double *vals;
    } sorting_args = {data.row_stride, NULL};

    {
        long *cluster_range = cluster_ranges;
        double *margin_up = margins.set_up;
        double *margin_down = margins.set_down;
        long cluster_id;

        for (cluster_id = 0; cluster_id < cluster_count;
                ++cluster_range, ++cluster_id)
        {
            long cluster_size = cluster_range[1] - cluster_range[0];

            if (!cluster_size)
            {
                continue;
            }

            if (cluster_size == 1)
            {
                double *cluster_data_vec = DATA_ID (data, cluster_range[0]);
                vec_set (margin_up, cluster_data_vec, data_flen, 1, data.col_stride);
                margin_up += data_flen;
                vec_set (margin_down, cluster_data_vec, data_flen, 1, data.col_stride);
                margin_down += data_flen;

                continue;
            }

            long cluster_last_id = cluster_size - 1;

            long range_up_id;
            double range_up_w;

            {
                double quantile_id = up_quantile * cluster_last_id;
                range_up_id = quantile_id;
                range_up_w = range_up_id + 1 - quantile_id;
            }

            long range_down_id;
            double range_down_w;

            {
                double quantile_id = down_quantile * cluster_last_id;
                range_down_id = quantile_id;
                range_down_w = range_down_id + 1 - quantile_id;
            }

            long range_mid_id;
            double range_mid_w;

            {
                double quantile_id = 0.5 * cluster_last_id;
                range_mid_id = quantile_id;
                range_mid_w = range_mid_id + 1 - quantile_id;
            }

            long *cluster_data_ids = clusters_data_ids + cluster_range[0];
            sorting_args.vals = data_set;
            long feature_id;

            for (feature_id = 0; feature_id < data_sflen;
                    sorting_args.vals += data_col_stride, ++margin_up,
                    ++margin_down, feature_id += data_col_stride)
            {
                qsort_r (cluster_data_ids, cluster_size, sizeof (double),
                         double_arg_cmp, &sorting_args);

                double quantile_up = range_up_w * DATA_ID (data, cluster_data_ids[range_up_id])[feature_id] +
                                     (1 - range_up_w) * DATA_ID (data, cluster_data_ids[range_up_id + 1])
                                     [feature_id];

                double quantile_down = range_down_w * DATA_ID (data, cluster_data_ids[range_down_id])[feature_id] +
                                       (1 - range_down_w) * DATA_ID (data, cluster_data_ids[range_down_id + 1])
                                       [feature_id];

                double median = range_mid_w * DATA_ID (data, cluster_data_ids[range_mid_id])[feature_id] +
                                (1 - range_mid_w) * DATA_ID (data, cluster_data_ids[range_mid_id + 1])
                                [feature_id];

                double interquantile_range = quantile_up - quantile_down;

                *margin_up = quantile_up + scale_up * min (interquantile_range, 2 * (quantile_up - median));

                *margin_down = quantile_down - scale_down * min (interquantile_range, 2 * (median - quantile_down));
            }
        }
    }

    #ifdef CLUSTER_STATS_VERBOSE

    {
        printf ("\n");

        double *margin_up = margins.set_up;
        double *margin_down = margins.set_down;

        long cluster_id;

        for (cluster_id = 0; cluster_id < cluster_count;
                ++cluster_id)
        {
            long feature_id;

            for (feature_id = 0; feature_id < data_flen;
                    ++margin_up, ++feature_id)
            {
                printf ("%.2E,", *margin_up);
            }

            printf ("\n");

            for (feature_id = 0; feature_id < data_flen;
                    ++margin_down, ++feature_id)
            {
                printf ("%.2E,", *margin_down);
            }

            printf ("\n");
        }
    }

    #endif
}

// margins based on range between quantiles, above and below median
// scale then is the multiple of such range added to the outer most quantiles considered
// in fact, the minimum (of the increment) of all possibilities
// (inter quantile, asymmetric* and inter inter quantile) is taken
// *asymmetric and inter quantile are taken relative to the quantile
// between the masses in each side (i.e. (up + sub) / 2 and (down + sub) / 2)
void clustering_assym_interinterquantile_margins
(data_workspc data, clustering_margins margins,
 clustering_sample_space sample_space,
 double mass_up, double mass_down,
 double sub_mass_up, double sub_mass_down,
 double scale_up, double scale_down)
{
    long data_flen = data.flen;
    long data_sflen = data.flen * data.col_stride;
    long data_col_stride = data.col_stride;
    double *data_set = data.set;

    long cluster_count = sample_space.cluster_count;
    long *clusters_data_ids = sample_space.clusters_data_ids;
    long *cluster_ranges = sample_space.cluster_ranges;


    double up_quantile = 0.5 + 0.5 * mass_up;
    double sub_up_quantile = 0.5 + 0.5 * sub_mass_up;
    double mid_up_quantile = (up_quantile + sub_up_quantile) / 2;

    double down_quantile = 0.5 - 0.5 * mass_down;
    double sub_down_quantile = 0.5 - 0.5 * sub_mass_down;
    double mid_down_quantile = (down_quantile + sub_down_quantile) / 2;

    double mid_mass_up = (mass_up + sub_mass_up) / 2;
    double inter_up_prop = mid_mass_up / (mass_up - sub_mass_up);

    double mid_mass_down = (mass_down + sub_mass_down) / 2;
    double inter_down_prop = mid_mass_down / (mass_down - sub_mass_down);

    struct
    {
        long stride;
        double *vals;
    } sorting_args = {data.row_stride, NULL};

    {
        long *cluster_range = cluster_ranges;
        double *margin_up = margins.set_up;
        double *margin_down = margins.set_down;
        long cluster_id;

        for (cluster_id = 0; cluster_id < cluster_count;
                ++cluster_range, ++cluster_id)
        {
            long cluster_size = cluster_range[1] - cluster_range[0];

            if (!cluster_size)
            {
                continue;
            }

            if (cluster_size == 1)
            {
                double *cluster_data_vec = DATA_ID (data, cluster_range[0]);
                vec_set (margin_up, cluster_data_vec, data_flen, 1, data.col_stride);
                margin_up += data_flen;
                vec_set (margin_down, cluster_data_vec, data_flen, 1, data.col_stride);
                margin_down += data_flen;

                continue;
            }

            long cluster_last_id = cluster_size - 1;

            long range_up_id;
            double range_up_w;

            {
                double quantile_id = up_quantile * cluster_last_id;
                range_up_id = quantile_id;
                range_up_w = range_up_id + 1 - quantile_id;
            }

            long range_sub_up_id;
            double range_sub_up_w;

            {
                double quantile_id = sub_up_quantile * cluster_last_id;
                range_sub_up_id = quantile_id;
                range_sub_up_w = range_sub_up_id + 1 - quantile_id;
            }

            long range_mid_up_id;
            double range_mid_up_w;

            {
                double quantile_id = mid_up_quantile * cluster_last_id;
                range_mid_up_id = quantile_id;
                range_mid_up_w = range_mid_up_id + 1 - quantile_id;
            }

            long range_down_id;
            double range_down_w;

            {
                double quantile_id = down_quantile * cluster_last_id;
                range_down_id = quantile_id;
                range_down_w = range_down_id + 1 - quantile_id;
            }

            long range_sub_down_id;
            double range_sub_down_w;

            {
                double quantile_id = sub_down_quantile * cluster_last_id;
                range_sub_down_id = quantile_id;
                range_sub_down_w = range_sub_down_id + 1 - quantile_id;
            }

            long range_mid_down_id;
            double range_mid_down_w;

            {
                double quantile_id = mid_down_quantile * cluster_last_id;
                range_mid_down_id = quantile_id;
                range_mid_down_w = range_mid_down_id + 1 - quantile_id;
            }

            long range_mid_id;
            double range_mid_w;

            {
                double quantile_id = 0.5 * cluster_last_id;
                range_mid_id = quantile_id;
                range_mid_w = range_mid_id + 1 - quantile_id;
            }

            long *cluster_data_ids = clusters_data_ids + cluster_range[0];
            sorting_args.vals = data_set;
            long feature_id;

            // continue to change from here
            for (feature_id = 0; feature_id < data_sflen;
                    sorting_args.vals += data_col_stride, ++margin_up,
                    ++margin_down, feature_id += data_col_stride)
            {
                qsort_r (cluster_data_ids, cluster_size, sizeof (double),
                         double_arg_cmp, &sorting_args);

                double quantile_up = range_up_w * DATA_ID (data, cluster_data_ids[range_up_id])[feature_id] +
                                     (1 - range_up_w) * DATA_ID (data, cluster_data_ids[range_up_id + 1])
                                     [feature_id];

                double quantile_sub_up = range_sub_up_w * DATA_ID (data, cluster_data_ids[range_sub_up_id])[feature_id] +
                                         (1 - range_sub_up_w) * DATA_ID (data, cluster_data_ids[range_sub_up_id + 1])
                                         [feature_id];

                double quantile_mid_up = range_mid_up_w * DATA_ID (data, cluster_data_ids[range_mid_up_id])[feature_id] +
                                         (1 - range_mid_up_w) * DATA_ID (data, cluster_data_ids[range_mid_up_id + 1])
                                         [feature_id];

                double quantile_down = range_down_w * DATA_ID (data, cluster_data_ids[range_down_id])[feature_id] +
                                       (1 - range_down_w) * DATA_ID (data, cluster_data_ids[range_down_id + 1])
                                       [feature_id];

                double quantile_sub_down = range_sub_down_w * DATA_ID (data, cluster_data_ids[range_sub_down_id])[feature_id] +
                                           (1 - range_sub_down_w) * DATA_ID (data, cluster_data_ids[range_sub_down_id + 1])
                                           [feature_id];

                double quantile_mid_down = range_mid_down_w * DATA_ID (data, cluster_data_ids[range_mid_down_id])[feature_id] +
                                           (1 - range_mid_down_w) * DATA_ID (data, cluster_data_ids[range_mid_down_id + 1])
                                           [feature_id];

                double median = range_mid_w * DATA_ID (data, cluster_data_ids[range_mid_id])[feature_id] +
                                (1 - range_mid_w) * DATA_ID (data, cluster_data_ids[range_mid_id + 1])
                                [feature_id];

                double interquantile_range = quantile_mid_up - quantile_mid_down;

                double inter_up_range = min (quantile_mid_up - median, inter_up_prop * (quantile_up - quantile_sub_up));
                *margin_up = quantile_mid_up + scale_up * min (interquantile_range, 2 * inter_up_range);

                double inter_down_range = min (median - quantile_mid_down, inter_down_prop * (quantile_sub_down - quantile_down));
                *margin_down = quantile_mid_down - scale_down * min (interquantile_range, 2 * inter_down_range);
            }
        }
    }

    #ifdef CLUSTER_STATS_VERBOSE

    {
        printf ("\n");

        double *margin_up = margins.set_up;
        double *margin_down = margins.set_down;

        long cluster_id;

        for (cluster_id = 0; cluster_id < cluster_count;
                ++cluster_id)
        {
            long feature_id;

            for (feature_id = 0; feature_id < data_flen;
                    ++margin_up, ++feature_id)
            {
                printf ("%.2E,", *margin_up);
            }

            printf ("\n");

            for (feature_id = 0; feature_id < data_flen;
                    ++margin_down, ++feature_id)
            {
                printf ("%.2E,", *margin_down);
            }

            printf ("\n");
        }
    }

    #endif
}

void clustering_assym_bounded_interinterquantile_margins
(data_workspc data, clustering_margins margins,
 clustering_sample_space sample_space,
 double mass_up, double mass_down,
 double sub_mass_up, double sub_mass_down,
 double scale_up, double scale_down,
 double bound_up, double bound_down)
{
    long data_flen = data.flen;
    long data_sflen = data.flen * data.col_stride;
    long data_col_stride = data.col_stride;
    double *data_set = data.set;

    long cluster_count = sample_space.cluster_count;
    long *clusters_data_ids = sample_space.clusters_data_ids;
    long *cluster_ranges = sample_space.cluster_ranges;

    double up_quantile = 0.5 + 0.5 * mass_up;
    double sub_up_quantile = 0.5 + 0.5 * sub_mass_up;
    double mid_up_quantile = (up_quantile + sub_up_quantile) / 2;

    double down_quantile = 0.5 - 0.5 * mass_down;
    double sub_down_quantile = 0.5 - 0.5 * sub_mass_down;
    double mid_down_quantile = (down_quantile + sub_down_quantile) / 2;

    double mid_mass_up = (mass_up + sub_mass_up) / 2;
    double inter_up_prop = mid_mass_up / (mass_up - sub_mass_up);

    double mid_mass_down = (mass_down + sub_mass_down) / 2;
    double inter_down_prop = mid_mass_down / (mass_down - sub_mass_down);

    struct
    {
        long stride;
        double *vals;
    } sorting_args = {data.row_stride, NULL};

    {
        long *cluster_range = cluster_ranges;
        double *margin_up = margins.set_up;
        double *margin_down = margins.set_down;
        long cluster_id;

        for (cluster_id = 0; cluster_id < cluster_count;
                ++cluster_range, ++cluster_id)
        {
            long cluster_size = cluster_range[1] - cluster_range[0];

            if (!cluster_size)
            {
                continue;
            }

            if (cluster_size == 1)
            {
                double *cluster_data_vec = DATA_ID (data, cluster_range[0]);
                vec_set (margin_up, cluster_data_vec, data_flen, 1, data.col_stride);
                margin_up += data_flen;
                vec_set (margin_down, cluster_data_vec, data_flen, 1, data.col_stride);
                margin_down += data_flen;

                continue;
            }

            long *cluster_data_ids = clusters_data_ids + cluster_range[0];
            sorting_args.vals = data_set;
            long feature_id;

            for (feature_id = 0; feature_id < data_sflen;
                    sorting_args.vals += data_col_stride, ++margin_up,
                    ++margin_down, feature_id += data_col_stride)
            {
                qsort_r (cluster_data_ids, cluster_size, sizeof (double),
                         double_arg_cmp, &sorting_args);

                long *bounded_data_ids;
                long bounded_count = argsorted_vec_within_bounds (sorting_args.vals, sorting_args.stride,
                                     cluster_data_ids, cluster_size, &bounded_data_ids,
                                     bound_up, bound_down, 0);

                if (bounded_count <= 1)
                {
                    *margin_up = bound_up;
                    *margin_down = bound_down;

                    continue;
                }

                double quantile_up = argsorted_vec_percentile (sorting_args.vals, sorting_args.stride, bounded_data_ids, bounded_count, up_quantile);

                double quantile_sub_up =  argsorted_vec_percentile (sorting_args.vals, sorting_args.stride, bounded_data_ids, bounded_count, sub_up_quantile);

                double quantile_mid_up =  argsorted_vec_percentile (sorting_args.vals, sorting_args.stride, bounded_data_ids, bounded_count, mid_up_quantile);

                double quantile_down =  argsorted_vec_percentile (sorting_args.vals, sorting_args.stride, bounded_data_ids, bounded_count, down_quantile);

                double quantile_sub_down =  argsorted_vec_percentile (sorting_args.vals, sorting_args.stride, bounded_data_ids, bounded_count, sub_down_quantile);

                double quantile_mid_down =  argsorted_vec_percentile (sorting_args.vals, sorting_args.stride, bounded_data_ids, bounded_count, mid_down_quantile);

                double median =  argsorted_vec_percentile (sorting_args.vals, sorting_args.stride, bounded_data_ids, bounded_count, .5);
                double interquantile_range = quantile_mid_up - quantile_mid_down;

                double inter_up_range = min (quantile_mid_up - median, inter_up_prop * (quantile_up - quantile_sub_up));
                *margin_up = min (quantile_mid_up + scale_up * min (interquantile_range, 2 * inter_up_range), bound_up);

                double inter_down_range = min (median - quantile_mid_down, inter_down_prop * (quantile_sub_down - quantile_down));
                *margin_down = max (quantile_mid_down - scale_down * min (interquantile_range, 2 * inter_down_range), bound_down);
            }
        }
    }

    #ifdef CLUSTER_STATS_VERBOSE

    {
        printf ("\n");

        double *margin_up = margins.set_up;
        double *margin_down = margins.set_down;

        long cluster_id;

        for (cluster_id = 0; cluster_id < cluster_count;
                ++cluster_id)
        {
            long feature_id;

            for (feature_id = 0; feature_id < data_flen;
                    ++margin_up, ++feature_id)
            {
                printf ("%.2E,", *margin_up);
            }

            printf ("\n");

            for (feature_id = 0; feature_id < data_flen;
                    ++margin_down, ++feature_id)
            {
                printf ("%.2E,", *margin_down);
            }

            printf ("\n");
        }
    }

    #endif
}

void data_assym_bounded_interinterquantile_margins
(data_workspc data, clustering_margins margins,
 long *data_ids, long data_count,
 double mass_up, double mass_down,
 double sub_mass_up, double sub_mass_down,
 double scale_up, double scale_down,
 double bound_up, double bound_down,
 long set_ids)
{
    if (set_ids)
    {
        long data_id;

        for (data_id = 0; data_id < data_count; ++data_id)
        {
            data_ids[data_id] = data_id;
        }
    }

    long data_sflen = data.flen * data.col_stride;
    long data_col_stride = data.col_stride;
    double *data_set = data.set;

    double up_quantile = 0.5 + 0.5 * mass_up;
    double sub_up_quantile = 0.5 + 0.5 * sub_mass_up;
    double mid_up_quantile = (up_quantile + sub_up_quantile) / 2;

    double down_quantile = 0.5 - 0.5 * mass_down;
    double sub_down_quantile = 0.5 - 0.5 * sub_mass_down;
    double mid_down_quantile = (down_quantile + sub_down_quantile) / 2;

    double mid_mass_up = (mass_up + sub_mass_up) / 2;
    double inter_up_prop = mid_mass_up / (mass_up - sub_mass_up);

    double mid_mass_down = (mass_down + sub_mass_down) / 2;
    double inter_down_prop = mid_mass_down / (mass_down - sub_mass_down);


    {
        double *margin_up = margins.set_up;
        double *margin_down = margins.set_down;

        struct
        {
            long stride;
            double *vals;
        } sorting_args = {data.row_stride, data_set};

        long feature_id;

        for (feature_id = 0; feature_id < data_sflen;
                sorting_args.vals += data_col_stride, ++margin_up,
                ++margin_down, feature_id += data_col_stride)
        {
            qsort_r (data_ids, data_count, sizeof (double),
                     double_arg_cmp, &sorting_args);

            long *bounded_data_ids;
            long bounded_count = argsorted_vec_within_bounds (sorting_args.vals, sorting_args.stride,
                                 data_ids, data_count, &bounded_data_ids,
                                 bound_up, bound_down, 0);

            if (bounded_count <= 1)
            {
                *margin_up = bound_up;
                *margin_down = bound_down;

                continue;
            }

            double quantile_up = argsorted_vec_percentile (sorting_args.vals, sorting_args.stride, bounded_data_ids, bounded_count, up_quantile);

            double quantile_sub_up =  argsorted_vec_percentile (sorting_args.vals, sorting_args.stride, bounded_data_ids, bounded_count, sub_up_quantile);

            double quantile_mid_up =  argsorted_vec_percentile (sorting_args.vals, sorting_args.stride, bounded_data_ids, bounded_count, mid_up_quantile);

            double quantile_down =  argsorted_vec_percentile (sorting_args.vals, sorting_args.stride, bounded_data_ids, bounded_count, down_quantile);

            double quantile_sub_down =  argsorted_vec_percentile (sorting_args.vals, sorting_args.stride, bounded_data_ids, bounded_count, sub_down_quantile);

            double quantile_mid_down =  argsorted_vec_percentile (sorting_args.vals, sorting_args.stride, bounded_data_ids, bounded_count, mid_down_quantile);

            double median =  argsorted_vec_percentile (sorting_args.vals, sorting_args.stride, bounded_data_ids, bounded_count, .5);
            double interquantile_range = quantile_mid_up - quantile_mid_down;

            double inter_up_range = min (quantile_mid_up - median, inter_up_prop * (quantile_up - quantile_sub_up));
            *margin_up = min (quantile_mid_up + scale_up * min (interquantile_range, 2 * inter_up_range), bound_up);

            double inter_down_range = min (median - quantile_mid_down, inter_down_prop * (quantile_sub_down - quantile_down));
            *margin_down = max (quantile_mid_down - scale_down * min (interquantile_range, 2 * inter_down_range), bound_down);
        }
    }

    #ifdef CLUSTER_STATS_VERBOSE

    {
        printf ("\n");

        double *margin_up = margins.set_up;
        double *margin_down = margins.set_down;

        long cluster_id;

        for (cluster_id = 0; cluster_id < cluster_count;
                ++cluster_id)
        {
            long feature_id;

            for (feature_id = 0; feature_id < data_flen;
                    ++margin_up, ++feature_id)
            {
                printf ("%.2E,", *margin_up);
            }

            printf ("\n");

            for (feature_id = 0; feature_id < data_flen;
                    ++margin_down, ++feature_id)
            {
                printf ("%.2E,", *margin_down);
            }

            printf ("\n");
        }
    }

    #endif
}

void clustering_percentile_centroids (data_workspc data, clustering_centroids centroids,
                                      clustering_sample_space sample_space,
                                      double pth_quantile)
{
    long data_sflen = data.flen * data.col_stride;
    long data_col_stride = data.col_stride;
    double *data_set = data.set;

    long cluster_count = sample_space.cluster_count;
    long *clusters_data_ids = sample_space.clusters_data_ids;
    long *cluster_ranges = sample_space.cluster_ranges;

    struct
    {
        long stride;
        double *vals;
    } sorting_args = {data.row_stride, NULL};

    {
        long *cluster_range = cluster_ranges;
        double *centroid = centroids.set;
        long cluster_id;

        for (cluster_id = 0; cluster_id < cluster_count;
                ++cluster_range, ++cluster_id)
        {
            long cluster_size = cluster_range[1] - cluster_range[0];

            if (!cluster_size)
            {
                continue;
            }

            long pth_id;
            double pth_w;

            {
                double quantile_id = pth_quantile *
                                     (cluster_size - 1);
                pth_id = quantile_id;
                pth_w = pth_id + 1 - quantile_id;
            }

            long *cluster_data_ids = clusters_data_ids + cluster_range[0];
            sorting_args.vals = data_set;
            long feature_id;

            for (feature_id = 0; feature_id < data_sflen;
                    sorting_args.vals += data_col_stride, ++centroid,
                    feature_id += data_col_stride)
            {
                qsort_r (cluster_data_ids, cluster_size, sizeof (double),
                         double_arg_cmp, &sorting_args);

                *centroid = pth_w * DATA_ID (data, cluster_data_ids[pth_id])[feature_id] +
                            (1 - pth_w) * DATA_ID (data, cluster_data_ids[pth_id + 1])[feature_id];

            }
        }
    }

    #ifdef CLUSTER_STATS_VERBOSE

    {
        printf ("\n");

        double *centroid = centroids.set;

        long cluster_id;

        for (cluster_id = 0; cluster_id < cluster_count;
                ++cluster_id)
        {
            long feature_id;

            for (feature_id = 0; feature_id < data_flen;
                    ++centroid, ++feature_id)
            {
                printf ("%.2E,", *centroid);
            }

            printf ("\n");
        }
    }

    #endif
}

void clustering_mean_centroids (data_workspc data, clustering_centroids centroids,
                                clustering_sample_space sample_space)
{
    long data_flen = data.flen;
    long data_col_stride = data.col_stride;
    long data_row_stride = data.row_stride;
    double *data_set = data.set;

    long cluster_count = sample_space.cluster_count;
    long *clusters_data_ids = sample_space.clusters_data_ids;
    long *cluster_ranges = sample_space.cluster_ranges;

    {
        long *cluster_range = cluster_ranges;
        double *centroid = centroids.set;
        long cluster_id;

        for (cluster_id = 0; cluster_id < cluster_count;
                ++cluster_range, ++cluster_id)
        {
            long cluster_size = cluster_range[1] - cluster_range[0];

            if (!cluster_size)
            {
                continue;
            }

            long *cluster_data_ids = clusters_data_ids + cluster_range[0];
            double *data_features = data_set;
            long feature_id;

            for (feature_id = 0; feature_id < data_flen;
                    ++centroid, data_features += data_col_stride,
                    ++feature_id)
            {
                *centroid = data_vec_index_int_sum (data_features, data_row_stride,
                                                    cluster_data_ids, cluster_size) / cluster_size;
            }
        }
    }

    #ifdef CLUSTER_STATS_VERBOSE

    {
        printf ("\n");

        double *centroid = centroids.set;

        long cluster_id;

        for (cluster_id = 0; cluster_id < cluster_count;
                ++cluster_id)
        {
            long feature_id;

            for (feature_id = 0; feature_id < data_flen;
                    ++centroid, ++feature_id)
            {
                printf ("%.2E,", *centroid);
            }

            printf ("\n");
        }
    }

    #endif
}

#endif