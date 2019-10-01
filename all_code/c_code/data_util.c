#include <general_util.c>
#include <rng_util.c>
#include <vector_util.c>

#ifndef DATA_UTIL_C
#define DATA_UTIL_C

// #define DATA_UTIL_VERBOSE

#define FLAG_SIZE(data_len, data_flen) \
    (data_len * data_flen * sizeof (double) + \
     data_len * sizeof (double))

#define DATA_SIZE(data_len, data_flen) \
    (data_len * data_flen * sizeof (double))

#define CLUSTERING_SIZE(data_len, cluster_count) \
    (data_len * sizeof (long) + cluster_count * sizeof (long))

#define FEATURE_RANGE_SIZE(feature_len) \
    (feature_len * sizeof (long))

#define MEDOID_SAMPLE_SIZE(medoid_count) \
    (medoid_count * sizeof (long))

void flag_data_alloc (flag_data *flags, long data_len, long data_flen)
{
    long memory_len = FLAG_SIZE (data_len, data_flen);
    void *memory_block = malloc (memory_len);

    flags->memory_len = memory_len;
    flags->memory_block = memory_block;

    flags->len = data_len;
    flags->point_set = memory_block;
    memory_block += data_len * sizeof (double);

    flags->flen = data_flen;
    flags->set = memory_block;
}

void flag_data_mem_load (flag_data *flags, double *flags_set, double *point_set,
                         long data_len, long data_flen)
{
    flags->memory_len = 0;
    flags->memory_block = NULL;

    flags->len = data_len;
    flags->point_set = point_set;

    flags->flen = data_flen;
    flags->set = flags_set;
}

void data_workspc_alloc (data_workspc *data, long data_len, long data_flen)
{
    long memory_len = DATA_SIZE (data_len, data_flen);
    void *memory_block = malloc (memory_len);

    data->memory_len = memory_len;
    data->memory_block = memory_block;

    data->len = data_len;
    data->flen = data_flen;
    data->set = memory_block;
    data->row_stride = data_flen;
    data->col_stride = 1;
}

void data_workspc_mem_load (data_workspc *data, double *data_set,
                            long data_len, long data_flen,
                            long row_stride, long col_stride)
{
    data->memory_len = 0;
    data->memory_block = NULL;

    data->len = data_len;
    data->flen = data_flen;
    data->set = data_set;
    data->row_stride = row_stride;
    data->col_stride = col_stride;
}

void data_clustering_alloc (data_clustering *clustering, long data_len,
                            long cluster_count)
{
    long memory_len = CLUSTERING_SIZE (data_len, cluster_count);
    void *memory_block = malloc (memory_len);

    clustering->memory_len = memory_len;
    clustering->memory_block = memory_block;

    clustering->data_len = data_len;
    clustering->data_cluster_ids = memory_block;
    memory_block += data_len * sizeof (long);

    clustering->cluster_count = cluster_count;
    clustering->cluster_sizes = memory_block;
}

void data_clustering_mem_load (data_clustering *clustering, long *data_cluster_ids, long data_len,
                               long *cluster_sizes, long cluster_count)
{
    clustering->memory_len = 0;
    clustering->memory_block = NULL;

    clustering->data_len = data_len;
    clustering->data_cluster_ids = data_cluster_ids;

    clustering->cluster_count = cluster_count;
    clustering->cluster_sizes = cluster_sizes;
}

void data_clustering_set_data_len (data_clustering clustering, long data_len)
{
    clustering.data_len = data_len;

    long *data_cluster_id = clustering.data_cluster_ids;
    long *cluster_sizes = clustering.cluster_sizes;
    vec_long_set_scal (cluster_sizes, 0, clustering.cluster_count, 1);

    long data_id;

    for (data_id = 0; data_id < data_len;
            ++data_cluster_id, ++data_id)
    {
        cluster_sizes[*data_cluster_id]++;
    }
}

void data_feature_range_alloc (data_feature_range *features, long feature_len)
{
    long memory_len = FEATURE_RANGE_SIZE (feature_len);
    void *memory_block = malloc (memory_len);

    features->memory_len = memory_len;
    features->memory_block = memory_block;

    features->len = feature_len;
    features->set = memory_block;
}

void data_feature_range_mem_load (data_feature_range *features, long *features_set, long feature_len)
{
    features->memory_len = 0;
    features->memory_block = NULL;

    features->len = feature_len;
    features->set = features_set;
}

void gen_rand_clustering (data_clustering clustering)
{
    long cluster_count = clustering.cluster_count;
    long data_len = clustering.data_len;

    long *cluster_sizes = clustering.cluster_sizes;
    memset (cluster_sizes, 0, cluster_count * sizeof (long));

    long *data_cluster_id = clustering.data_cluster_ids;
    long data_id;

    for (data_id = 0; data_id < data_len;
            ++data_id, ++data_cluster_id)
    {
        long cluster_id = gsl_rng_uniform_int (rand_gen, cluster_count);

        *data_cluster_id = cluster_id;
        cluster_sizes[cluster_id]++;
    }

    #ifdef DATA_UTIL_VERBOSE

    {
        printf ("\n");
        printf ("%ld,\n", cluster_count);

        long *cluster_size = clustering.cluster_sizes;
        long cluster_id;

        for (cluster_id = 0; cluster_id < cluster_count;
                ++cluster_id, ++cluster_size)
        {
            printf ("%ld,", *cluster_size);
        }

        printf ("\n");
    }

    {
        printf ("\n");
        printf ("%ld,\n", data_len);

        long *data_cluster_id = clustering.data_cluster_ids;
        long data_id;

        for (data_id = 0; data_id < data_len;
                ++data_id, ++data_cluster_id)
        {
            printf ("%ld,", *data_cluster_id);
        }

        printf ("\n");
    }

    #endif
}

void gen_rand_data (data_workspc data, double scale,
                    double offset)
{
    long data_len = data.len;
    long data_flen = data.flen;

    double *data_point = data.set;
    long row_id;

    for (row_id = 0; row_id < data_len;
            ++row_id)
    {
        long col_id;

        for (col_id = 0; col_id < data_flen;
                ++col_id, ++data_point)
        {
            *data_point = scale * gsl_ran_ugaussian (rand_gen) + offset;
        }
    }

    #ifdef DATA_UTIL_VERBOSE

    {
        printf ("\n");

        printf ("%ld, %ld,\n", data.len, data.flen);

        double *data_point = data.set;
        long row_id;

        for (row_id = 0; row_id < data_len;
                ++row_id)
        {
            long col_id;

            for (col_id = 0; col_id < data_flen;
                    ++col_id, ++data_point)
            {
                printf ("%.2E,", *data_point);
            }

            printf ("\n");
        }

        printf ("\n");
    }

    #endif
}

#endif