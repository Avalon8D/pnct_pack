#include <data_types.h>
#include <cluster_stats_types.h>
#include <vector_util.c>
#include <general_util.c>

#ifndef IO_ALGOS_C
#define IO_ALGOS_C

// #define IO_ALGOS_VERBOSE

void flags_load (char *flags_path, double *point_flag_weights,
                 flag_data flags)
{
    FILE *flags_file = fopen (flags_path, "r");

    long data_len = flags.len;
    long data_flen = flags.flen;

    double *flags_bin = flags.set;
    double *point_flag = flags.point_set;

    long data_id;

    for (data_id = 0; data_id < data_len;
            ++point_flag, ++data_id)
    {
        double *flags_bin_weight = point_flag_weights;

        long feature_id;

        fscanf (flags_file, "%lf", flags_bin);
        ++flags_bin_weight;

        for (feature_id = 1, *point_flag = 0;
                feature_id < data_flen; ++flags_bin,
                ++flags_bin_weight, ++feature_id)
        {
            fscanf (flags_file, ",%lf", flags_bin);
            *point_flag += *flags_bin * *flags_bin_weight;
        }

        printf ("\n");
    }

    fclose (flags_file);

    #ifdef IO_ALGOS_VERBOSE

    {
        printf ("\n%ld,%ld,\n", data_len, data_flen);

        double *point_flag;

        for (data_id = 0, point_flag = flags.point_set;
                data_id < data_len; ++point_flag, ++data_id)
        {
            printf ("%.2E,", *point_flag);
        }

        printf ("\n");
    }

    {
        printf ("\n");

        double *flags_bin = flags.set;

        for (data_id = 0; data_id < data_len; ++data_id)
        {
            long feature_id;

            for (feature_id = 0; feature_id < data_flen;
                    ++flags_bin, ++feature_id)
            {
                printf ("%.2E,", *flags_bin);
            }

            printf ("\n");
        }
    }

    #endif
}

long flags_point_flag_set (flag_data flags, double threshold)
{
    long data_len = flags.len;
    double *point_flag = flags.point_set;

    long data_id;
    long count;

    for (data_id = 0, count = 0;
            data_id < data_len;
            ++point_flag, ++data_id)
    {
        *point_flag = (*point_flag >= threshold ? 1 : 0);
        count += *point_flag;
    }

    return count;
}

void flags_point_flag_reeval (flag_data flags, double *point_flag_weights)
{
    long data_len = flags.len;
    long data_flen = flags.flen;
    double *point_flag = flags.point_set;
    double *flags_set = flags.set;

    long data_id;

    for (data_id = 0; data_id < data_len;
            ++point_flag, flags_set += data_flen,
            ++data_id)
    {
        *point_flag = vec_wsum (flags_set, point_flag_weights,
                                data_flen, 1);
    }
}

void data_load (char *data_path, data_workspc data)
{
    FILE *data_file = fopen (data_path, "r");

    long data_len = data.len;
    long data_flen = data.flen;

    double *data_point_bin = data.set;
    long data_id;

    for (data_id = 0; data_id < data_len;
            ++data_id)
    {
        long feature_id;

        fscanf (data_file, "%lf", data_point_bin);
        ++data_point_bin;

        for (feature_id = 1; feature_id < data_flen;
                ++data_point_bin, ++feature_id)
        {
            fscanf (data_file, ",%lf", data_point_bin);
        }
    }

    fclose (data_file);

    #ifdef IO_ALGOS_VERBOSE

    {
        printf ("\n%ld,%ld,\n", data_len, data_flen);

        double *data_set = data.set;
        long data_id;

        for (data_id = 0; data_id < data_len;
                ++data_id)
        {
            long feature_id;

            for (feature_id = 0; feature_id < data_flen;
                    ++data_set, ++feature_id)
            {
                printf ("%.2E,", *data_set);
            }

            printf ("\n");
        }
    }

    #endif
}

long data_flag_bipartitions_load (char *data_path, char *flags_path,
                                  flag_data flags, data_workspc data,
                                  double *point_flag_weights,
                                  double flag_threshold,
                                  double *point_buffer)
{
    FILE *data_file = fopen (data_path, "r");
    FILE *flags_file = fopen (flags_path, "r");

    long data_flen = data.flen;
    long data_len = data.len;

    double *flags_set_a = flags.set;
    double *flags_set_b = flags.set + data_len * data_flen - 1;

    double *point_set_a = flags.point_set;
    double *point_set_b = flags.point_set + data_len * data_flen - 1;

    double *data_set_a = data.set;
    double *data_set_b = data.set + data_len * data_flen - 1;

    long part_a_size = 0;
    long data_id;

    for (data_id = 0; data_id < data_len;
            ++data_id)
    {
        double *flag_bin = point_buffer;
        long feature_id;

        fscanf (flags_file, "%lf", flag_bin);
        ++flag_bin;

        for (feature_id = 1; feature_id < data_flen;
                ++flag_bin, ++feature_id)
        {
            fscanf (flags_file, ",%lf", flag_bin);
        }

        long point_flag = vec_wsum (point_buffer, point_flag_weights,
                                    data_flen, 1);

        double *data_point;

        if (point_flag > flag_threshold)
        {
            memcpy (flags_set_b, point_buffer,
                    data_flen * sizeof (double));

            flags_set_b -= data_flen;
            *point_set_b = point_flag;
            --point_set_b;

            data_point = data_set_b;
            data_set_b -= data_flen;
        }

        else
        {
            memcpy (flags_set_a, point_buffer,
                    data_flen * sizeof (double));

            flags_set_a += data_flen;
            *point_set_a = point_flag;
            ++point_set_a;

            data_point = data_set_a;
            data_set_a += data_flen;

            ++part_a_size;
        }

        fscanf (data_file, "%lf", data_point);
        ++data_point;

        for (feature_id = 0; feature_id < data_flen;
                ++data_point, ++feature_id)
        {
            fscanf (data_file, ",%lf", data_point);
        }
    }

    fclose (data_file);
    fclose (flags_file);

    #ifndef IO_ALGOS_VERBOSE

    {
        printf ("\n%ld,%ld,%ld,%ld,\n", data_len,
                part_a_size, data_len - part_a_size,
                data_flen);

        long data_id;
        double *flags_point_bin = flags.set;
        double *flags_point = flags.point_set;

        for (data_id = 0; data_id < data_len;
                ++flags_point, ++data_id)
        {
            printf ("(%.2E),", *flags_point);

            long feature_id;

            for (feature_id = 0; feature_id < data_flen;
                    ++feature_id, ++flags_point_bin)
            {
                printf ("%.2E,", *flags_point_bin);
            }

            printf ("\n");
        }
    }

    {
        printf ("\n%ld,%ld,%ld,%ld,\n", data_len,
                part_a_size, data_len - part_a_size,
                data_flen);

        long data_id;
        double *data_point = data.set;

        for (data_id = 0; data_id < data_len;
                ++data_id)
        {
            long feature_id;

            for (feature_id = 0; feature_id < data_flen;
                    ++feature_id, ++data_point)
            {
                printf ("%.2E,", *data_point);
            }

            printf ("\n");
        }
    }

    #endif

    return part_a_size;
}

void matrix_load (char *matrix_path, double *matrix,
                  long dim_row, long dim_col)
{
    FILE *matrix_file = fopen (matrix_path, "r");

    double *matrix_bin = matrix;
    long mat_id_row;

    for (mat_id_row = 0; mat_id_row < dim_row;
            ++mat_id_row)
    {
        long mat_id_col;

        fscanf (matrix_file, "%lf", matrix_bin);
        ++matrix_bin;

        for (mat_id_col = 1; mat_id_col < dim_col;
                ++matrix_bin, ++mat_id_col)
        {
            fscanf (matrix_file, ",%lf", matrix_bin);
        }
    }

    fclose (matrix_file);

    #ifdef IO_ALGOS_VERBOSE

    {
        printf ("\n%ld,%ld,\n", dim_row, dim_col);

        double *matrix_bin = matrix;
        long mat_id_row;

        for (mat_id_row = 0; mat_id_row < dim_row;
                ++mat_id_row)
        {
            long mat_id_col;

            for (mat_id_col = 0; mat_id_col < dim_col;
                    ++matrix_bin, ++mat_id_col)
            {
                printf ("%.2E,", *matrix_bin);
            }

            printf ("\n");
        }
    }

    #endif
}

void matrix_long_load (char *matrix_path, long *matrix,
                       long dim_row, long dim_col)
{
    FILE *matrix_file = fopen (matrix_path, "r");

    long *matrix_bin = matrix;
    long mat_id_row;

    for (mat_id_row = 0; mat_id_row < dim_row;
            ++mat_id_row)
    {
        long mat_id_col;

        fscanf (matrix_file, "%ld", matrix_bin);
        ++matrix_bin;

        for (mat_id_col = 1; mat_id_col < dim_col;
                ++matrix_bin, ++mat_id_col)
        {
            fscanf (matrix_file, ",%ld", matrix_bin);
        }
    }

    fclose (matrix_file);

    #ifdef IO_ALGOS_VERBOSE

    {
        printf ("\n%ld,%ld,\n", dim_row, dim_col);

        long *matrix_bin = matrix;
        long mat_id_row;

        for (mat_id_row = 0; mat_id_row < dim_row;
                ++mat_id_row)
        {
            long mat_id_col;

            for (mat_id_col = 0; mat_id_col < dim_col;
                    ++matrix_bin, ++mat_id_col)
            {
                printf ("%ld,", *matrix_bin);
            }

            printf ("\n");
        }
    }

    #endif
}

void matrix_save (char *matrix_path, double *matrix,
                  long dim_row, long dim_col)
{
    FILE *matrix_file = fopen (matrix_path, "w");

    double *matrix_bin = matrix;
    long mat_id_row;

    for (mat_id_row = 0; mat_id_row < dim_row;
            ++mat_id_row)
    {
        long mat_id_col;

        fprintf (matrix_file, "%lf", *matrix_bin);
        ++matrix_bin;

        for (mat_id_col = 1; mat_id_col < dim_col;
                ++matrix_bin, ++mat_id_col)
        {
            fprintf (matrix_file, ",%lf", *matrix_bin);
        }

        fprintf (matrix_file, "\n");
    }

    fclose (matrix_file);
}

void matrix_long_save (char *matrix_path, long *matrix,
                       long dim_row, long dim_col)
{
    FILE *matrix_file = fopen (matrix_path, "w");

    long *matrix_bin = matrix;
    long mat_id_row;

    for (mat_id_row = 0; mat_id_row < dim_row;
            ++mat_id_row)
    {
        long mat_id_col;

        fprintf (matrix_file, "%ld", *matrix_bin);
        ++matrix_bin;

        for (mat_id_col = 1; mat_id_col < dim_col;
                ++matrix_bin, ++mat_id_col)
        {
            fprintf (matrix_file, ",%ld", *matrix_bin);
        }

        fprintf (matrix_file, "\n");
    }

    fclose (matrix_file);
}

void data_clustering_load (char *clustering_path, data_clustering clustering)
{
    FILE *clustering_file = fopen (clustering_path, "r");

    long data_len = clustering.data_len;
    long cluster_count = clustering.cluster_count;

    long *data_cluster_ids = clustering.data_cluster_ids;
    long *cluster_sizes = clustering.cluster_sizes;
    memset (cluster_sizes, 0, cluster_count * sizeof (long));

    long data_id;

    fscanf (clustering_file, "%ld", data_cluster_ids);
    cluster_sizes[*data_cluster_ids]++;
    ++data_cluster_ids;

    for (data_id = 1; data_id < data_len;
            ++data_cluster_ids, ++data_id)
    {
        fscanf (clustering_file, ",%ld", data_cluster_ids);
        cluster_sizes[*data_cluster_ids]++;
    }

    fclose (clustering_file);

    #ifdef IO_ALGOS_VERBOSE

    {
        printf ("\n%ld,\n", data_len);

        long *data_cluster_ids = clustering.data_cluster_ids;

        for (data_id = 0; data_id < data_len;
                ++data_cluster_ids, ++data_id)
        {
            printf ("%ld,", *data_cluster_ids);
        }

        printf ("\n");
    }

    {
        printf ("\n%ld,\n", cluster_count);

        long cluster_id;

        for (cluster_id = 0; cluster_id < cluster_count;
                ++cluster_id, ++cluster_sizes)
        {
            printf ("%ld,", *cluster_sizes);
        }

        printf ("\n");
    }

    #endif

}

void clustering_margins_load (char *margins_path, clustering_margins margins)
{
    FILE *margins_file = fopen (margins_path, "r");

    long cluster_count = margins.count;
    long data_flen = margins.flen;

    double *margins_up = margins.set_up;
    double *margins_down = margins.set_down;

    long cluster_id;

    for (cluster_id = 0; cluster_id < cluster_count;
            ++cluster_id)
    {
        long feature_id;

        fscanf (margins_file, "%lf", margins_up);
        ++margins_up;

        for (feature_id = 1; feature_id < data_flen;
                ++margins_up, ++feature_id)
        {
            fscanf (margins_file, ",%lf", margins_up);
        }

        fscanf (margins_file, "%lf", margins_down);
        ++margins_down;

        for (feature_id = 1; feature_id < data_flen;
                ++margins_down, ++feature_id)
        {
            fscanf (margins_file, ",%lf", margins_down);
        }
    }

    fclose (margins_file);

    #ifdef IO_ALGOS_VERBOSE

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

#endif