#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <vector_util.c>

#ifndef GENERAL_UTIL_C
#define GENERAL_UTIL_C

// #define GENERAL_UTIL_VERBOSE

#define GET_TIME(now)\
    {\
        struct timespec time; \
        clock_gettime (CLOCK_MONOTONIC_RAW, &time); \
        now = time.tv_sec + time.tv_nsec / 1000000000.0; \
    }

#define min(a, b) (a < b ? a : b)
#define max(a, b) (a > b ? a : b)

void memswap (void *a, void *b, size_t size)
{
    char *a_swap = (char *)a, *b_swap = (char *)b;
    char *a_end = a + size;

    while (a_swap < a_end)
    {
        char temp = *a_swap;
        *a_swap = *b_swap;
        *b_swap = temp;

        a_swap++, b_swap++;
    }
}

void double_insert_keep_least (double key, double *double_array,
                               long double_count)
{
    double *key_id;

    long found = sorted_vec_find (double_array, double_count, 1,
                                  key, &key_id, 0);

    double *double_array_end = double_array + double_count;

    if (!found)
    {
        for (; key_id != double_array_end; ++key_id)
        {
            double swap = key;
            key = *key_id;
            *key_id = swap;
        }
    }
}

void double_mat_sum_matches (double *mat, long *matches, long dim)
{
    long dim_row;

    for (dim_row = 0; dim_row < dim;
            ++dim_row)
    {
        long dim_col;
        long row_match = matches [dim_row];
        mat[dim_row * dim + dim_row] += 1.0;

        for (dim_col = dim_row + 1; dim_col < dim;
                ++dim_col)
        {
            double match = (row_match == matches[dim_col] ? 1.0 : 0.0);
            mat[dim_row * dim + dim_col] += match;
            mat[dim_col * dim + dim_row] += match;
        }
    }
}

void double_mat_wsum_matches (double *mat, long *matches, long dim, double match_w)
{
    long dim_row;

    for (dim_row = 0; dim_row < dim;
            ++dim_row)
    {
        long dim_col;
        long row_match = matches [dim_row];
        mat[dim_row * dim + dim_row] += match_w;

        for (dim_col = dim_row + 1; dim_col < dim;
                ++dim_col)
        {
            double match = (row_match == matches[dim_col] ? 1.0 : 0.0);
            match *= match_w;
            mat[dim_row * dim + dim_col] += match;
            mat[dim_col * dim + dim_row] += match;
        }
    }
}

int double_cmp (const void *a, const void *b)
{
    double val_a = * (double *)a;
    double val_b = * (double *)b;

    if (val_a == val_b)
    {
        return 0;
    }

    return (val_a > val_b ? 1 : -1);
}

int double_arg_cmp (const void *a, const void *b, void *args)
{
    struct
    {
        long stride;
        double *vals;
    } *typed_args = args;

    long vals_stride = typed_args->stride;
    double *arg_vals = typed_args->vals;
    double arg_a = arg_vals[vals_stride * * (long *)a];
    double arb_b = arg_vals[vals_stride * * (long *)b];

    return (arg_a > arb_b ? 1 : -1);
}

int long_quo_rem_cmp (const void *a, const void *b, void *args)
{
    int dividend = * (long *) args;

    long a_val = * (long *)a;
    long b_val = * (long *)b;

    long a_div = (a_val / dividend);
    long b_div = (b_val / dividend);

    long a_rem = a_val - a_div * dividend;
    long b_rem = b_val - b_div * dividend;

    long div_cmp = a_div - b_div;
    long rem_cmp = (a_rem > b_rem ? 1 : -1);

    return (div_cmp ? (div_cmp > 0 ? 1 : -1) : rem_cmp);
}

void byte_array_permute (void *array, long *permutation,
                         long array_len, long byte_len,
                         void *bytes_buffer)
{
    long curr_id = 0;
    memcpy (bytes_buffer, array, byte_len);

    do
    {
        long next_id = permutation[curr_id];

        if (next_id != curr_id)
        {
            memswap (bytes_buffer, array + next_id * byte_len, byte_len);
        }

        permutation[curr_id] = array_len;
        curr_id = next_id;

        while (curr_id < array_len && permutation[curr_id] == array_len)
        {
            ++curr_id;
        }
    }
    while (curr_id < array_len);
}

void long_array_permute (long *array, long *permutation, long array_len)
{
    long curr_id = 0;
    long item_buffer = array[0];

    do
    {
        long next_id = permutation[curr_id];

        if (next_id != curr_id)
        {
            long item_swap = array[next_id];
            array[next_id] = item_buffer;
            item_buffer = item_swap;
        }

        permutation[curr_id] = array_len;
        curr_id = next_id;

        while (curr_id < array_len && permutation[curr_id] == array_len)
        {
            ++curr_id;
        }
    }
    while (curr_id < array_len);
}

void permutation_transpose (long *permutation_t, long *permutation, long permutation_len)
{
    long curr_id;

    for (curr_id = 0; curr_id < permutation_len;
            ++curr_id, permutation++)
    {
        permutation_t[*permutation] = curr_id;
    }
}

#endif