#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <float.h>
#include <lapacke.h>

#ifndef VECTOR_UTIL_C
#define VECTOR_UTIL_C

// #define VECTOR_UTIL_VERBOSE

void vec_add (double *a, double *b, long dim, long stride_a, long stride_b)
{
    double *a_sup = a + dim * stride_a;

    for (; a < a_sup; a += stride_a, b += stride_b)
    {
        *a += *b;
    }
}

void vec_lin_comb (double *a, double *b, double c_a, double c_b,
                   long dim, long stride_a, long stride_b)
{
    double *a_sup = a + dim * stride_a;

    for (; a < a_sup; a += stride_a, b += stride_b)
    {
        *a *= *a * c_a + *b * c_b;
    }
}

void vec_add_long (double *a, long *b, long dim, long stride_a, long stride_b)
{
    double *a_sup = a + dim * stride_a;

    for (; a < a_sup; a += stride_a, b += stride_b)
    {
        *a += *b;
    }
}

void vec_sub (double *a, double *b, long dim, long stride_a, long stride_b)
{
    double *a_sup = a + dim * stride_a;

    for (; a < a_sup; a += stride_a, b += stride_b)
    {
        *a -= *b;
    }
}

void vec_div (double *a, double *b, long dim, long stride_a, long stride_b)
{
    double *a_sup = a + dim * stride_a;

    for (; a < a_sup; a += stride_a, b += stride_b)
    {
        *a /= *b;
    }
}

void vec_div_long (double *a, long *b, long dim, long stride_a, long stride_b)
{
    double *a_sup = a + dim * stride_a;

    for (; a < a_sup; a += stride_a, b += stride_b)
    {
        *a /= *b;
    }
}

void vec_mul (double *a, double *b, long dim, long stride_a, long stride_b)
{
    double *a_sup = a + dim * stride_a;

    for (; a < a_sup; a += stride_a, b += stride_b)
    {
        *a *= *b;
    }
}

void vec_afine (double *a, double b, double c, long dim, long stride)
{
    double *a_sup = a + dim * stride;

    for (; a < a_sup; a += stride)
    {
        *a += b + c;
    }
}

void vec_set_scal (double *a, double b, long dim, long stride)
{
    double *a_sup = a + dim * stride;

    for (; a < a_sup; a += stride)
    {
        *a = b;
    }
}

void vec_set (double *a, double *b, long dim, long stride_a, long stride_b)
{
    double *a_sup = a + dim * stride_a;

    for (; a < a_sup; a += stride_a, b += stride_b)
    {
        *a = *b;
    }
}

void vec_long_set_scal (long *a, long b, long dim, long stride)
{
    long *a_sup = a + dim * stride;

    for (; a < a_sup; a += stride)
    {
        *a = b;
    }
}

void vec_long_set (long *a, long *b, long dim, long stride_a, long stride_b)
{
    long *a_sup = a + dim * stride_a;

    for (; a < a_sup; a += stride_a, b += stride_b)
    {
        *a = *b;
    }
}

void vec_mul_scal (double *a, double b, long dim, long stride)
{
    double *a_sup = a + dim * stride;

    for (; a < a_sup; a += stride)
    {
        *a *= b;
    }
}

void vec_div_scal (double *a, double b, long dim, long stride)
{
    double inv_b = 1 / b;
    double *a_sup = a + dim * stride;

    for (; a < a_sup; a += stride)
    {
        *a *= inv_b;
    }
}

double vec_norm1 (double *a, long dim, long stride)
{
    double norm = 0;
    double *a_sup = a + dim * stride;

    for (; a < a_sup; a += stride)
    {
        norm += fabs (*a);
    }

    return norm;
}

double vec_1normalize (double *a, long dim, long stride)
{
    double norm = 0;
    double *a_sup = a + dim * stride;
    double *a_beg = a + dim * stride;

    for (; a_beg < a_sup; a_beg += stride)
    {
        norm += fabs (*a_beg);
    }

    if (norm > 1e-12)
    {
        double inv_norm = 1 / norm;

        for (; a < a_sup; a += stride)
        {
            *a *= inv_norm;
        }
    }

    return norm;
}

double vec_dist1 (double *a, double *b, long dim, long stride_a, long stride_b)
{
    double dist = 0;
    double *a_sup = a + dim * stride_a;

    for (; a < a_sup; a += stride_a, b += stride_b)
    {
        dist += fabs (*a - *b);
    }

    return dist;
}

double vec_wdist1 (double *a, double *b, double *w, long dim, long stride_a, long stride_b)
{
    double dist = 0;
    double *a_sup = a + dim * stride_a;

    for (; a < a_sup; a += stride_a, b += stride_b, ++w)
    {
        dist += *w * fabs (*a - *b);
    }

    return dist;
}

double vec_fdist1 (double *a, double *b, long *fs, long fdim, long stride_a, long stride_b)
{
    double dist = 0;
    long *fs_sup = fs + fdim;

    for (; fs < fs_sup; ++fs)
    {
        dist += fabs (a[*fs * stride_a] - b[*fs * stride_b]);
    }

    return dist;
}

double vec_dot (double *a, double *b, long dim, long stride_a, long stride_b)
{
    double dot = 0;
    double *a_sup = a + dim * stride_a;

    for (; a < a_sup; a += stride_a, b += stride_b)
    {
        dot += *a * *b;
    }

    return dot;
}

double vec_sqr_dist2 (double *a, double *b, long dim, long stride_a, long stride_b)
{
    double dist = 0;
    double *a_sup = a + dim * stride_a;

    for (; a < a_sup; a += stride_a, b += stride_b)
    {
        double diff = *a - *b;
        dist += diff * diff;
    }

    return dist;
}

double vec_sum (double *a, long dim, long stride)
{
    double sum = 0;
    double *a_sup = a + dim * stride;

    for (; a < a_sup; a += stride)
    {
        sum += *a;
    }

    return sum;
}

double vec_sqr_sum (double *a, long dim, long stride)
{
    double sqr_sum = 0;
    double *a_sup = a + dim * stride;

    for (; a < a_sup; a += stride)
    {

        double a_id = *a;
        sqr_sum += a_id * a_id;
    }

    return sqr_sum;
}

double vec_sqr_dev_scal (double *a, double b, long dim, long stride)
{
    double sqr_dev = 0;
    double *a_sup = a + dim * stride;

    for (; a < a_sup; a += stride)
    {
        double dev = *a - b;
        sqr_dev += dev * dev;
    }

    return sqr_dev;
}

double vec_wndist1 (double *a, double *b, double *w, long dim, long stride_a, long stride_b)
{
    double dist = 0;
    double w_sum = 0;
    double *a_sup = a + dim * stride_a;

    for (; a < a_sup; a += stride_a, b += stride_b, ++w)
    {
        dist += *w * fabs (*a - *b);
        w_sum += *w;
    }

    return dist / w_sum;
}

double vec_sqr_wndist2 (double *a, double *b, double *w, long dim, long stride_a, long stride_b)
{
    double dist = 0;
    double w_sum = 0;
    double *a_sup = a + dim * stride_a;

    for (; a < a_sup; a += stride_a, b += stride_b, ++w)
    {
        double diff = *a - *b;
        dist += *w * diff * diff;
        w_sum += *w;
    }

    return dist / w_sum;
}

double vec_wsum (double *a, double *w, long dim, long stride)
{
    double sum = 0;
    double *a_sup = a + dim * stride;

    for (; a < a_sup; a += stride, ++w)
    {
        sum += *w * *a;
    }

    return sum;
}

double vec_int_wsum (double *a, long *w, long dim, long stride)
{
    double sum = 0;
    double *a_sup = a + dim * stride;

    for (; a < a_sup; a += stride, ++w)
    {
        sum += *w * *a;
    }

    return sum;
}

double vec_harmonic_weights (double *a, long dim, long stride)
{
    double inv_sum = 0;
    double *a_sup = a + dim * stride;
    double *a_beg = a;

    for (; a_beg < a_sup; a_beg += stride)
    {
        inv_sum += (*a_beg = 1 / *a_beg);
    }

    for (;  a < a_sup; a += stride)
    {
        *a /= inv_sum;
    }

    return inv_sum;
}

void vec_abs_dev (double *a, double *b, double *res, long dim, long stride_a, long stride_b)
{
    double *a_sup = a + dim * stride_a;

    for (; a < a_sup; a += stride_a, b += stride_b, ++res)
    {
        *res = fabs (*a - *b);
    }
}


void mat_normalize2_rows (double *mat_a, long dim_row, long dim_col,
                          long row_stride, long col_stride, long transp)
{
    if (transp)
    {
        double *mat_a_sup = mat_a + dim_col * col_stride;

        for (; mat_a < mat_a_sup; mat_a += col_stride)
        {
            double col_norm = vec_sqr_sum (mat_a, dim_row, row_stride);

            if (col_norm < 1e-12)
            {
                continue;
            }

            vec_div_scal (mat_a, sqrt (col_norm), dim_row, row_stride);
        }
    }

    else
    {
        double *mat_a_sup = mat_a + dim_row * row_stride;

        for (; mat_a < mat_a_sup; mat_a += row_stride)
        {
            double row_norm = vec_sqr_sum (mat_a, dim_col, col_stride);

            if (row_norm < 1e-12)
            {
                continue;
            }

            vec_div_scal (mat_a, sqrt (row_norm), dim_col, col_stride);
        }
    }
}

void mat_normalize1_rows (double *mat_a, long dim_row, long dim_col,
                          long row_stride, long col_stride, long transp)
{
    if (transp)
    {
        double *mat_a_sup = mat_a + dim_col * col_stride;

        for (; mat_a < mat_a_sup; mat_a += col_stride)
        {
            double col_norm = vec_norm1 (mat_a, dim_row, row_stride);

            if (col_norm < 1e-12)
            {
                continue;
            }

            vec_div_scal (mat_a, col_norm, dim_row, row_stride);
        }
    }

    else
    {
        double *mat_a_sup = mat_a + dim_row * row_stride;

        for (; mat_a < mat_a_sup; mat_a += row_stride)
        {
            double row_norm = vec_norm1 (mat_a, dim_col, col_stride);

            if (row_norm < 1e-12)
            {
                continue;
            }

            vec_div_scal (mat_a, row_norm, dim_col, col_stride);
        }
    }
}

void mat_sub_idtty (double *mat_a, double c_idtty, long dim_row,
                    long dim_col, long col_stride, long row_stride,
                    long minus)
{
    double *mat_a_sup = mat_a + dim_col * col_stride + dim_row * row_stride;
    double *mat_a_diag = mat_a;
    long diag_stride = row_stride + col_stride;

    for (; mat_a_diag < mat_a_sup; mat_a_diag += diag_stride)
    {
        *mat_a_diag -= c_idtty;
    }

    if (minus)
    {
        mat_a_sup = mat_a + dim_row * row_stride;

        for (; mat_a < mat_a_sup; mat_a += row_stride)
        {
            vec_mul_scal (mat_a, -1, dim_col, col_stride);
        }
    }
}

double data_vec_index_int_sum (double *data_a, long stride,
                               long *index, long dim_index)
{
    long id;
    double sum = 0;

    for (id = 0; id < dim_index; ++id)
    {
        sum += data_a[index[id] * stride];
    }

    return sum;
}

void data_vec_index_int_add (double *data_a, long stride_a,
                             double *data_b, long stride_b,
                             long *index, long dim_index)
{
    long id;

    for (id = 0; id < dim_index; ++id)
    {
        data_a[index[id] * stride_a] += data_b[index[id] * stride_b];
    }
}

double data_vec_index_int_sqr_sum (double *data_a, long stride,
                                   long *index, long dim_index)
{
    long id;
    double sqr_sum = 0;

    for (id = 0; id < dim_index; ++id)
    {
        double val = data_a[index[id] * stride];
        sqr_sum += val * val;
    }

    return sqr_sum;
}

double data_vec_index_int_sum_sqr_sum (double *data_a, long stride,
                                       long *index, long dim_index,
                                       double *sum_ref)
{
    long id;
    double sum = 0;
    double sqr_sum = 0;

    for (id = 0; id < dim_index; ++id)
    {
        double val = data_a[index[id] * stride];
        sum += val;
        sqr_sum += val * val;
    }

    *sum_ref = sum;

    return sqr_sum;
}

double data_vec_index_int_wsum (double *data_a, long stride,
                                long *index, long *w,
                                long dim_index)
{
    long id;
    double sum = 0;

    for (id = 0; id < dim_index; ++id)
    {
        sum += data_a[index[id] * stride] * w[id];
    }

    return sum;
}

double data_vec_index_int_wsqrdev
(double *data_a, long stride, long *index,
 long *w, long dim_index, double mean)
{
    long id;
    double sum = 0;

    for (id = 0; id < dim_index; ++id)
    {
        double dev = data_a[index[id] * stride] - mean;
        sum += dev * dev * w[id];
    }

    return sum;
}

// key_found contains, at the end, the position of the vector in which,
// by shifting every element of the array forward from that position,
// and inserting key there, the vector would still be sorted
// key address will be beyond vector bounds if key is larger than every
// element of the vector
// if reverse is non zero, assumes array sorted in descending order
long sorted_vec_find  (double *sorted_a, long dim, long stride, double key, double **key_found, long reverse)
{
    double *up_key = sorted_a + (dim - 1) * stride;
    double *down_key = sorted_a;

    if (!reverse)
    {
        if (key < *down_key)
        {
            *key_found = down_key;
            return 0;
        }

        if (key > *up_key)
        {
            *key_found = up_key + stride;
            return 0;
        }

        double *mid_key = down_key + ((up_key - down_key) / 2);

        while (mid_key != down_key)
        {
            if (*mid_key < key)
            {
                down_key = mid_key;
            }

            else
            {
                up_key = mid_key;
            }

            mid_key = down_key + ((up_key - down_key) / 2);
        }

        *key_found = mid_key;

        if (*mid_key == key)
        {
            return 1;
        }
    }

    else
    {
        if (key > *down_key)
        {
            *key_found = down_key;
            return 0;
        }

        if (key < *up_key)
        {
            *key_found = up_key + stride;
            return 0;
        }

        double *mid_key = down_key + ((up_key - down_key) / 2);

        while (mid_key != down_key)
        {
            if (*mid_key > key)
            {
                down_key = mid_key;
            }

            else
            {
                up_key = mid_key;
            }

            mid_key = down_key + ((up_key - down_key) / 2);
        }

        *key_found = mid_key;

        if (*mid_key == key)
        {
            return 1;
        }
    }

    return 0;
}

double sorted_vec_percentile (double *sorted_a, long dim, long stride, double percentage)
{
    double quantile_pos = (dim - 1) * percentage;
    long quantile_id = (long) quantile_pos;
    double quantile_w = quantile_pos + 1 - quantile_id;
    double *quantile_val = sorted_a + quantile_id * stride;

    return quantile_w * quantile_val[0] + (1 - quantile_w) * quantile_val[stride];
}

// index_found contains, at the end, the index of the index vector in which,
// by shifting every element of the array forward from that position,
// and inserting key there, the index vector would still be sorted
// index address will be beyond index vector bounds if key is larger than every
// element of the vector
// if reverse is non zero, assumes array sorted in descending order
long argsorted_vec_find  (double *a, long stride, long *sorted_index, long dim_index,
                          double key, long **index_found, long reverse)
{
    long *up_index = sorted_index + dim_index - 1;
    long *down_index = sorted_index;

    if (!reverse)
    {
        if (key < a[*down_index * stride])
        {
            *index_found = down_index;
        }

        if (key > a[*up_index * stride])
        {
            *index_found = up_index + 1;
        }

        long *mid_index = down_index + ((up_index - down_index) / 2);

        while (down_index != up_index)
        {
            if (a[*mid_index * stride] < key)
            {
                down_index = mid_index + 1;
            }

            else
            {
                up_index = mid_index;
            }

            mid_index = down_index + ((up_index - down_index) / 2);
        }

        *index_found = mid_index;

        if (a[*mid_index * stride] == key)
        {
            return 1;
        }
    }

    else
    {
        if (key < a[*up_index * stride])
        {
            *index_found = down_index;
        }

        if (key > a[*up_index * stride])
        {
            *index_found = up_index + 1;
        }

        long *mid_index = down_index + ((up_index - down_index) / 2);

        while (down_index != up_index)
        {
            if (a[*mid_index * stride] > key)
            {
                down_index = mid_index + 1;
            }

            else
            {
                up_index = mid_index;
            }

            mid_index = down_index + ((up_index - down_index) / 2);
        }

        *index_found = mid_index;

        if (a[*mid_index * stride] == key)
        {
            return 1;
        }
    }

    return 0;
}


double argsorted_vec_percentile (double *a, long stride, long *sorted_index,
                                 long dim_index, double percentage)
{
    double quantile_pos = (dim_index - 1) * percentage;
    long quantile_id = (long) quantile_pos;
    double quantile_w = quantile_pos + 1 - quantile_id;

    return quantile_w * a[sorted_index[quantile_id] * stride] + (1 - quantile_w) * a[sorted_index[quantile_id + 1] * stride];
}

// return the amount of indices in index, of the vector
// which lie within bound_up and bound_down
// bounded_index contents become the address of index
// within bounds, if there's any, and NULL otherwise
// bounded_index content's address is assumed valid
// if reverse is non zero, assumes array sorted in descending order
long argsorted_vec_within_bounds (double *a, long stride, long *sorted_index,
                                  long dim_index, long **bounded_index,
                                  double bound_up, double bound_down, long reverse)
{
    if (bound_up < bound_down)
    {
        *bounded_index = NULL;
        return -1;
    }

    long *posi_up;
    long *posi_down;

    argsorted_vec_find (a, stride, sorted_index, dim_index,
                        bound_up, &posi_up, reverse);

    argsorted_vec_find (a, stride, sorted_index, dim_index,
                        bound_down, &posi_down, reverse);

    *bounded_index = posi_down;

    return posi_up - posi_down;
}

#define TRI_BUFFER_SIZE(dim)\
    2 * dim * sizeof (double)

// on exit, SPD_matrix contains the eigenvectors of SPD_matrix
void SPD_spectral_decomp (double *SPD_matrix, double *eigen_values, long dim)
{
    double *tri_buffer = malloc (TRI_BUFFER_SIZE (dim));
    double *tri_subdiag = tri_buffer;
    double *tri_tau = tri_buffer + dim;

    LAPACKE_dsytrd (LAPACK_ROW_MAJOR, 'U', dim, SPD_matrix,
                    dim, eigen_values, tri_subdiag,
                    tri_tau);

    LAPACKE_dorgtr (LAPACK_ROW_MAJOR, 'U', dim, SPD_matrix,
                    dim, tri_tau);

    LAPACKE_dpteqr (LAPACK_ROW_MAJOR, 'V', dim, eigen_values,
                    tri_subdiag, SPD_matrix, dim);
}

#endif
