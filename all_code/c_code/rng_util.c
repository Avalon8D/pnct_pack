#include <rng_types.h>

#ifndef RNG_UTIL_C
#define RNG_UTIL_C

// #define RNG_UTIL_VERBOSE

#define min(a, b) (a < b ? a : b)

#define PAIR_SAMPLE_SIZE(sample_len) \
    (2 * sample_len * sizeof (long))

#define SAMPLE_WEIGHTS_SIZE(sample_range) \
    (sample_range * sizeof (double) + sample_range * sizeof (long))

gsl_rng *rand_gen = NULL;

void init_rand (unsigned *seed)
{
    rand_gen = gsl_rng_alloc (gsl_rng_mt19937);

    if (seed)
    {
        gsl_rng_set (rand_gen, *seed);
    }

    else
    {
        gsl_rng_set (rand_gen, (unsigned long) time (NULL));
    }
}

void pair_sample_alloc (pair_sample *pairs, long sample_len)
{
    long memory_len = SAMPLE_WEIGHTS_SIZE (sample_len);
    void *memory_block = malloc (memory_len);

    pairs->memory_len = memory_len;
    pairs->memory_block = memory_block;

    pairs->count = sample_len;
    pairs->sample = memory_block;
}

void sample_weights_tables_alloc (sample_weights_tables *sample_tables, long sample_range)
{
    long memory_len = SAMPLE_WEIGHTS_SIZE (sample_range);
    void *memory_block = malloc (memory_len);

    sample_tables->memory_len = memory_len;
    sample_tables->memory_block = memory_block;

    sample_tables->sample_range = sample_range;
    sample_tables->w_threshold_table = memory_block;
    memory_block += sample_range * sizeof (double);

    sample_tables->w_id_table = memory_block;
}

void integer_partial_shuffle (long sup, long shuffle_sup, long *sample)
{
    long sample_id;

    for (sample_id = 0; sample_id < shuffle_sup;
            ++sample_id)
    {
        long swap_id = gsl_rng_uniform_int (rand_gen, sup);

        long swap = sample[swap_id];
        sample[swap_id] = sample[sample_id];
        sample[sample_id] = swap;
    }
}

long *id_content_search (long *id_content_beg,
                         long id_content_count,
                         long id)
{
    if (id_content_count <= 1)
    {
        return id_content_beg;
    }

    long half_size = id_content_count / 2;
    long *half_id = id_content_beg + 2 * half_size;

    if (id < *half_id)
    {
        return id_content_search (id_content_beg,
                                  half_size, id);
    }

    return id_content_search (half_id, id_content_count -
                              half_size, id);
}

long id_content_find (long id, long **id_content_found,
                      long *already_shuffled,
                      long id_content_count)
{
    if (!id_content_count)
    {
        *id_content_found = already_shuffled;
        return 0;
    }

    *id_content_found = id_content_search (already_shuffled, id_content_count, id);

    if (**id_content_found == id)
    {
        return 1;
    }

    if (**id_content_found < id)
    {
        *id_content_found += 2;
    }

    return 0;
}

void insert_id_content (long id, long content, long *already_shuffled,
                        long shuffled_count, long *insert_id)
{
    long *last_id = already_shuffled + 2 * shuffled_count;

    for (; insert_id <= last_id; insert_id += 2)
    {
        long swap = insert_id[0];
        insert_id[0] = id;
        id = swap;

        swap = insert_id[1];
        insert_id[1] = content;
        content = swap;
    }
}

#define SAMPLE_BUFFER_SIZE(sample_size)\
    (2 * sample_size * sizeof (long))

void integer_subset_sample (long sup, long sample_size,
                            long *sample, long *sample_buffer)
{
    sample_size = min (sup, sample_size);

    {
        long id;

        for (id = 0; id < sample_size; ++id)
        {
            sample[id] = id;
        }
    }

    long *already_shuffled = sample_buffer;
    long sample_id;
    long shuffled_count = 0;

    for (sample_id = 0; sample_id < sample_size;
            ++sample_id)
    {
        long new_id = gsl_rng_uniform_int (rand_gen, sup);

        if (new_id < sample_size)
        {
            long swap = sample[sample_id];
            sample[sample_id] = sample[new_id];
            sample[new_id] = swap;

            continue;
        }

        long *already_shuffled_id;
        long found = id_content_find (new_id, &already_shuffled_id,
                                      already_shuffled, shuffled_count);

        if (found)
        {
            new_id = sample[sample_id];
            sample[sample_id] = already_shuffled_id[1];
            already_shuffled_id[1] = new_id;
        }

        else
        {
            insert_id_content (new_id, sample[sample_id],
                               already_shuffled, shuffled_count,
                               already_shuffled_id);

            sample[sample_id] = new_id;
            ++shuffled_count;
        }
    }

    #ifdef RNG_UTIL_VERBOSE

    {
        printf ("\n");

        long id_content_id;

        for (id_content_id = 0;
                id_content_id < shuffled_count;
                ++id_content_id, already_shuffled += 2)
        {
            printf ("(%ld,%ld),", already_shuffled[0], already_shuffled[1]);
        }

        printf ("\n");
    }

    {
        printf ("\n");

        long sample_id;

        for (sample_id = 0;
                sample_id < sample_size;
                ++sample_id, ++sample)
        {
            printf ("%ld,", *sample);
        }

        printf ("\n");
    }

    #endif
}

// try to make it faster
void pair_from_integer (long n, long *pair)
{
    long pair_a = 0;
    long pair_count = 0;

    while (pair_count <= n)
    {
        ++pair_a;
        pair_count += pair_a;
    }

    pair[0] = pair_a;
    pair[1] = n - pair_count + pair_a;
}

// samples sample_size unique pairs of integers (i,j) with i,j in [0..data_len)
// sampled pair are guaranteed to no belong to the diagonal (i,i)
// and to be pseudo uniform on the set of those pairs not in the diagonal
void gen_pair_sample (pair_sample *pairs, long sample_size, long data_len)
{
    long *id_sample = malloc (sample_size * sizeof (long));
    long *sample_buffer = malloc (SAMPLE_BUFFER_SIZE (sample_size));

    pair_sample_alloc (pairs, sample_size);
    long *sample = pairs->sample;

    integer_subset_sample ((data_len * (data_len - 1)) / 2, sample_size,
                           id_sample, sample_buffer);

    long sample_id;

    for (sample_id = 0; sample_id < sample_size;
            ++sample_id, sample += 2)
    {
        pair_from_integer (id_sample[sample_id], sample);

        if (gsl_rng_uniform_int (rand_gen, 2))
        {
            long swap_id = sample[0];
            sample[0] = sample[1];
            sample[0] = swap_id;
        }
    }

    #ifdef RNG_UTIL_VERBOSE

    {
        printf ("\n");

        long sample_id;

        for (sample_id = 0; sample_id < sample_size;
                ++sample_id, sample += 2)
        {
            printf ("(%ld,%ld),", sample[0], sample[1]);
        }

        printf ("\n");
    }

    #endif
}

// generates a sample of sample_size pairs of integers
// corresponding to the set of edges of a tree created
// from a prufer sequence of size sample_size - 2
// pairs_ids must be 2 * (sample_size - 1) in length
// degree buffer and arg_buffer must be sample_size in length
void random_pairs_from_prufer (long *pairs_ids, long sample_size,
                               long *degree_buffer, long *arg_buffer)
{
    vec_long_set_scal (degree_buffer, 1, sample_size, 1);

    {
        long arg_id;

        for (arg_id = 0; arg_id < sample_size; ++arg_id)
        {
            arg_buffer[arg_id] = arg_id;
        }
    }

    long *large_nodes = arg_buffer + sample_size - 1;

    {
        long sample_id;

        for (sample_id = 2; sample_id < sample_size; ++sample_id)
        {
            long rand_arg_id = gsl_rng_uniform_int (rand_gen, sample_size);
            long rand_arg = arg_buffer[rand_arg_id];
            degree_buffer[rand_arg]++;

            if (degree_buffer[rand_arg] == 2)
            {
                arg_buffer[rand_arg_id] = *large_nodes;
                *large_nodes = rand_arg;
                large_nodes--;
            }
        }

        large_nodes++;
    }

    {
        long *deg_one_node = arg_buffer;
        long *arg_sup = arg_buffer + sample_size - 2;
        long *large_node = large_nodes;
        long *pair = pairs_ids;

        while (deg_one_node < arg_sup)
        {
            long node_id = *large_node;
            long node_deg = degree_buffer[node_id];

            while (node_deg > 1)
            {
                pair[0] = *deg_one_node;
                pair[1] = node_id;
                pair += 2;
                ++deg_one_node;
                --node_deg;
            }

            ++large_node;
        }

        --large_node;
        pair[0] = *deg_one_node;
        pair[1] = *large_node;
    }

    #ifdef RNG_UTIL_VERBOSE

    {
        printf ("\n");

        long pair_id;
        long *pair = pairs_ids;

        for (pair_id = 1; pair_id < sample_size; ++pair_id, pair += 2)
        {
            printf ("(%ld, %ld), ", pair[0], pair[1]);
        }

        printf ("\n");
    }

    #endif
}

// here, sample_size refers to the amount of prufer trees sampled  in order to gen pairs
// effectively, the amount of pairs in the sample is sample_size * (data_len - 1)
// there is no guarantee of uniqueness in samples with sample_size > 1
// though, the samples are always guaranteed to not have diagonal pairs (i,i)
void gen_prufer_pair_sample (pair_sample *pairs, long sample_size, long data_len)
{
    pair_sample_alloc (pairs, sample_size * (data_len - 1));
    long *sample = pairs->sample;

    long *degree_buffer = malloc (data_len * sizeof (long));
    long *arg_buffer = malloc (data_len * sizeof (long));

    {
        long sample_id;

        for (sample_id = 0; sample_id < sample_size; ++sample_id, sample += 2 * (data_len - 1))
        {
            random_pairs_from_prufer (sample, data_len, degree_buffer, arg_buffer);
        }
    }

    #ifdef RNG_UTIL_VERBOSE

    {
        printf ("\n");

        long pair_id;
        long *pair = pairs->sample;
        sample_size *= (data_len - 1);

        for (pair_id = 0; pair_id < sample_size; ++pair_id, pair += 2)
        {
            printf ("(%ld, %ld), ", pair[0], pair[1]);
        }

        printf ("\n");
    }

    #endif
}

void rng_weighted_integer_lookup_tables (sample_weights_tables sample_tables, double *weights)
{
    long sup = sample_tables.sample_range;
    double *w_threshold_table = sample_tables.w_threshold_table;
    long *w_id_table = sample_tables.w_id_table;

    {
        double sum = 0;
        double fsup = sup;

        long id;

        for (id = sum = 0; id < sup; ++id)
        {
            double weight = weights[id];
            sum += weight;

            w_threshold_table[id] = fsup * weight;
            w_id_table[id] = -1;
        }

        long all_filled = 0;

        for (id = 0; id < sup; ++id)
        {
            w_threshold_table[id] /= sum;
            all_filled += (w_threshold_table[id] < 1.0 ? 0 : 1);
        }

        if (all_filled == sup)
        {
            return;
        }
    }

    double *over_weight = w_threshold_table;
    long over_id = 0;
    double *under_weight = w_threshold_table;
    long under_id = 0;

    while (over_id != sup)
    {
        if (*over_weight > 1.0)
        {
            while (under_id != sup && ! (*under_weight < 1.0))
            {
                ++under_weight;
                ++under_id;
            }

            *over_weight -= 1 - *under_weight;
            w_id_table[under_id] = over_id;

            ++under_weight;
            ++under_id;

            if (*over_weight < 1.0 && under_id > over_id)
            {
                under_weight = over_weight;
                under_id = over_id;

                ++over_weight;
                ++over_id;
            }
        }

        else
        {
            ++over_weight;
            ++over_id;
        }
    }

    #ifdef RNG_UTIL_VERBOSE

    {
        printf ("\n");

        long threshold_id;

        for (threshold_id = 0; threshold_id < sup;
                ++w_threshold_table, ++threshold_id)
        {
            printf ("%.2E,", *w_threshold_table);
        }

        printf ("\n");
    }

    {
        printf ("\n");

        long threshold_id;

        for (threshold_id = 0; threshold_id < sup;
                ++w_id_table, ++threshold_id)
        {
            printf ("%ld,", *w_id_table);
        }

        printf ("\n");
    }

    #endif
}

long rng_weighted_integer (sample_weights_tables sample_tables,
                           long offset)
{
    long sup = sample_tables.sample_range;
    double *w_threshold_table = sample_tables.w_threshold_table;
    long *w_id_table = sample_tables.w_id_table;

    double ftable_pos;
    double rand_key = modf (((double) sup) * gsl_rng_uniform (rand_gen),
                            &ftable_pos);
    long table_pos = ftable_pos;

    if (rand_key < w_threshold_table[table_pos])
    {
        return table_pos + offset;
    }

    return w_id_table[table_pos] + offset;
}

void rng_uniform_increments (long sup, long sample_size,
                             long *increments, long *sample_buffer)
{
    integer_subset_sample (sup - 2, sample_size - 1, increments + 1,
                           sample_buffer);

    increments[0] = 0;
    gsl_sort_long (increments, 1, sample_size);

    long sample_id;
    long *increments_cur = increments;
    long *increments_pp = increments + 1;

    for (sample_id = 1; sample_id < sample_size;
            ++sample_id, ++increments_cur, ++increments_pp)
    {
        *increments_cur = *increments_pp - *increments_cur + 1;
    }

    *increments_cur = sup - *increments_cur + 1;

    #ifdef RNG_UTIL_VERBOSE

    {
        printf ("\n");

        long *increment = increments;
        long incr_id;

        for (incr_id = 0; incr_id < sample_size;
                ++incr_id, ++increment)
        {
            printf ("%ld,", *increment);
        }

        printf ("\n");
    }

    #endif
}

long rng_weighted_int_naive (long sup, double *weights,
                             double weights_sum, long offset)
{
    double rand_key = weights_sum * gsl_rng_uniform (rand_gen);

    double partial_sum = 0;
    long id = 0;

    do
    {
        partial_sum += weights[id];
        ++id;
    }
    while (id < sup && rand_key > partial_sum);

    return id - 1 + offset;
}

#endif
