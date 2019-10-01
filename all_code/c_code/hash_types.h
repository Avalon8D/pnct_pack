#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifndef HASH_TYPES_H
#define HASH_TYPES_H

typedef struct
{
    void *memory_block;
    long memory_len;

    long table_len;
    long table_count;
    long flat_len;

    long *tables;
    long *min_counts
} count_htables;

#endif