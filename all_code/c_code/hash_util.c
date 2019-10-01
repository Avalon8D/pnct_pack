#include <hash_types.h>

#ifndef HASH_UTIL_C
#define HASH_UTIL_C

#define HASH_BITS 61

#define HASH_INF 314159
#define HASH_NAN 0
#define HASH_MODULUS (((size_t)1 << HASH_BITS) - 1)

#define COUNT_HTABLE_SIZE(table_len, table_count) \
    ((table_count + 1) * table_len * sizeof (long))

void count_htable_alloc (count_htables *counts, long table_len, long table_count)
{
    long memory_len = COUNT_HTABLE_SIZE (table_len, table_count);
    void *memory_block = malloc (memory_len);
    memset (memory_block, 0, memory_len);

    counts->memory_len = memory_len;
    counts->memory_block = memory_block;

    counts->table_len = table_len;
    counts->flat_len = table_len * table_count;
    counts->tables = memory_block;

    memory_block += counts->flat_len * sizeof (long);
    counts->min_counts = memory_block;
}

/*
    shamelessly stolen from cpython pyhash functions and definitions
    merely adapted to not need all of python libraries
    credit NOT mine for these functions

    Some of the documentation is bellow, for it is very good
*/
/*
    For numeric types, the hash of a number x is based on the reduction
    of x modulo the prime P = 2**_PyHASH_BITS - 1.  It's designed so that
    hash(x) == hash(y) whenever x and y are numerically equal, even if
    x and y have different types.
    A quick summary of the hashing strategy:
    (1) First define the 'reduction of x modulo P' for any rational
    number x; this is a standard extension of the usual notion of
    reduction modulo P for integers.  If x == p/q (written in lowest
    terms), the reduction is interpreted as the reduction of p times
    the inverse of the reduction of q, all modulo P; if q is exactly
    divisible by P then define the reduction to be infinity.  So we've
    got a well-defined map
      reduce : { rational numbers } -> { 0, 1, 2, ..., P-1, infinity }.
    (2) Now for a rational number x, define hash(x) by:
      reduce(x)   if x >= 0
      -reduce(-x) if x < 0
    If the result of the reduction is infinity (this is impossible for
    integers, floats and Decimals) then use the predefined hash value
    _PyHASH_INF for x >= 0, or -_PyHASH_INF for x < 0, instead.
    _PyHASH_INF, -_PyHASH_INF and _PyHASH_NAN are also used for the
    hashes of float and Decimal infinities and nans.
    A selling point for the above strategy is that it makes it possible
    to compute hashes of decimal and binary floating-point numbers
    efficiently, even if the exponent of the binary or decimal number
    is large.  The key point is that
      reduce(x * y) == reduce(x) * reduce(y) (modulo _PyHASH_MODULUS)
    provided that {reduce(x), reduce(y)} != {0, infinity}.  The reduction of a
    binary or decimal float is never infinity, since the denominator is a power
    of 2 (for binary) or a divisor of a power of 10 (for decimal).  So we have,
    for nonnegative x,
      reduce(x * 2**e) == reduce(x) * reduce(2**e) % _PyHASH_MODULUS
      reduce(x * 10**e) == reduce(x) * reduce(10**e) % _PyHASH_MODULUS
    and reduce(10**e) can be computed efficiently by the usual modular
    exponentiation algorithm.  For reduce(2**e) it's even better: since
    P is of the form 2**n-1, reduce(2**e) is 2**(e mod n), and multiplication
    by 2**(e mod n) modulo 2**n-1 just amounts to a rotation of bits.
*/
long HashDouble (double v)
{
    int e, sign;
    double m;
    unsigned long x, y;

    if (!isfinite (v))
    {
        if (isinf (v))
        { return v > 0 ? HASH_INF : -HASH_INF; }

        else
        { return HASH_NAN; }
    }

    m = frexp (v, &e);

    sign = 1;

    if (m < 0)
    {
        sign = -1;
        m = -m;
    }

    /*  process 28 bits at a time;  this should work well both for binary
        and hexadecimal floating point. */
    x = 0;

    while (m)
    {
        x = ((x << 28) & HASH_MODULUS) | x >> (HASH_BITS - 28);
        m *= 268435456.0;  /* 2**28 */
        e -= 28;
        y = (unsigned long)m;  /* pull out integer part */
        m -= y;
        x += y;

        if (x >= HASH_MODULUS)
        { x -= HASH_MODULUS; }
    }

    /* adjust for the exponent;  first reduce it modulo HASH_BITS */
    e = e >= 0 ? e % HASH_BITS : HASH_BITS - 1 - ((-1 - e) % HASH_BITS);
    x = ((x << e) & HASH_MODULUS) | x >> (HASH_BITS - e);

    x = x * sign;

    if (x == (unsigned long) - 1)
    { x = (unsigned long) - 2; }

    return (long)x;
}

// truncates an affine map applied entry wise to the nearest integer
// then xors values together to obtain a single value to hash
// affine here means (x - a) / b, where a is a vector and b a real value
// i.e. (x - sample_min) / (sample_max - sample_mean),
// (x - sample_mean) / (sample_std), etc

long trunc_affine_hash (count_htables *counts, double *x, double *a, double b, long len):
{

}

#endif