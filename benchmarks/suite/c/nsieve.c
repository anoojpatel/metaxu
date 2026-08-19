#include <stdio.h>
#include <stdlib.h>
static long sieve(long n) {
    long *flags = malloc(n * sizeof(long));
    for (long i = 0; i < n; i++) flags[i] = 1;
    long count = 0;
    for (long p = 2; p < n; p++) {
        if (flags[p] == 1) {
            count++;
            for (long k = p + p; k < n; k += p) flags[k] = 0;
        }
    }
    free(flags);
    return count;
}
int main(void) {
    long total = 0;
    for (int r = 0; r < 3; r++) total += sieve(2000000);
    printf("%ld\n", total);
    return 0;
}
