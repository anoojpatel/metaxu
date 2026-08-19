#include <stdio.h>
#include <pthread.h>
typedef struct { long lo, hi, out; } Job;
static long work(long lo, long hi) {
    long acc = 0;
    for (long i = lo; i < hi; i++) acc += (i * i) % 1000;
    return acc;
}
static void *runjob(void *p) { Job *j = p; j->out = work(j->lo, j->hi); return NULL; }
int main(void) {
    long n = 40000000, q = n / 4;
    long serial = work(0, n);
    Job jobs[4] = {{0,q,0},{q,2*q,0},{2*q,3*q,0},{3*q,n,0}};
    pthread_t ts[4];
    for (int i = 0; i < 4; i++) pthread_create(&ts[i], NULL, runjob, &jobs[i]);
    long parallel = 0;
    for (int i = 0; i < 4; i++) { pthread_join(ts[i], NULL); parallel += jobs[i].out; }
    printf("%ld\n%ld\n", serial, parallel);
    return serial == parallel ? 0 : 1;
}
