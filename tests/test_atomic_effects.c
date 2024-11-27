#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <pthread.h>
#include <inttypes.h>
#include "../src/metaxu/runtimes/c/effects.h"

#define NUM_THREADS 4
#define ITERATIONS 1000000

atomic_state_t counter = {0};

void* increment_counter(void* arg) {
    for (int i = 0; i < ITERATIONS; i++) {
        atomic_arith_args_t args = {
            .state = &counter,
            .operand = 1
        };
        handle_atomic_add(&args);
    }
    return NULL;
}

int main() {
    init_effect_system();
    
    // Create threads
    pthread_t threads[NUM_THREADS];
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_create(&threads[i], NULL, increment_counter, NULL);
    }
    
    // Wait for threads
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }
    
    // Check result
    atomic_load_args_t args = {
        .state = &counter
    };
    int64_t* final_count = (int64_t*)handle_atomic_load(&args);
    printf("Final count: %" PRId64 "\n", *final_count);
    assert(*final_count == NUM_THREADS * ITERATIONS);
    free(final_count);
    
    cleanup_effect_system();
    printf("All tests passed!\n");
    return 0;
}
