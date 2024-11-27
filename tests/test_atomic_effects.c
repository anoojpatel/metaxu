#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <pthread.h>
#include <inttypes.h>
#include "../src/metaxu/runtimes/c/effects.h"

#define NUM_THREADS 4
#define ITERATIONS 1000000

// Global runtime for testing
static mx_runtime_t* runtime;
static atomic_state_t* counter;

void* increment_counter(void* arg) {
    for (int i = 0; i < ITERATIONS; i++) {
        mx_value_t* args[] = {
            mx_value_from_ptr(runtime, counter),
            mx_value_from_int(runtime, 1)
        };
        mx_perform_effect(runtime, "Atomic::add", 2, args);
        free(args[0]);
        free(args[1]);
    }
    return NULL;
}

int main() {
    // Initialize runtime and effect system
    runtime = malloc(sizeof(mx_runtime_t));
    runtime->effect_system = mx_effect_system_create();
    mx_runtime_init_effects(runtime);
    
    // Create atomic counter
    mx_value_t* args[] = {
        mx_value_from_int(runtime, 0)
    };
    mx_value_t* result = mx_perform_effect(runtime, "Atomic::alloc", 1, args);
    counter = mx_value_to_ptr(result);
    free(args[0]);
    free(result);
    
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
    mx_value_t* load_args[] = {
        mx_value_from_ptr(runtime, counter)
    };
    result = mx_perform_effect(runtime, "Atomic::load", 1, load_args);
    int64_t final_count = mx_value_to_int(result);
    printf("Final count: %" PRId64 "\n", final_count);
    assert(final_count == NUM_THREADS * ITERATIONS);
    free(load_args[0]);
    free(result);
    
    // Cleanup
    free(counter);
    free(runtime->effect_system);
    free(runtime);
    
    printf("All tests passed!\n");
    return 0;
}
