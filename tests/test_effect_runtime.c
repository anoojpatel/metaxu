#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "../src/metaxu/runtimes/c/effects.h"

// Global runtime for testing
static mx_runtime_t* runtime;

// Test function to run in thread
static void* test_thread_fn(void* arg) {
    int* value = (int*)arg;
    *value = 42;
    return value;
}

void test_thread_effects() {
    printf("Testing thread effects...\n");
    
    // Test spawn
    int value = 0;
    mx_value_t* spawn_args[] = {
        mx_value_from_fn(runtime, test_thread_fn),
        mx_value_from_ptr(runtime, &value)
    };
    
    mx_value_t* thread_result = mx_perform_effect(runtime, "Thread::spawn", 2, spawn_args);
    thread_t* thread = mx_value_to_ptr(thread_result);
    assert(thread != NULL);
    free(spawn_args[0]);
    free(spawn_args[1]);
    free(thread_result);
    
    // Test join
    mx_value_t* join_args[] = {
        mx_value_from_ptr(runtime, thread)
    };
    
    mx_value_t* join_result = mx_perform_effect(runtime, "Thread::join", 1, join_args);
    int* result = mx_value_to_ptr(join_result);
    assert(result != NULL);
    assert(*result == 42);
    free(join_args[0]);
    free(join_result);
    
    // Test yield
    mx_perform_effect(runtime, "Thread::yield", 0, NULL);
    
    printf("Thread effects tests passed!\n");
}

void test_domain_effects() {
    printf("Testing domain effects...\n");
    
    // Test alloc
    mx_value_t* alloc_args[] = {
        mx_value_from_int(runtime, sizeof(int))
    };
    
    mx_value_t* domain1_result = mx_perform_effect(runtime, "Domain::alloc", 1, alloc_args);
    domain_t* domain1 = mx_value_to_ptr(domain1_result);
    assert(domain1 != NULL);
    assert(domain1->size == sizeof(int));
    assert(!domain1->borrowed);
    free(alloc_args[0]);
    free(domain1_result);
    
    mx_value_t* domain2_result = mx_perform_effect(runtime, "Domain::alloc", 1, alloc_args);
    domain_t* domain2 = mx_value_to_ptr(domain2_result);
    assert(domain2 != NULL);
    free(domain2_result);
    
    // Initialize domain1 data
    *(int*)domain1->data = 42;
    
    // Test move
    mx_value_t* move_args[] = {
        mx_value_from_ptr(runtime, domain1),
        mx_value_from_ptr(runtime, domain2)
    };
    
    mx_perform_effect(runtime, "Domain::move", 2, move_args);
    assert(domain1->data == NULL);
    assert(*(int*)domain2->data == 42);
    free(move_args[0]);
    free(move_args[1]);
    
    // Test borrow
    mx_value_t* borrow_args[] = {
        mx_value_from_ptr(runtime, domain2)
    };
    
    mx_value_t* borrow_result = mx_perform_effect(runtime, "Domain::borrow", 1, borrow_args);
    void* borrowed = mx_value_to_ptr(borrow_result);
    assert(borrowed != NULL);
    assert(*(int*)borrowed == 42);
    assert(domain2->borrowed);
    free(borrow_args[0]);
    free(borrow_result);
    
    // Test free
    mx_value_t* free_args1[] = {
        mx_value_from_ptr(runtime, domain1)
    };
    mx_value_t* free_args2[] = {
        mx_value_from_ptr(runtime, domain2)
    };
    
    mx_perform_effect(runtime, "Domain::free", 1, free_args1);
    mx_perform_effect(runtime, "Domain::free", 1, free_args2);
    free(free_args1[0]);
    free(free_args2[0]);
    
    printf("Domain effects tests passed!\n");
}

int main() {
    // Initialize runtime and effect system
    runtime = malloc(sizeof(mx_runtime_t));
    runtime->effect_system = mx_effect_system_create();
    mx_runtime_init_effects(runtime);
    
    // Run tests
    test_thread_effects();
    test_domain_effects();
    
    // Cleanup
    free(runtime->effect_system);
    free(runtime);
    
    printf("All tests passed!\n");
    return 0;
}
