#include <assert.h>
#include <stdio.h>
#include <string.h>
#include "../src/metaxu/runtimes/c/effects.h"

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
    spawn_args_t spawn_args = {
        .fn = test_thread_fn,
        .arg = &value
    };
    
    thread_state_t* thread = invoke_effect(EFFECT_SPAWN, &spawn_args);
    assert(thread != NULL);
    
    // Test join
    join_args_t join_args = {
        .thread = thread
    };
    
    int* result = invoke_effect(EFFECT_JOIN, &join_args);
    assert(result != NULL);
    assert(*result == 42);
    
    // Test yield
    invoke_effect(EFFECT_YIELD, NULL);
    
    printf("Thread effects tests passed!\n");
}

void test_domain_effects() {
    printf("Testing domain effects...\n");
    
    // Test alloc
    alloc_args_t alloc_args = {
        .size = sizeof(int)
    };
    
    domain_t* domain1 = invoke_effect(EFFECT_ALLOC, &alloc_args);
    assert(domain1 != NULL);
    assert(domain1->size == sizeof(int));
    assert(!domain1->borrowed);
    
    domain_t* domain2 = invoke_effect(EFFECT_ALLOC, &alloc_args);
    assert(domain2 != NULL);
    
    // Initialize domain1 data
    *(int*)domain1->data = 42;
    
    // Test move
    move_args_t move_args = {
        .from = domain1,
        .to = domain2
    };
    
    invoke_effect(EFFECT_MOVE, &move_args);
    assert(domain1->data == NULL);
    assert(*(int*)domain2->data == 42);
    
    // Test borrow
    borrow_args_t borrow_args = {
        .domain = domain2
    };
    
    void* borrowed = invoke_effect(EFFECT_BORROW, &borrow_args);
    assert(borrowed != NULL);
    assert(*(int*)borrowed == 42);
    assert(domain2->borrowed);
    
    // Test free
    free_args_t free_args1 = {
        .domain = domain1
    };
    free_args_t free_args2 = {
        .domain = domain2
    };
    
    invoke_effect(EFFECT_FREE, &free_args1);
    invoke_effect(EFFECT_FREE, &free_args2);
    
    printf("Domain effects tests passed!\n");
}

int main() {
    // Initialize effect system
    init_effect_system();
    
    // Run tests
    test_thread_effects();
    test_domain_effects();
    
    // Cleanup
    cleanup_effect_system();
    
    printf("All tests passed!\n");
    return 0;
}
