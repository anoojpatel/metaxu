#ifndef METAXU_EFFECTS_H
#define METAXU_EFFECTS_H

#include <stdint.h>
#include <stdbool.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include "values.h"

// Forward declarations
typedef struct mx_runtime mx_runtime_t;
typedef struct mx_value mx_value_t;

// Runtime types
typedef struct {
    pthread_t thread;
    void* result;
} thread_t;

typedef struct {
    atomic_int value;
} atomic_state_t;

typedef struct {
    pthread_mutex_t mutex;
} mutex_t;

typedef struct {
    pthread_rwlock_t rwlock;
} rwlock_t;

typedef struct {
    void* data;
    size_t size;
    bool borrowed;
    pthread_mutex_t mutex;
} domain_t;

// Effect handler type
typedef mx_value_t* (*mx_effect_handler_t)(mx_runtime_t* rt, int argc, mx_value_t** argv);

// Hash table size - should be prime and larger than number of effects
#define MX_EFFECT_TABLE_SIZE 101

// Effect system
typedef struct mx_effect_system {
    mx_effect_handler_t handlers[MX_EFFECT_TABLE_SIZE];  // Hash table of handlers
    const char* effect_names[MX_EFFECT_TABLE_SIZE];      // Effect names for each slot
} mx_effect_system_t;

// Runtime structure
struct mx_runtime {
    mx_effect_system_t* effect_system;
    // ... other runtime fields ...
};

// Effect system API
mx_effect_system_t* mx_effect_system_create();
void mx_register_effect(mx_effect_system_t* system, const char* effect_name, mx_effect_handler_t handler);
mx_effect_handler_t mx_find_handler(mx_effect_system_t* system, const char* effect_name);
mx_value_t* mx_perform_effect(mx_runtime_t* rt, const char* effect_name, int argc, mx_value_t** argv);
void mx_runtime_init_effects(mx_runtime_t* rt);

// Atomic effect handlers
mx_value_t* handle_alloc_atomic(mx_runtime_t* rt, int argc, mx_value_t** argv);
mx_value_t* handle_atomic_load(mx_runtime_t* rt, int argc, mx_value_t** argv);
mx_value_t* handle_atomic_store(mx_runtime_t* rt, int argc, mx_value_t** argv);
mx_value_t* handle_atomic_cas(mx_runtime_t* rt, int argc, mx_value_t** argv);
mx_value_t* handle_atomic_add(mx_runtime_t* rt, int argc, mx_value_t** argv);
mx_value_t* handle_atomic_sub(mx_runtime_t* rt, int argc, mx_value_t** argv);

// Thread effect handlers
mx_value_t* handle_thread_spawn(mx_runtime_t* rt, int argc, mx_value_t** argv);
mx_value_t* handle_thread_join(mx_runtime_t* rt, int argc, mx_value_t** argv);
mx_value_t* handle_thread_yield(mx_runtime_t* rt, int argc, mx_value_t** argv);

// Domain effect handlers
mx_value_t* handle_domain_alloc(mx_runtime_t* rt, int argc, mx_value_t** argv);
mx_value_t* handle_domain_move(mx_runtime_t* rt, int argc, mx_value_t** argv);
mx_value_t* handle_domain_borrow(mx_runtime_t* rt, int argc, mx_value_t** argv);
mx_value_t* handle_domain_free(mx_runtime_t* rt, int argc, mx_value_t** argv);

// Mutex effect handlers
mx_value_t* handle_mutex_create(mx_runtime_t* rt, int argc, mx_value_t** argv);
mx_value_t* handle_mutex_lock(mx_runtime_t* rt, int argc, mx_value_t** argv);
mx_value_t* handle_mutex_unlock(mx_runtime_t* rt, int argc, mx_value_t** argv);

// RwLock effect handlers
mx_value_t* handle_rwlock_create(mx_runtime_t* rt, int argc, mx_value_t** argv);
mx_value_t* handle_rwlock_read_lock(mx_runtime_t* rt, int argc, mx_value_t** argv);
mx_value_t* handle_rwlock_read_unlock(mx_runtime_t* rt, int argc, mx_value_t** argv);
mx_value_t* handle_rwlock_write_lock(mx_runtime_t* rt, int argc, mx_value_t** argv);
mx_value_t* handle_rwlock_write_unlock(mx_runtime_t* rt, int argc, mx_value_t** argv);

#endif // METAXU_EFFECTS_H
