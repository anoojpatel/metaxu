#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "effects.h"

// FNV-1a hash function
static uint32_t fnv1a_hash(const char* str) {
    uint32_t hash = 2166136261u;
    for (const char* s = str; *s; s++) {
        hash ^= (uint32_t)*s;
        hash *= 16777619u;
    }
    return hash;
}

mx_effect_system_t* mx_effect_system_create() {
    mx_effect_system_t* system = malloc(sizeof(mx_effect_system_t));
    memset(system, 0, sizeof(mx_effect_system_t));
    return system;
}

void mx_register_effect(mx_effect_system_t* system, const char* effect_name, mx_effect_handler_t handler) {
    uint32_t hash = fnv1a_hash(effect_name);
    uint32_t index = hash % MX_EFFECT_TABLE_SIZE;
    
    // Linear probing
    uint32_t original_index = index;
    while (system->handlers[index] != NULL) {
        if (strcmp(system->effect_names[index], effect_name) == 0) {
            // Effect already registered, update handler
            system->handlers[index] = handler;
            return;
        }
        index = (index + 1) % MX_EFFECT_TABLE_SIZE;
        if (index == original_index) {
            fprintf(stderr, "Effect table full, cannot register %s\n", effect_name);
            return;
        }
    }
    
    // Found empty slot
    system->handlers[index] = handler;
    system->effect_names[index] = effect_name;
}

mx_effect_handler_t mx_find_handler(mx_effect_system_t* system, const char* effect_name) {
    uint32_t hash = fnv1a_hash(effect_name);
    uint32_t index = hash % MX_EFFECT_TABLE_SIZE;
    
    // Linear probing
    uint32_t original_index = index;
    while (system->handlers[index] != NULL) {
        if (strcmp(system->effect_names[index], effect_name) == 0) {
            return system->handlers[index];
        }
        index = (index + 1) % MX_EFFECT_TABLE_SIZE;
        if (index == original_index) break;
    }
    
    fprintf(stderr, "Unhandled effect: %s\n", effect_name);
    return NULL;
}

mx_value_t* mx_perform_effect(mx_runtime_t* rt, const char* effect_name, int argc, mx_value_t** argv) {
    mx_effect_handler_t handler = mx_find_handler(rt->effect_system, effect_name);
    if (handler) {
        return handler(rt, argc, argv);
    }
    return NULL;
}

// Atomic effect handlers
mx_value_t* handle_atomic_alloc(mx_runtime_t* rt, int argc, mx_value_t** argv) {
    atomic_state_t* state = malloc(sizeof(atomic_state_t));
    atomic_init(&state->value, 0);
    return mx_value_from_ptr(rt, state);
}

mx_value_t* handle_atomic_load(mx_runtime_t* rt, int argc, mx_value_t** argv) {
    if (argc != 1) return mx_value_from_int(rt, 0);
    atomic_state_t* state = mx_value_to_ptr(argv[0]);
    int64_t value = atomic_load(&state->value);
    return mx_value_from_int(rt, value);
}

mx_value_t* handle_atomic_store(mx_runtime_t* rt, int argc, mx_value_t** argv) {
    if (argc != 2) return mx_value_from_bool(rt, false);
    atomic_state_t* state = mx_value_to_ptr(argv[0]);
    int64_t value = mx_value_to_int(argv[1]);
    atomic_store(&state->value, value);
    return mx_value_from_bool(rt, true);
}

mx_value_t* handle_atomic_cas(mx_runtime_t* rt, int argc, mx_value_t** argv) {
    if (argc != 3) return mx_value_from_bool(rt, false);
    atomic_state_t* state = mx_value_to_ptr(argv[0]);
    int64_t expected = mx_value_to_int(argv[1]);
    int64_t desired = mx_value_to_int(argv[2]);
    int64_t result = atomic_compare_exchange_strong(&state->value, (int*)&expected, desired);
    return mx_value_from_bool(rt, result);
}

mx_value_t* handle_atomic_add(mx_runtime_t* rt, int argc, mx_value_t** argv) {
    if (argc != 2) return mx_value_from_int(rt, 0);
    atomic_state_t* state = mx_value_to_ptr(argv[0]);
    int64_t value = mx_value_to_int(argv[1]);
    int64_t result = atomic_fetch_add(&state->value, value);
    return mx_value_from_int(rt, result);
}

mx_value_t* handle_atomic_sub(mx_runtime_t* rt, int argc, mx_value_t** argv) {
    if (argc != 2) return mx_value_from_int(rt, 0);
    atomic_state_t* state = mx_value_to_ptr(argv[0]);
    int64_t value = mx_value_to_int(argv[1]);
    int64_t result = atomic_fetch_sub(&state->value, value);
    return mx_value_from_int(rt, result);
}

// Thread effect handlers
mx_value_t* handle_thread_spawn(mx_runtime_t* rt, int argc, mx_value_t** argv) {
    if (argc != 2) return NULL;
    void* (*fn)(void*) = mx_value_to_fn(argv[0]);
    void* arg = mx_value_to_ptr(argv[1]);
    
    thread_t* thread = malloc(sizeof(thread_t));
    pthread_create(&thread->thread, NULL, fn, arg);
    return mx_value_from_ptr(rt, thread);
}

mx_value_t* handle_thread_join(mx_runtime_t* rt, int argc, mx_value_t** argv) {
    if (argc != 1) return NULL;
    thread_t* thread = mx_value_to_ptr(argv[0]);
    pthread_join(thread->thread, &thread->result);
    void* result = thread->result;
    free(thread);
    return mx_value_from_ptr(rt, result);
}

mx_value_t* handle_thread_yield(mx_runtime_t* rt, int argc, mx_value_t** argv) {
    #ifdef _POSIX_PRIORITY_SCHEDULING
    sched_yield();
    #endif
    return mx_value_from_bool(rt, true);
}

// Domain effect handlers
mx_value_t* handle_domain_alloc(mx_runtime_t* rt, int argc, mx_value_t** argv) {
    if (argc != 1) return NULL;
    size_t size = mx_value_to_int(argv[0]);
    
    domain_t* domain = malloc(sizeof(domain_t));
    domain->data = malloc(size);
    domain->size = size;
    domain->borrowed = false;
    pthread_mutex_init(&domain->mutex, NULL);
    
    return mx_value_from_ptr(rt, domain);
}

mx_value_t* handle_domain_move(mx_runtime_t* rt, int argc, mx_value_t** argv) {
    if (argc != 2) return mx_value_from_bool(rt, false);
    domain_t* from = mx_value_to_ptr(argv[0]);
    domain_t* to = mx_value_to_ptr(argv[1]);
    
    pthread_mutex_lock(&from->mutex);
    pthread_mutex_lock(&to->mutex);
    
    if (from->borrowed || to->borrowed) {
        pthread_mutex_unlock(&to->mutex);
        pthread_mutex_unlock(&from->mutex);
        return mx_value_from_bool(rt, false);
    }
    
    free(to->data);
    to->data = from->data;
    to->size = from->size;
    from->data = NULL;
    from->size = 0;
    
    pthread_mutex_unlock(&to->mutex);
    pthread_mutex_unlock(&from->mutex);
    
    return mx_value_from_bool(rt, true);
}

mx_value_t* handle_domain_borrow(mx_runtime_t* rt, int argc, mx_value_t** argv) {
    if (argc != 1) return NULL;
    domain_t* domain = mx_value_to_ptr(argv[0]);
    
    pthread_mutex_lock(&domain->mutex);
    if (domain->borrowed) {
        pthread_mutex_unlock(&domain->mutex);
        return NULL;
    }
    domain->borrowed = true;
    pthread_mutex_unlock(&domain->mutex);
    
    return mx_value_from_ptr(rt, domain->data);
}

mx_value_t* handle_domain_free(mx_runtime_t* rt, int argc, mx_value_t** argv) {
    if (argc != 1) return mx_value_from_bool(rt, false);
    domain_t* domain = mx_value_to_ptr(argv[0]);
    
    pthread_mutex_lock(&domain->mutex);
    if (domain->borrowed) {
        pthread_mutex_unlock(&domain->mutex);
        return mx_value_from_bool(rt, false);
    }
    
    free(domain->data);
    pthread_mutex_unlock(&domain->mutex);
    pthread_mutex_destroy(&domain->mutex);
    free(domain);
    
    return mx_value_from_bool(rt, true);
}

// Mutex effect handlers
mx_value_t* handle_mutex_create(mx_runtime_t* rt, int argc, mx_value_t** argv) {
    mutex_t* mutex = malloc(sizeof(mutex_t));
    pthread_mutex_init(&mutex->mutex, NULL);
    return mx_value_from_ptr(rt, mutex);
}

mx_value_t* handle_mutex_lock(mx_runtime_t* rt, int argc, mx_value_t** argv) {
    if (argc != 1) return mx_value_from_bool(rt, false);
    mutex_t* mutex = mx_value_to_ptr(argv[0]);
    int result = pthread_mutex_lock(&mutex->mutex);
    return mx_value_from_bool(rt, result == 0);
}

mx_value_t* handle_mutex_unlock(mx_runtime_t* rt, int argc, mx_value_t** argv) {
    if (argc != 1) return mx_value_from_bool(rt, false);
    mutex_t* mutex = mx_value_to_ptr(argv[0]);
    int result = pthread_mutex_unlock(&mutex->mutex);
    return mx_value_from_bool(rt, result == 0);
}

// RwLock effect handlers
mx_value_t* handle_rwlock_create(mx_runtime_t* rt, int argc, mx_value_t** argv) {
    rwlock_t* lock = malloc(sizeof(rwlock_t));
    pthread_rwlock_init(&lock->rwlock, NULL);
    return mx_value_from_ptr(rt, lock);
}

mx_value_t* handle_rwlock_read_lock(mx_runtime_t* rt, int argc, mx_value_t** argv) {
    if (argc != 1) return mx_value_from_bool(rt, false);
    rwlock_t* lock = mx_value_to_ptr(argv[0]);
    int result = pthread_rwlock_rdlock(&lock->rwlock);
    return mx_value_from_bool(rt, result == 0);
}

mx_value_t* handle_rwlock_read_unlock(mx_runtime_t* rt, int argc, mx_value_t** argv) {
    if (argc != 1) return mx_value_from_bool(rt, false);
    rwlock_t* lock = mx_value_to_ptr(argv[0]);
    int result = pthread_rwlock_unlock(&lock->rwlock);
    return mx_value_from_bool(rt, result == 0);
}

mx_value_t* handle_rwlock_write_lock(mx_runtime_t* rt, int argc, mx_value_t** argv) {
    if (argc != 1) return mx_value_from_bool(rt, false);
    rwlock_t* lock = mx_value_to_ptr(argv[0]);
    int result = pthread_rwlock_wrlock(&lock->rwlock);
    return mx_value_from_bool(rt, result == 0);
}

mx_value_t* handle_rwlock_write_unlock(mx_runtime_t* rt, int argc, mx_value_t** argv) {
    if (argc != 1) return mx_value_from_bool(rt, false);
    rwlock_t* lock = mx_value_to_ptr(argv[0]);
    int result = pthread_rwlock_unlock(&lock->rwlock);
    return mx_value_from_bool(rt, result == 0);
}

void mx_runtime_init_effects(mx_runtime_t* rt) {
    // Register atomic effects
    mx_register_effect(rt->effect_system, "Atomic::alloc", handle_atomic_alloc);
    mx_register_effect(rt->effect_system, "Atomic::load", handle_atomic_load);
    mx_register_effect(rt->effect_system, "Atomic::store", handle_atomic_store);
    mx_register_effect(rt->effect_system, "Atomic::cas", handle_atomic_cas);
    mx_register_effect(rt->effect_system, "Atomic::add", handle_atomic_add);
    mx_register_effect(rt->effect_system, "Atomic::sub", handle_atomic_sub);
    
    // Register thread effects
    mx_register_effect(rt->effect_system, "Thread::spawn", handle_thread_spawn);
    mx_register_effect(rt->effect_system, "Thread::join", handle_thread_join);
    mx_register_effect(rt->effect_system, "Thread::yield", handle_thread_yield);
    
    // Register domain effects
    mx_register_effect(rt->effect_system, "Domain::alloc", handle_domain_alloc);
    mx_register_effect(rt->effect_system, "Domain::move", handle_domain_move);
    mx_register_effect(rt->effect_system, "Domain::borrow", handle_domain_borrow);
    mx_register_effect(rt->effect_system, "Domain::free", handle_domain_free);
    
    // Register mutex effects
    mx_register_effect(rt->effect_system, "Mutex::create", handle_mutex_create);
    mx_register_effect(rt->effect_system, "Mutex::lock", handle_mutex_lock);
    mx_register_effect(rt->effect_system, "Mutex::unlock", handle_mutex_unlock);
    
    // Register rwlock effects
    mx_register_effect(rt->effect_system, "RwLock::create", handle_rwlock_create);
    mx_register_effect(rt->effect_system, "RwLock::read_lock", handle_rwlock_read_lock);
    mx_register_effect(rt->effect_system, "RwLock::read_unlock", handle_rwlock_read_unlock);
    mx_register_effect(rt->effect_system, "RwLock::write_lock", handle_rwlock_write_lock);
    mx_register_effect(rt->effect_system, "RwLock::write_unlock", handle_rwlock_write_unlock);
}
