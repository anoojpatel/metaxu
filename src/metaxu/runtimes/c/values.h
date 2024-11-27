#ifndef METAXU_VALUES_H
#define METAXU_VALUES_H

#include <stdint.h>
#include <stdbool.h>

// Forward declarations
typedef struct mx_runtime mx_runtime_t;

// Value type enum
typedef enum {
    MX_VALUE_INT,
    MX_VALUE_BOOL,
    MX_VALUE_PTR,
    MX_VALUE_FN
} mx_value_type_t;

// Value structure
typedef struct mx_value {
    mx_value_type_t type;
    union {
        int64_t int_val;
        bool bool_val;
        void* ptr_val;
        void* fn_val;
    } data;
} mx_value_t;

// Value creation functions
mx_value_t* mx_value_from_int(mx_runtime_t* rt, int64_t val);
mx_value_t* mx_value_from_bool(mx_runtime_t* rt, bool val);
mx_value_t* mx_value_from_ptr(mx_runtime_t* rt, void* val);
mx_value_t* mx_value_from_fn(mx_runtime_t* rt, void* val);

// Value extraction functions
int64_t mx_value_to_int(mx_value_t* val);
bool mx_value_to_bool(mx_value_t* val);
void* mx_value_to_ptr(mx_value_t* val);
void* mx_value_to_fn(mx_value_t* val);

#endif // METAXU_VALUES_H
