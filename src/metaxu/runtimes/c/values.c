#include <stdlib.h>
#include "values.h"

mx_value_t* mx_value_from_int(mx_runtime_t* rt, int64_t val) {
    mx_value_t* value = malloc(sizeof(mx_value_t));
    value->type = MX_VALUE_INT;
    value->data.int_val = val;
    return value;
}

mx_value_t* mx_value_from_bool(mx_runtime_t* rt, bool val) {
    mx_value_t* value = malloc(sizeof(mx_value_t));
    value->type = MX_VALUE_BOOL;
    value->data.bool_val = val;
    return value;
}

mx_value_t* mx_value_from_ptr(mx_runtime_t* rt, void* val) {
    mx_value_t* value = malloc(sizeof(mx_value_t));
    value->type = MX_VALUE_PTR;
    value->data.ptr_val = val;
    return value;
}

mx_value_t* mx_value_from_fn(mx_runtime_t* rt, void* val) {
    mx_value_t* value = malloc(sizeof(mx_value_t));
    value->type = MX_VALUE_FN;
    value->data.fn_val = val;
    return value;
}

int64_t mx_value_to_int(mx_value_t* val) {
    if (val->type != MX_VALUE_INT) {
        return 0; // Error case
    }
    return val->data.int_val;
}

bool mx_value_to_bool(mx_value_t* val) {
    if (val->type != MX_VALUE_BOOL) {
        return false; // Error case
    }
    return val->data.bool_val;
}

void* mx_value_to_ptr(mx_value_t* val) {
    if (val->type != MX_VALUE_PTR) {
        return NULL; // Error case
    }
    return val->data.ptr_val;
}

void* mx_value_to_fn(mx_value_t* val) {
    if (val->type != MX_VALUE_FN) {
        return NULL; // Error case
    }
    return val->data.fn_val;
}
