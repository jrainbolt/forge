#ifndef CENGINE_QUOTA_H
#define CENGINE_QUOTA_H

#include <stdbool.h>
#include <stddef.h>

typedef struct quota {
    size_t used;
    size_t limit;
} quota;

bool quota_reserve(quota *value, size_t amount);
void quota_release(quota *value, size_t amount);

#endif
