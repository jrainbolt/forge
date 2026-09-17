#include "quota.h"

bool quota_reserve(quota *value, size_t amount)
{
    if (value == NULL || amount > value->limit - value->used) {
        return false;
    }
    value->used += amount;
    return true;
}

void quota_release(quota *value, size_t amount)
{
    if (value == NULL) {
        return;
    }
    value->used = amount > value->used ? 0 : value->used - amount;
}
