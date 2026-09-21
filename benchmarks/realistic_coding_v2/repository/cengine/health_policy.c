#include "health_policy.h"

int health_is_healthy(unsigned failures, unsigned limit)
{
    return failures < limit;
}
