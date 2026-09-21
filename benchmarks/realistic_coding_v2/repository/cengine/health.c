#include "health_policy.h"

int health_status(unsigned failures, int limit)
{
    if (limit <= 0) {
        return -1;
    }
    return health_is_healthy(failures, (unsigned)limit) ? 1 : 0;
}
