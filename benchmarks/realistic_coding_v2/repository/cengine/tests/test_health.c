#include "../health_policy.h"

#include <assert.h>

int health_status(unsigned failures, int limit);

int main(void)
{
    assert(health_status(0, 0) == -1);
    assert(health_status(0, 3) == 1);
    assert(health_status(3, 3) == 0);
    assert(health_status(4, 3) == 0);
    assert(health_is_healthy(2, 3) == 1);
    return 0;
}
