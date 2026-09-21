#include "../backoff.h"

#include <assert.h>

unsigned retry_wait(unsigned attempt);

int main(void)
{
    assert(retry_wait(0) == 2);
    assert(retry_wait(1) == 4);
    assert(retry_wait(4) == 30);
    assert(backoff_delay(3, 1, 20) == 8);
    assert(backoff_delay(100, 2, 30) == 30);
    return 0;
}
