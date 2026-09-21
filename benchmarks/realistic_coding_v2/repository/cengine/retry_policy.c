#include "backoff.h"

unsigned retry_wait(unsigned attempt)
{
    return backoff_delay(attempt, 2, 30);
}
