#include "backoff.h"

unsigned backoff_delay(unsigned attempt, unsigned base, unsigned cap)
{
    unsigned delay = base > cap ? cap : base;
    for (unsigned index = 0; index < attempt && delay < cap; ++index) {
        if (delay > cap / 2) {
            return cap;
        }
        delay *= 2;
    }
    return delay;
}
