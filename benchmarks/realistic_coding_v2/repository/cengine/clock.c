#include "clock.h"

uint64_t clock_elapsed(uint64_t started, uint64_t now)
{
    return now >= started ? now - started : 0;
}
