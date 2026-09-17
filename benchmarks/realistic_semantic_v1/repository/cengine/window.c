#include "window.h"

bool window_accepts(size_t events, size_t limit)
{
    return limit > 0 && events < limit;
}
