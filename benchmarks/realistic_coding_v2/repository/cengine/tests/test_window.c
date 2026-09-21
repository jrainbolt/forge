#include "../window.h"

#include <assert.h>

int main(void)
{
    assert(window_accepts(0, 3));
    assert(window_accepts(2, 3));
    assert(!window_accepts(3, 3));
    assert(!window_accepts(4, 3));
    return 0;
}
