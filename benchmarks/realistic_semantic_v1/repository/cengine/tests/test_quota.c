#include "../quota.h"

#include <assert.h>

int main(void)
{
    quota value = {3, 10};
    assert(quota_reserve(&value, 7));
    assert(value.used == 10);
    assert(!quota_reserve(&value, 1));
    quota_release(&value, 4);
    assert(value.used == 6);
    quota_release(&value, 20);
    assert(value.used == 0);
    return 0;
}
