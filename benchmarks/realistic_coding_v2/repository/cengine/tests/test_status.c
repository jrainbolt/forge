#include "../status.h"

#include <assert.h>

int main(void)
{
    assert(normalize_dependency_status(ENGINE_OK) == ENGINE_OK);
    assert(normalize_dependency_status(ENGINE_RETRY) == ENGINE_RETRY);
    assert(normalize_dependency_status(ENGINE_IO_ERROR) == ENGINE_IO_ERROR);
    assert(normalize_dependency_status((engine_status)99) == ENGINE_INVALID);
    return 0;
}
