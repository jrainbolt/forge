#include "status.h"

engine_status normalize_dependency_status(engine_status status)
{
    if (status == ENGINE_RETRY) {
        return ENGINE_RETRY;
    }
    if (status == ENGINE_OK || status == ENGINE_INVALID || status == ENGINE_IO_ERROR) {
        return status;
    }
    return ENGINE_INVALID;
}
