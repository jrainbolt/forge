#ifndef CENGINE_STATUS_H
#define CENGINE_STATUS_H

typedef enum engine_status {
    ENGINE_OK = 0,
    ENGINE_RETRY = 1,
    ENGINE_INVALID = 2,
    ENGINE_IO_ERROR = 3
} engine_status;

engine_status normalize_dependency_status(engine_status status);

#endif
