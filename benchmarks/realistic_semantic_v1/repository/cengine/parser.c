#include "parser.h"

#include <errno.h>
#include <limits.h>
#include <stdlib.h>

bool parse_port(const char *text, int *output)
{
    char *end = NULL;
    long value;
    if (text == NULL || output == NULL || *text == '\0') {
        return false;
    }
    errno = 0;
    value = strtol(text, &end, 10);
    if (errno != 0 || *end != '\0' || value < 1 || value > 65535) {
        return false;
    }
    *output = (int)value;
    return true;
}
