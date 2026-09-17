#include "checksum.h"

unsigned checksum_bytes(const unsigned char *data, size_t size)
{
    unsigned result = 0;
    for (size_t index = 0; index < size; ++index) {
        result = (result * 33U) ^ data[index];
    }
    return result;
}
