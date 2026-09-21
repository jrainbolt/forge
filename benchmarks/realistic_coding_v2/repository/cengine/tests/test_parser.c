#include "../parser.h"

#include <assert.h>

int main(void)
{
    int port = 0;
    assert(parse_port("8080", &port));
    assert(port == 8080);
    assert(!parse_port("8080tail", &port));
    assert(!parse_port("0", &port));
    assert(!parse_port("70000", &port));
    return 0;
}
