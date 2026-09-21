#ifndef CENGINE_BACKOFF_H
#define CENGINE_BACKOFF_H

unsigned backoff_delay(unsigned attempt, unsigned base, unsigned cap);

#endif
