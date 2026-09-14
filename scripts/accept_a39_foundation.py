"""Clean Foundation E04 acceptance under A39 process-isolation modes."""

import sys

from accept_a37_foundation import main

if __name__ == "__main__":
    raise SystemExit(main(milestone="a39strict" if "--strict" in sys.argv else "a39"))
