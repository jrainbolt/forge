from pyservice.config import parse_enabled, parse_timeout

assert parse_enabled({}, "FEATURE", default=True)
assert parse_enabled({"FEATURE": " yes "}, "FEATURE", default=False)
assert not parse_enabled({"FEATURE": "off"}, "FEATURE", default=True)
assert not parse_enabled({"FEATURE": "0"}, "FEATURE", default=True)
assert parse_timeout({"TIMEOUT": "9"}, "TIMEOUT", 3) == 9
