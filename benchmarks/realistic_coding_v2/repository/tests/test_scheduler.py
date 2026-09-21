from pyservice.scheduler import schedule

assert schedule([("a", "low"), ("b", "HIGH"), ("c", "normal"), ("d", "high")]) == [
    "b",
    "d",
    "c",
    "a",
]

try:
    schedule([("x", "urgent")])
except ValueError:
    pass
else:
    raise AssertionError("unknown priority should be rejected")
