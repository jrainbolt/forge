from pathlib import Path

required = (
    "pyservice/retry.py",
    "pyservice/config.py",
    "cengine/parser.c",
    "cengine/quota.c",
)
assert all(Path(path).is_file() for path in required)
