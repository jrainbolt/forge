from pyservice.headers import normalize_header_name


def rejects(value: str) -> bool:
    try:
        normalize_header_name(value)
    except ValueError:
        return True
    return False


assert normalize_header_name("Content-Type") == "content-type"
assert normalize_header_name(" X-Trace-Id ") == "x-trace-id"
assert rejects("Bad Header")
assert rejects("x_header")
