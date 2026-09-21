from pyservice.models import Response
from pyservice.response_wire import wire_response

assert wire_response(Response(200, b"hi\x00!")) == b"200:4:hi\x00!"
assert wire_response(Response(204, b"")) == b"204:0:"

try:
    wire_response(Response(99, b""))
except ValueError:
    pass
else:
    raise AssertionError("invalid status should be rejected")
