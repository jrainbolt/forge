from pyservice.models import Request
from pyservice.service import handle

assert handle(Request("r1", b"hello")).status == 200
assert handle(Request("", b"hello")).status == 400
assert handle(Request("r2", b"")).status == 204
