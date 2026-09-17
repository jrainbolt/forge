from pyservice.models import Response
from pyservice.storage import ResponseStore

store = ResponseStore()
store.put("a", Response(200, b"ok"))
assert store.get("a") == Response(200, b"ok")
assert store.get("missing") is None
