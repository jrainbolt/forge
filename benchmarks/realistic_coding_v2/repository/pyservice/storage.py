from __future__ import annotations

from pyservice.models import Response


class ResponseStore:
    def __init__(self) -> None:
        self._values: dict[str, Response] = {}

    def put(self, request_id: str, response: Response) -> None:
        self._values[request_id] = response

    def get(self, request_id: str) -> Response | None:
        return self._values.get(request_id)
