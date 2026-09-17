from __future__ import annotations

from collections import deque

from pyservice.models import Request


class RequestQueue:
    def __init__(self) -> None:
        self._values: deque[Request] = deque()

    def push(self, request: Request) -> None:
        self._values.append(request)

    def pop(self) -> Request | None:
        return self._values.popleft() if self._values else None
