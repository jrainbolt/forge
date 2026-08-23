"""In-memory FIFO storage."""

from collections import deque

from tinyqueue.models import Task


class MemoryStorage:
    def __init__(self) -> None:
        self._pending: deque[Task] = deque()

    def append(self, task: Task) -> None:
        self._pending.append(task)

    def take_next(self) -> Task | None:
        return self._pending.popleft() if self._pending else None

    def __len__(self) -> int:
        return len(self._pending)
