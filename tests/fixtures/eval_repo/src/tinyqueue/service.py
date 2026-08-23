"""Application service coordinating storage and retry policy."""

from tinyqueue.models import Task
from tinyqueue.retry import RetryPolicy
from tinyqueue.storage import MemoryStorage


class QueueService:
    def __init__(self, storage: MemoryStorage, retries: RetryPolicy) -> None:
        self._storage = storage
        self._retries = retries

    def submit(self, task_id: str, payload: str) -> None:
        self._storage.append(Task(task_id, payload))

    def reserve(self) -> Task | None:
        return self._storage.take_next()

    def fail(self, task: Task) -> bool:
        attempted = self._retries.next_attempt(task)
        if not self._retries.should_retry(attempted):
            return False
        self._storage.append(attempted)
        return True
