"""Retry policy independent from task storage."""

from tinyqueue.models import Task


class RetryPolicy:
    def __init__(self, max_attempts: int) -> None:
        self.max_attempts = max_attempts

    def should_retry(self, task: Task) -> bool:
        return task.attempts <= self.max_attempts

    def next_attempt(self, task: Task) -> Task:
        return Task(task.task_id, task.payload, task.attempts + 1)
