from tinyqueue.models import Task
from tinyqueue.retry import RetryPolicy


def test_retry_before_limit() -> None:
    assert RetryPolicy(3).should_retry(Task("x", "work", attempts=2))


def test_retry_stops_at_limit() -> None:
    assert not RetryPolicy(3).should_retry(Task("x", "work", attempts=3))
