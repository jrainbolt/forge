from tinyqueue.retry import RetryPolicy
from tinyqueue.service import QueueService
from tinyqueue.storage import MemoryStorage


def test_failed_task_is_requeued() -> None:
    storage = MemoryStorage()
    service = QueueService(storage, RetryPolicy(3))
    service.submit("job-1", "payload")
    task = service.reserve()
    assert service.fail(task)
    assert service.reserve().attempts == 1
