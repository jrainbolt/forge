from tinyqueue.models import Task
from tinyqueue.storage import MemoryStorage


def test_storage_is_fifo() -> None:
    storage = MemoryStorage()
    storage.append(Task("first", "one"))
    storage.append(Task("second", "two"))
    assert storage.take_next().task_id == "first"
