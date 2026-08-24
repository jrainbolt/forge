"""The stable coding-v1 task suite and controlled fixture location."""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path

from forge.evaluation.types import EvaluationTask, RequiredFact, TaskCategory

CODING_V1 = "coding-v1"
CONTEXT_V1 = "context-v1"
CONTEXT_SUITE_VERSION = 1


def fixture_workspace() -> Path:
    """Locate TinyQueue as deliberate installed evaluation package data."""
    resource = files("forge.evaluation").joinpath("fixtures", "eval_repo")
    workspace = Path(str(resource))
    if not workspace.is_dir():
        raise RuntimeError("installed TinyQueue evaluation fixture is unavailable")
    return workspace


def _fact(*alternatives: str) -> RequiredFact:
    return RequiredFact(alternatives)


CODING_V1_TASKS = (
    EvaluationTask(
        "E01",
        TaskCategory.LOCALIZATION,
        "Where is the Task data model defined? Identify its file and class.",
        ("src/tinyqueue/models.py",),
        (_fact("task data model", "class task", "task is defined"),),
        ("src/tinyqueue/models.py",),
        ("Task",),
        3,
    ),
    EvaluationTask(
        "E02",
        TaskCategory.LOCALIZATION,
        "Which implementation file owns FIFO removal, and which method performs it?",
        ("src/tinyqueue/storage.py",),
        (_fact("fifo", "popleft"),),
        ("src/tinyqueue/storage.py",),
        ("take_next",),
        3,
    ),
    EvaluationTask(
        "E03",
        TaskCategory.EXPLANATION,
        "Explain how QueueService.fail handles a failed task.",
        ("src/tinyqueue/service.py",),
        (
            _fact("next_attempt", "increments", "incremented"),
            _fact("should_retry", "retry policy"),
            _fact("append", "requeue", "re-queue"),
        ),
        ("src/tinyqueue/service.py",),
        ("fail",),
        4,
    ),
    EvaluationTask(
        "E04",
        TaskCategory.TRACE,
        "Trace a submitted task from service submission through storage and retry "
        "scheduling after failure.",
        (
            "src/tinyqueue/service.py",
            "src/tinyqueue/storage.py",
            "src/tinyqueue/retry.py",
        ),
        (
            _fact("submit"),
            _fact("append"),
            _fact("next_attempt"),
            _fact("should_retry"),
        ),
        (
            "src/tinyqueue/service.py",
            "src/tinyqueue/storage.py",
            "src/tinyqueue/retry.py",
        ),
        ("QueueService", "MemoryStorage", "RetryPolicy"),
        6,
    ),
    EvaluationTask(
        "E05",
        TaskCategory.BUG_FIND,
        "Retry exhaustion permits one attempt too many. Locate the defect and "
        "identify the incorrect condition.",
        ("src/tinyqueue/retry.py",),
        (
            _fact("<=", "less than or equal"),
            _fact(
                "should use <",
                "change <= to <",
                "strictly less",
                "correct condition should be `task.attempts <",
                "correct condition should be task.attempts <",
            ),
        ),
        ("src/tinyqueue/retry.py",),
        ("should_retry",),
        4,
    ),
    EvaluationTask(
        "E06",
        TaskCategory.BUG_EXPLAIN,
        "Explain the behavioral consequence of the retry exhaustion defect, "
        "including the boundary case.",
        ("src/tinyqueue/retry.py", "src/tinyqueue/service.py"),
        (
            _fact("one extra", "additional retry", "extra attempt"),
            _fact("max_attempts", "maximum attempts"),
            _fact("requeue", "re-queue", "append"),
        ),
        ("src/tinyqueue/retry.py", "src/tinyqueue/service.py"),
        ("should_retry", "fail"),
        5,
    ),
    EvaluationTask(
        "E07",
        TaskCategory.TEST_COVERAGE,
        "Which tests cover retry exhaustion behavior, and what boundary do they "
        "assert?",
        ("tests/test_retry.py",),
        (
            _fact("test_retry_stops_at_limit"),
            _fact(
                "attempts=3",
                "attempts = 3",
                "at the limit",
                "equals the retry limit",
                "attempt count equals",
                "with 3 attempts",
            ),
        ),
        ("tests/test_retry.py",),
        ("test_retry_stops_at_limit",),
        4,
    ),
    EvaluationTask(
        "E08",
        TaskCategory.ARCHITECTURE,
        "How does TinyQueue separate storage mechanics from retry policy while "
        "coordinating both?",
        (
            "src/tinyqueue/service.py",
            "src/tinyqueue/storage.py",
            "src/tinyqueue/retry.py",
        ),
        (
            _fact("memorystorage", "memory storage"),
            _fact("retrypolicy", "retry policy"),
            _fact("queueservice", "queue service"),
        ),
        (
            "src/tinyqueue/service.py",
            "src/tinyqueue/storage.py",
            "src/tinyqueue/retry.py",
        ),
        ("MemoryStorage", "RetryPolicy", "QueueService"),
        6,
    ),
)

CONTEXT_V1_TASKS = (
    EvaluationTask(
        "C01",
        TaskCategory.CONTEXT,
        "Where is RetryPolicy.should_retry defined? Give its exact file and symbol.",
        ("src/tinyqueue/retry.py",),
        (_fact("retrypolicy.should_retry", "should_retry"),),
        ("src/tinyqueue/retry.py",),
        ("RetryPolicy.should_retry",),
        3,
    ),
    EvaluationTask(
        "C02",
        TaskCategory.CONTEXT,
        "Explain RetryPolicy.should_retry using targeted source context, without "
        "reading unrelated files.",
        ("src/tinyqueue/retry.py",),
        (_fact("<=", "less than or equal"), _fact("max_attempts")),
        ("src/tinyqueue/retry.py",),
        ("RetryPolicy.should_retry",),
        3,
    ),
    EvaluationTask(
        "C03",
        TaskCategory.CONTEXT,
        "Where is RetryPolicy.should_retry referenced? Identify structural candidates.",
        ("src/tinyqueue/service.py",),
        (_fact("service.py"), _fact("fail")),
        ("src/tinyqueue/service.py",),
        ("should_retry",),
        4,
    ),
    EvaluationTask(
        "C04",
        TaskCategory.CONTEXT,
        "Which tests reference RetryPolicy.should_retry or exercise its boundary?",
        ("tests/test_retry.py",),
        (_fact("test_retry_stops_at_limit"), _fact("attempts=3", "attempts = 3")),
        ("tests/test_retry.py",),
        ("should_retry",),
        4,
    ),
    EvaluationTask(
        "C05",
        TaskCategory.CONTEXT,
        "Trace RetryPolicy.should_retry from its definition to the service caller "
        "and the focused test using structural lookups and targeted reads.",
        (
            "src/tinyqueue/retry.py",
            "src/tinyqueue/service.py",
            "tests/test_retry.py",
        ),
        (_fact("retrypolicy"), _fact("queueservice", "service"), _fact("test_retry")),
        (
            "src/tinyqueue/retry.py",
            "src/tinyqueue/service.py",
            "tests/test_retry.py",
        ),
        ("should_retry",),
        7,
    ),
)


def load_suite(name: str) -> tuple[EvaluationTask, ...]:
    if name == CODING_V1:
        return CODING_V1_TASKS
    if name == CONTEXT_V1:
        return CONTEXT_V1_TASKS
    raise ValueError(
        f"unknown evaluation suite {name!r}; available: {CODING_V1}, {CONTEXT_V1}"
    )
