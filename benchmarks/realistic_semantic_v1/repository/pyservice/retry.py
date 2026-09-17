from dataclasses import dataclass


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int
    base_delay_ms: int = 100


def should_retry(attempt: int, status: int, policy: RetryPolicy) -> bool:
    """Return whether a failed zero-based attempt may be followed by another."""
    if attempt < 0 or policy.max_attempts < 1:
        return False
    retryable = status == 429 or 500 <= status < 600
    return retryable and attempt + 1 < policy.max_attempts


def retry_delay_ms(attempt: int, policy: RetryPolicy) -> int:
    return policy.base_delay_ms * (2 ** max(0, attempt))
