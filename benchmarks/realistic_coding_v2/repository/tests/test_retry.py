from pyservice.retry import RetryPolicy, retry_delay_ms, should_retry

policy = RetryPolicy(max_attempts=3)
assert should_retry(0, 500, policy)
assert should_retry(1, 503, policy)
assert not should_retry(2, 500, policy)
assert not should_retry(0, 404, policy)
assert retry_delay_ms(2, policy) == 400
