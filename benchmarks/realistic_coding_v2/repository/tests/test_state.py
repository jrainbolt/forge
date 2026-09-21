from pyservice.state import JobState, can_transition

assert can_transition(JobState.QUEUED, JobState.RUNNING)
assert can_transition(JobState.RUNNING, JobState.SUCCEEDED)
assert not can_transition(JobState.SUCCEEDED, JobState.RUNNING)
assert not can_transition(JobState.FAILED, JobState.RUNNING)
assert not can_transition(JobState.CANCELLED, JobState.RUNNING)
