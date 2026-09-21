# Service Engine Fixture

This mixed Python/C repository models request handling, configuration, retry,
validation, parsing, quotas, state transitions, and accounting. The benchmark uses
disposable copies only. The v2 project additionally includes scheduling, response
framing, dispatch, billing, retry backoff, and health-policy modules. Its task
set covers existing edits, coordinated edits, new modules, and mixed edit/create
transactions. Task-specific reference and wrong-state metadata stay outside
model-facing workspaces.
