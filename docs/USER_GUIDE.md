# Forge practical user guide

Set the local configuration once in each terminal:

```bash
export FORGE_CONFIG="$HOME/Models/forge/forge.toml"
```

## Modes

- `forge chat --model PROFILE` starts tool-free chat.
- `forge repo PATH --model PROFILE` answers read-only repository questions.
- `forge assist PATH` permits one approval-gated coding mutation and configured
  verification.
- `forge agent PATH` permits bounded multi-step work with the same approval
  boundary.
- A bare `forge` presents these choices interactively.

Coding modes use the configured coding default, normally `qwen-large`. Chat and
read-only repository use only an explicit profile or a configured role default.
Pass `--model qwen-small` to override selection for one session.

## Approval and verification

Mutation prompts identify the tool and affected path, then display the exact diff.
Grouped changes display all child diffs together. The default answer is no.
Execution approval is separate and displays the immutable command arguments,
workspace, and timeout. A successful write is not reported as verified until the
configured verification plan completes. Operational executable-check errors fail
open; a proven behavioral failure is reported as verification failure.

## Sessions

Follow-up questions retain conversation and repository context in the current
process. Use `/info` for the active model and mode, `/clear` for a clean
conversation, and `/exit` to close Forge. Persistence and cross-process resume are
not currently implemented.

## Manual test guide

A. Normal chat:

```bash
forge chat --model qwen-small
```

B. Repository explanation:

```bash
forge repo . --model qwen-small
# Ask: Explain the main command-line entry point.
```

C. Read-only repository question:

```bash
forge repo . --model qwen-small
# Ask: Where is model selection resolved? Do not change files.
```

D. Small approved coding edit in a disposable checkout:

```bash
forge assist . --edit-path path/to/file.py
# Ask naturally for one small change; inspect the diff before entering y.
```

E. Test/build request:

```bash
forge agent .
# Ask: Run the configured tests and explain any failure. Approve execution only
# after checking the displayed argv and workspace.
```

F. Explicit qwen-small override:

```bash
forge assist . --model qwen-small --edit-path path/to/file.py
```

G. Start a fresh session:

```text
/clear
```

Start a completely new process with `/exit` followed by the desired `forge`
command.

## Troubleshooting

- “requires --config or FORGE_CONFIG”: set `FORGE_CONFIG` or pass `--config`.
- Unknown/unavailable model: inspect configured profile names and local model paths.
- Invalid repository: pass an existing directory.
- A denied mutation remains unapplied; rerun the request if you want a new proposal.
- Verification failures leave the approved workspace mutation visible and report
  the failed phase. Inspect it before requesting a refinement.
- Use the global `--verbose` option before the command for diagnostic logging:
  `forge --verbose assist .`.
