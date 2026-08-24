# Forge Coding Evaluation

## coding-v1

`coding-v1` is Forge's first controlled, read-only coding benchmark. It runs
eight independent questions against the packaged TinyQueue fixture under
`forge.evaluation`. The suite covers symbol and implementation
localization, implementation explanation, multi-file tracing, defect discovery
and consequence analysis, test coverage, and an architecture boundary.

TinyQueue contains source, tests, and documentation. Its retry policy
deliberately uses an inclusive exhaustion comparison, allowing one additional
attempt. That ground truth exists in task metadata and is not marked in fixture
source.

## Running locally

Use an existing Forge model configuration:

```bash
forge eval \
  --model qwen-small \
  --config ~/Models/forge/forge.toml \
  --suite coding-v1
```

The command prints the controlled workspace before loading the model. Add
`--verbose` to show captured answers and inspected paths. Real model evaluation
is opt-in and is not part of normal CI.

To save a machine-readable result, provide an explicit destination:

```bash
forge eval \
  --model qwen-large \
  --config ~/Models/forge/forge.toml \
  --suite coding-v1 \
  --output eval-results/qwen-large.json
```

Forge never writes a report without `--output`. `eval-results/` is ignored so
local baselines are not accidentally committed.

## Ground truth and scoring

Each immutable task declares its prompt separately from expected source files,
answer facts, file references, symbols, and a coarse tool-call threshold. The
model sees only the prompt and the normal repository-chat system instructions.

Scoring is deterministic and exposes five dimensions:

- correctness awards one point per required fact matched by a declared
  normalized phrase alternative;
- grounding awards one point per required file actually read through A7;
- localization awards one point per expected file and symbol named in the
  answer;
- efficiency awards one point within the task's soft tool threshold;
- completion awards one point when orchestration returns a final answer.

There is no LLM grader, fuzzy semantic grade, embedding search, or fixture
shortcut. Correct prose without the required read activity loses grounding
credit. Thresholds tolerate a small amount of useful exploration while making
large tool-count regressions visible.

## Execution and isolation

The runner loads one selected generic `Model` and reuses it for all tasks. A
single production `RepositoryChatSession` is cleared after every task, so
conversation history and failed-turn state cannot influence the next question.
Individual exceptions become structured task failures and do not abort the run.
All repository access continues through the A7 orchestrator, central executor,
A6 tools, read-only policy, and workspace confinement.

Results contain bounded final answers, concise tool records, unique files
actually read, protocol corrections, orchestration steps, elapsed monotonic
duration, scores, failure information, and backend-reported token usage when
available. Full source contents and the internal model/tool transcript are not
persisted.

## JSON reports and comparison

JSON reports begin with `schema_version: 1` and `suite_version: 1`. They include
run summary metrics and per-task dimensions. Compatibility is explicitly
versioned but not yet promised indefinitely.

Run the identical suite for two profiles and compare task scores, grounding,
tool counts, corrections, timing, and available token usage. These are
controlled local baselines, not claims that one model is universally superior.
Future milestones may compare saved reports with A8 observations; A8 does not
provide a benchmark database or CI quality threshold.

## Limitations

Phrase scoring rewards explicit, benchmark-designed facts rather than semantic
equivalence. Timing varies with local hardware, and token counts are unavailable
when a backend does not report them. The first fixture is intentionally small
and Python-only. It measures read-only repository reasoning before controlled
writes arrive; it does not execute fixture tests or evaluate repairs.

`coding-v1` remains the frozen pre-execution, read-only baseline after A11.
Evaluation runs continue to construct the read-only registry and policy;
assist-mode write, build, and test tools are not exposed, and the fixture is
never mutated or executed.

## coding-write-v1

A11 adds a deliberately small write-capable suite with three one-mutation tasks:
`W01` fixes the retry boundary, `W02` adds a focused test, and `W03` adds a small retry
helper. `CodingWriteEvaluationRunner` copies the committed TinyQueue fixture into a
fresh temporary workspace for every task, composes the production assist registry,
policy, executor, provenance, task state, and orchestration, and requires an explicitly
injected approval callback. Production never auto-approves evaluation mutations.

Scoring is deterministic and checks whether the expected file actually changed,
required content exists, unexpected files stayed byte-identical, exactly one mutation
occurred, current-generation verification succeeded, and the task completed. The
tasks permit one-step solutions and contain no repair loop. Full source and process
output are not retained in results. The original fixture is never modified, and
`coding-v1` task definitions and scoring remain unchanged.

## agent-v1

A12 adds four deterministic agent-loop scenarios while leaving `coding-v1` and
`coding-write-v1` unchanged. `G01` requires multi-step read-only investigation, `G02`
requires multiple reads followed by one targeted mutation, `G03` checks recovery from
an irrelevant search, and `G04` checks truthful handling of a failed verification
without a repair mutation.

`AgentEvaluationRunner` copies TinyQueue into a fresh temporary workspace for every
task and invokes the production agent session with an explicitly injected approval
callback. Scoring uses observable state only: multi-step tool use, expected bytes,
exactly zero or one mutation as appropriate, hard-budget compliance, the expected
machine-readable stop reason, and absence of unexpected file changes. The original
fixture is byte-checked by deterministic tests and never modified. No model grades
answers, no evaluator tool bypass exists, and real-model scores are not CI gates.

## repair-v1

A13 adds four repair scenarios without changing historical suites. `R01` uses a
deterministic test failure followed by successful repair, `R02` uses a syntax failure
followed by successful rebuild, `R03` keeps failing after the sole repair, and `R04`
produces a process-start failure that must not grant repair authority.

Each scenario copies TinyQueue into a fresh temporary workspace and uses production
repair state, tools, provenance, permission checks, and approvals. Scripted responses
deliberately create the first failure; evaluation never relies on a real model making
a predictable mistake. Commands and approvals are injected explicitly.

Scoring checks the initial mutation and failure, repair eligibility, the two-mutation
ceiling, per-operation two-attempt ceiling, truthful final status/stop reason, and
absence of unexpected paths. Tests byte-check the committed fixture. There is no LLM
grader, automatic approval, evaluator-only write path, or cross-task mutation leak.

## Autonomy policy coverage

A14 adds deterministic infrastructure tests rather than another model-quality suite.
They exhaustively exercise the built-in permission matrix, autonomy ceilings,
production tool classifications, immutable snapshots, trusted configured execution,
write approval, and existing AGENT/REPAIR budgets. The frozen `coding-v1`,
`coding-write-v1`, `agent-v1`, and `repair-v1` definitions and scoring are unchanged.

## context-v1

A15 adds five deterministic TinyQueue navigation tasks. C01 locates an exact qualified
method, C02 explains it from a targeted range, C03 finds service reference candidates,
C04 traces focused test coverage, and C05 combines definition, caller, and test
evidence across files. Private ground truth continues to declare required files,
facts, symbols, and tool thresholds without exposing them in prompts.

The suite uses the production read-only session, executor, policy, workspace
confinement, and source-grounding rules. Reports count all tools and inspected files
and now distinguish `repository.read_file` from `repository.read_range`, recording
returned bytes and lines when available. This makes whole-file reads, range reads,
files inspected, context volume, grounding, completion, and tool count directly
comparable. Structural discovery alone receives no source grounding credit.

`find_references` scores bounded structural candidates rather than a perfect semantic
call graph. Normal CI uses scripted models only; real qwen-small runs remain optional
and are not score gates. Historical `coding-v1`, `coding-write-v1`, `agent-v1`, and
`repair-v1` task definitions and scores are unchanged.

Milestone A17 leaves the context-v1 tasks and scoring frozen. Normal CLI repository
chat now supplies the same structural tools with a persistent local index; deterministic
index tests inject temporary cache roots. Index build/refresh metrics are operational
diagnostics only and do not change grounding: only current source reads receive source
evidence credit.

## context-budget-v1

A18 adds five separate tasks without changing `context-v1`: B01 measures exact-symbol
narrowing, B02 discourages inefficient whole-file context, B03 traces references with
targeted reads, B04 exposes repeated broad-search behavior, and B05 checks final-answer
survival under compaction. Reports add estimated context admitted/dropped and peak,
compaction and rejection counts, final remaining budget, and whole-file/range-read
counts. Scoring remains deterministic and source-grounded; no model grader or semantic
retrieval is introduced. Large-file admission behavior is exercised through generated
temporary fixtures in deterministic tests rather than committed filler.
