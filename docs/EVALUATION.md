# Forge Coding Evaluation

## coding-v1

`coding-v1` is Forge's first controlled, read-only coding benchmark. It runs
eight independent questions against the committed TinyQueue fixture at
`tests/fixtures/eval_repo`. The suite covers symbol and implementation
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

`coding-v1` remains the frozen pre-execution, read-only baseline after A10.
Evaluation runs continue to construct the read-only registry and policy;
assist-mode write, build, and test tools are not exposed, and the fixture is
never mutated or executed.
