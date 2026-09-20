# Forge Coding Evaluation

## coding-v1

`coding-v1` is Forge's first controlled, read-only coding benchmark. It runs
eight independent questions against the explicitly packaged TinyQueue runtime
resource under `forge.evaluation`. This resource is required by the documented
installed `forge eval` default and exported `fixture_workspace()` lookup; no other
evaluator fixture is permitted in the wheel or sdist. The suite covers symbol and implementation
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

## semantic-v1

A19 defines conceptual navigation scenarios for approval policy, context planning,
repair eligibility, persistent indexing, and configured project execution. Semantic
metrics distinguish whether conceptual retrieval was used, its top-k candidates, and
which candidates were subsequently read as current source. Deterministic CI uses the
mock embedding model; real local embedding and qwen-small runs are optional smoke
evidence, not historical-suite score gates. Existing suites remain frozen.

## context-budget-v1

A18 adds five separate tasks without changing `context-v1`: B01 measures exact-symbol
narrowing, B02 discourages inefficient whole-file context, B03 traces references with
targeted reads, B04 exposes repeated broad-search behavior, and B05 checks final-answer
survival under compaction. Reports add estimated context admitted/dropped and peak,
compaction and rejection counts, final remaining budget, and whole-file/range-read
counts. Scoring remains deterministic and source-grounded; no model grader or semantic
retrieval is introduced. Large-file admission behavior is exercised through generated
temporary fixtures in deterministic tests rather than committed filler.
# Retrieval evaluation v1

`retrieval-v1` is a deterministic six-task harness for repair eligibility,
context compaction, approval policy, persistent repository indexing, configured
project execution, and workspace path confinement. It compares raw semantic and
hybrid-reranked ranks without invoking a generation model.

For each task it records the first expected-file rank, top-1/top-3/top-5 success,
generated-metadata contamination, documentation-before-implementation ordering,
and distinct-file diversity. Aggregate metrics report both raw and reranked
values so improvements and regressions remain visible. The harness accepts a
`SemanticIndex` directly through `evaluate_retrieval`; its task definitions and
scoring are fixed and model-independent, while acceptance may be run against a
real configured embedding artifact.

## routing-v1

`routing-v1` fixes six deterministic workflow cases: semantic-to-read routing,
exact-symbol narrowing, repeated candidate-set detection, read-failure
reopening, multi-file inspection, and broad-search suppression after sufficient
source. Router metrics count transitions, discovered/inspected/failed
candidates, broad and targeted operations, suppressed attempts, repeated sets,
source acquisition, and queue truncation without retaining source contents.

CI uses scripted results and dynamic-schema assertions. Real qwen-small runs
remain opt-in observations used to compare unrestricted behavior; they do not
alter deterministic suite scores.

The completed runner sends scripted `MockModel` structured responses through
the production `RepositoryChatSession`, dynamic routed schemas,
`RetrievalStrategy`, `ToolExecutor`, and real tools operating on temporary
repositories. Its six tasks aggregate attempted/executed/suppressed discovery,
candidate inspection and failure, repeated candidate sets, acquired source
files, completion, final state, and tool count.

These measurements distinguish three boundaries. Routing correctness concerns
whether candidate state narrows and reopens tools deterministically. Retrieval
relevance concerns which candidates lexical, structural, or semantic retrieval
returns. Generative interpretation quality concerns whether a model correctly
understands valid inspected source. Unrestricted qwen-small observations showed
that A21 removed broad-discovery and repeated-invalid-range loops, while nearby
source interpretation and conceptual coverage remain model/retrieval-quality
limitations rather than router correctness failures.

## coverage-v1

`coverage-v1` runs six deterministic tasks for single and independent goals,
relationship dependencies, wrong-goal isolation, premature-final redirection,
and generation invalidation. Scripted production sessions exercise A21 routing,
A18 admission, real temporary repository tools, explicit evidence plans, and
the finalization gate. Metrics report required/covered goals, coverage ratio,
premature finals, source reads, distinct paths, and completion without model
grading.

The production-decomposition DEC01–DEC04 subset covers relationship planning,
explicit-item planning, wrong-goal isolation, and required-goal exhaustion. It
reports plan counts, total/covered/failed goals, premature finals, transitions,
wrong-goal reads, and complete tasks. These metrics do not grade model prose.

## bootstrap-v1

`bootstrap-v1` runs B01–B06 through normal production decomposition, orchestration,
`ToolExecutor`, a `MockEmbeddingModel` semantic index, A20/A21, and model-directed
source reads. It covers single and sequential goals, broad-search suppression,
empty-result fallback, unavailable semantics, and candidate exhaustion without a
repeat bootstrap. Metrics separate automatic executions/candidates from subsequent
model discovery calls. Bootstrap correctness does not imply candidate relevance,
good model candidate choice, or correct source interpretation.

## finalization-v1

`finalization-v1` runs F01–F06 through normal repository orchestration. It covers
single and multi-goal final-only transitions, a blocked post-coverage tool call,
goal-balanced synthesis evidence, stale-source invalidation and reread, and failed
required-goal non-finalization. Metrics report entries, final model calls, bounded
protocol corrections, prevented/executed post-coverage tools, represented required
goals, and grounded completions. These measurements establish workflow locking and
evidence availability, not semantic answer correctness.

Evidence coverage is separate from retrieval relevance and model interpretation:
it establishes that each explicit facet has associated trusted evidence, not
that the retrieved source was ideal or the generated explanation is correct.

## realworld-v1

`realworld-v1` is an opt-in realistic coding baseline, not a CI model-quality
gate. Its eight fixed tasks measure repository reasoning (E01–E03), single-file
coding (E04–E06), bounded repair (E07), and the current multi-file capability
ceiling (E08). Read-only tasks run seeds 7 and 42; mutation tasks use seed 42.

The source repository is copied without `.git`, build products, or caches into a
fresh temporary directory for every run. Evaluator-owned defect transformations
require one exact source match. File hashes before and after Forge detect created,
modified, and deleted paths independently of Git. The source repository is hashed
before and after the suite and any canonical change is an infrastructure failure.

Writes are approved only when their `MutationPreview.path` is declared by the task.
Build and test approvals require the exact evaluator-configured argument array,
workspace, and timeout. An independent evaluator oracle may subsequently execute
fixed build/test arrays with `shell=False` inside the disposable copy. Forge
verification, model completion, evaluator infrastructure, and oracle outcome are
reported separately.

Statuses are `PASS`, `PARTIAL`, `FAIL`, and `UNSUPPORTED`. The bounded failure
taxonomy is model quality, retrieval, context, protocol, tool limit, mutation,
verification, repair, unsupported capability, and infrastructure. No LLM judge is
used. Reports contain paths, hashes, metrics, timing, seed, and bounded failure text,
not repository source excerpts.

Run the local Foundation C baseline with a user-owned repository and model paths:

```bash
PYTHONPATH=src python scripts/run_realworld_v1.py \
  --repository /path/to/foundation \
  --config /path/to/forge.toml \
  --model qwen-small \
  --embedding-config /path/to/embeddings.toml \
  --embedding-profile nomic \
  --output eval-results/realworld-v1-qwen-small.json
```

Real-model results measure current behavior; poor scores do not fail deterministic
CI. A `PASS` for A25 means the measurement is isolated and trustworthy, not that
Forge is an everyday production coding agent. Fresh absolute workspace identities
can make every semantic index cold; `--skip-semantic-index` exists only for an
explicitly documented lexical-fallback measurement after cold-index cost has been
measured separately.

## discovery-v1

`discovery-v1` is the model-free A26 suite. D01-D08 cover C implementation and
header discovery, Java implementation discovery, configuration/test/documentation
preferences, exact incremental refresh accounting, generated-directory exclusion,
and third-party classification. It records ranked paths without treating them as
source evidence.

The local Foundation A25→A26 comparison preserves the fixed A25 task definitions,
seeds, prompts, budgets, disposable-copy rules, and independent oracle. Reports add
the selected bootstrap provider, expected implementation acquisition, lexical cold
and warm index timing, postings, and search latency. The A25 baseline is retained;
A26 results are a comparison rather than a replacement. Foundation and model runs
remain local and are not CI gates.

## mutation-transition-v1

`mutation-transition-v1` is the model-free A27 production-orchestration suite.
M01-M06 cover source-backed phase entry and approved mutation, suppression of broad
discovery, premature-final correction, one bounded candidate reread, stale-source
invalidation and reacquisition, and write-policy denial. Metrics distinguish
mutation-ready entry, model calls, proposals, post-ready discovery attempts,
targeted rereads, approvals, successful mutations, and tools before transition.

The unchanged Foundation E04 comparison separates retrieval success from coding
action: A26 acquired source but exhausted context before mutation; A27 reports
whether mutation readiness and an actual patch proposal were reached independently
from patch correctness, verification, and the evaluator oracle.

## structured-mutation-v1

`structured-mutation-v1` is the fixed eight-case A28 production-orchestration
suite. P01 covers exact replacement; P02 bounded mismatch recovery; P03 ambiguity;
P04 stale-source invalidation; P05 range confinement; P06 proposal bounds; P07
exact preview rejection; and P08 reuse during bounded repair. The cases use
`MockModel`, real temporary files, A9 previews and approval, `ToolExecutor`, and
the normal verification/repair handoff.

The proposal ladder separately reports mutation-ready, structured response,
validation, preview, approval, mutation, verification, and oracle outcomes. A
valid but logically incorrect replacement is model quality; failure to validate
or faithfully preview a valid replacement is mutation mechanics.

On Foundation E04, A27 found `src/clock.c` and reached mutation-ready but its raw
patch proposal failed before preview. A28 removes unified-diff construction from
the model-facing response without increasing model, context, or tool budgets.

## verification-gate-v1

`verification-gate-v1` is the fixed eight-case A29 production-orchestration suite.
V01 verifies immediate `ALLOW`; V02 the existing `ASK` approval path; V03 `DENY`;
V04 absent project verification; V05 a nonzero test outside repair; V06 failure,
fresh diagnosis, one repair, and re-verification; V07 explicit trusted-caller skip;
and V08 that a final model response cannot bypass the post-mutation gate.

The suite uses `MockModel`, A27/A28 mutation mechanics, real temporary files,
mutation previews and approvals, `ToolExecutor`, and configured subprocess argument
arrays with `shell=False`. It records verification-ready entries, provider,
permission, approval, execution, result, verification tools, and intervening reads.
V08 also runs at the normal near-budget boundary to reserve one legitimate
verification execution without increasing any configured tool limit.

Real-world reports include the same gate metrics. A logically incorrect patch that
Forge successfully tests is classified separately from failure to execute required
verification. Foundation data and generated real-world JSON remain local and are
not included in distributions.

## repair-grounding-v1

`repair-grounding-v1` is the fixed eight-case A30 production-orchestration suite.
R01 establishes trusted diagnostics and automatic fresh-source readiness; R02 rejects
stale primary text; R03 inspects the prioritized repair request; R04 suppresses broad
post-failure discovery; R05 completes a structured repair and passing reverification;
R06 proves a second verification failure is terminal; R07 excludes process-launch
failure from repair; and R08 treats diagnostic prompt injection as inert data.

The cases use `MockModel`, real configured subprocesses, temporary workspaces, A28
structured validation and previews, A29 automatic verification, and the production
A13 repair limit. Metrics report diagnosis entry, diagnostic registration, fresh
source, repair readiness, proposal, preview, mutation, reverification execution, and
reverification result without persisting source or full diagnostics.

The unchanged Foundation E07 task is the real A30 observation. Its ladder separates
primary retrieval/reasoning and mutation mechanics from initial verification,
diagnosis grounding, repair proposal mechanics, repair correctness, reverification,
and the independent oracle. Real model quality remains observational rather than a CI
gate.

## edit-intent-v1

`edit-intent-v1` is the fixed eight-case A31 production evaluation. I01 covers a
normal changed proposal; I02 one no-op followed by a changed correction; I03 two
no-ops and truthful termination; I04 the independent materialized-no-delta defense;
I05 mutation snapshot intent/source fidelity; I06 retrieval-chatter exclusion; I07
equivalent repair recovery and automatic reverification; and I08 inert repository
instructions with an unchanged trusted target.

The suite records mutation-intent requests, structured attempts, no-op attempts and
corrections, correction successes, non-no-op proposals, materialized deltas,
previews, and mutations without recording source content. `realworld-v1` additionally
reports no-op proposal count, whether structured recovery was used, and whether an
actual delta was proposed while preserving historical scoring.

Foundation E07 previously reached `MUTATION_READY` but returned two identical
replacements, blocking A30 before verification. The unchanged A31 comparison asks
whether explicit task/source anchoring produces a non-no-op proposal. That measures
mechanical intent expression separately from semantic patch correctness. If a model
still returns two no-ops with correct retrieval, target, source, schema, and bounded
correction, further work should compare model capability instead of stacking more
prompt or state heuristics.

The accepted A31 Foundation E07 seed-42 run produced a valid non-no-op primary
proposal without correction, created an approved preview, mutated `src/clock.c`, and
reached verification. Its failed verification then supplied the previously missing
real A30 observation: trusted diagnosis, automatic fresh source, `REPAIR_READY`, a
valid approved repair mutation, and automatic reverification. Reverification and the
independent oracle failed, isolating the remaining problem as semantic patch quality
rather than mutation mechanics or repair orchestration.

## model-capability-v1

`model-capability-v1` compares configured local generative profiles by running the
existing `realworld-v1` task objects through the unchanged production
`RepositoryChatSession`. The matrix varies only profile and fixed seed. Retrieval,
context limits, generation settings, permissions, approval, tool and mutation
budgets, verification, repair, disposable-copy isolation, and the independent oracle
remain identical.

Profiles come exclusively from `ModelCatalog`; missing artifacts, load failures, or
failed smoke generations are recorded as `UNAVAILABLE` with a bounded stage,
exception category, and message. No models are downloaded, and a load failure skips
that profile's tasks without aborting other profiles. Models are loaded and closed
sequentially.

Per-run results retain source acquisition, grounding, mutation-ready, no-op,
structured-delta, preview, mutation, verification, repair, oracle, tool, model-call,
context, and elapsed metrics. Per-model aggregates report grounded read-only runs,
valid-delta, preview, mutation, verification-pass, oracle-pass, and repair-success
rates. This keeps mechanical patch production distinct from semantic correctness and
uses no LLM judge.

The accepted Foundation seed-42 matrix evaluated `qwen-small` and `qwen-large` on
unchanged E01, E03, E04, and E07 definitions with lexical cold bootstrap and 8192
context. Both profiles grounded E01/E03. `qwen-small` produced valid previews and
mutations for both coding tasks but passed neither verification nor oracle;
`qwen-large` produced no structured delta for either coding task and therefore never
reached verification. The larger configured profile was slower overall. The result
confirms model/profile capability is a major variable, but does not show that the
available stronger profile resolves Forge's coding-quality limitation.

## protocol-compatibility-v1

`protocol-compatibility-v1` separates model behavior into four layers using
evaluation prompt version `protocol-diagnostic-v1`. L1 requests a concise conceptual
answer with no schema or tools. L2 requests only evaluation-owned `old_text` and
`new_text` JSON fields. L3 uses the actual production mutation-ready output builder
and A28 validator without repository tools. L4 is unchanged `realworld-v1` E04
production orchestration. L1–L3 prompts are diagnostic-only and are not used by
Forge sessions.

Three synthetic tasks cover a C boundary comparison, Python boolean inversion, and
a C default constant. Disposable compiler/interpreter checks distinguish valid
representation and material delta from semantic oracle success. Foundation L1–L3
use the unchanged E04 task and an evaluator-established expected source inside
disposable copies; L4 performs normal retrieval. Per-model aggregates report each
layer independently and compute transition loss only across task IDs shared by the
adjacent layers.

At seed 42, qwen-small identified all four concepts and formed mechanically valid L2
edits for all tasks, but its Foundation L2 edit failed the oracle. Foundation L3
then failed exact-source matching, while L4 again produced a valid but semantically
wrong delta. qwen-large identified all four concepts and produced correct L2/L3
edits on most synthetic tasks, but on Foundation E04 it could not form L2 fields,
emitted schema-invalid L3 output, and produced no L4 delta. The comparison localizes
qwen-small primarily to semantic edit choice and qwen-large to realistic edit
representation/protocol compatibility rather than basic conceptual reasoning.

## edit-representation-v1

`edit-representation-v1` holds model, task wording, complete current source, seed 42,
temperature 0, 512-token output limit, and deterministic semantic oracle constant
while varying only an evaluation-owned edit representation. R1 reuses A33's exact
`old_text`/`new_text` evaluator. R2 presents stable line-number decoration and accepts
a strict, bounded `start_line`, `end_line`, and `new_text` object. R3 partitions the
same source into three to eight deterministic contiguous regions with opaque `S1`,
`S2`, ... labels and accepts a strict `span_id` and `new_text` object. Short synthetic
files may necessarily expose fewer than three non-empty spans.

R2 and R3 validate one bounded contiguous target and replacement, preserve all
unchanged prefix and suffix content, materialize only in disposable evaluator copies,
and never enter the production mutation protocol. Results record response validity,
target selectability, evaluator-known target-region accuracy, replacement validity,
material delta, semantic oracle outcome, and generation-only latency separately.
Per-model representation aggregates expose transparent R2-minus-R1 and R3-minus-R1
oracle-rate gains; no composite score or model judge is used.

At seed 42, both models selected the correct target region for every R2/R3 task.
qwen-small's Foundation E04 edit failed the unchanged oracle under R1, R2, and R3,
confirming that easier targeting did not fix its semantic replacement. qwen-large
failed to express an E04 R1 exact edit, but its R2 line-range edit selected the right
line and passed the unchanged oracle. Its R3 span edit selected the right span but
failed semantically. This is direct evidence of exact-copy friction for qwen-large on
E04 and supports investigating a separately designed safe range adapter later without
weakening A28 or changing production behavior in A34.

## line-range-mutation-v1

`line-range-mutation-v1` is the fixed eight-case A35 production-orchestration
evaluation. It covers one-line and multi-line replacement, invalid and unauthorized
range correction, stale source, no-op rejection, approval binding, and repair plus
automatic reverification. It records representation-specific attempts, validity,
target validity, materialization, corrections, previews, and mutations without
retaining source contents. `structured-mutation-v1` remains the exact-text regression
suite.

A34 showed that qwen-large could solve unchanged Foundation E04 when the evaluator
used line ranges despite failing the exact-text representation. A35 therefore adds
an explicitly selected production adapter: Forge derives exact old text from trusted
current source and converges into the unchanged A28/A9 authority path.

The A35 production-expression isolation found that the original mutation-ready
request retained a read-only system prompt encouraging final answers and raw patch
calls, despite a line-range schema. A direct production-schema E04 diagnostic
produced a valid oracle-passing edit; the original production-style snapshot
returned `final`; a mutation-specific system snapshot again produced a valid
oracle-passing edit. Forge now uses representation-neutral mutation-ready system
framing and a line-range-specific correction for premature finals, without changing
the schema, validator, mutation authority, or retry count.

The one subsequent unchanged qwen-large E04 seed-42 production run selected lines
16–19, passed line-range validation, created an exact approved preview, and made
one mutation. The independent rebuild-and-test oracle passed and canonical
Foundation remained unchanged. The configured immediate `project.test` failed,
so Forge truthfully recorded `mutated_verification_failed` rather than claiming
verified completion. This meets A35's minimum and strong adapter gates, while
leaving the immediate verification outcome explicit.

## verification-attribution-v1

`verification-attribution-v1` runs eight deterministic cases through temporary
workspaces and real configured subprocesses. A01 covers baseline pass followed by
failure; A02 a matching pre-existing failure; A03 different failure markers; A04
passing verification; A05 unavailable baseline; A06 changed trusted command; A07
repair suppression for a matching baseline failure; and A08 preservation of the
existing repair path for a mutation-associated failure. Separate fingerprint tests
cover workspace roots, ANSI, distinct failures, timeout, process launch, and
incomplete output. No model interprets logs.

The explicit `--verification-baseline` option runs one extra permission-controlled
verification before a realworld coding task; ordinary tasks do not do this by
default. Reports separate baseline execution, baseline/post duration, attribution,
fingerprint equality, extra tool executions, and extra elapsed time. The independent
oracle remains evaluator-only and never overrides Forge verification.

The A35 Foundation E04 mismatch was reproduced using two disposable copies of the
same canonical repository. After setup and CMake configuration but before building,
the configured `ctest --test-dir build --output-on-failure` exits 8 because all 58
test executables are missing. The clean baseline and exact A35-corrected copy have
the same bounded failure fingerprint. A separate evaluator rebuild-and-test oracle
passes after the correction. Attribution is therefore `PREEXISTING_OR_UNRELATED`,
while configured verification remains failed and never constitutes verified success.

## verification-plan-v1

A37 adds eight deterministic full-orchestration cases using real A10 subprocesses
in disposable workspaces. P01 passes build then test with no inter-step model
decision. P02 stops after failed build. P03 preserves a failing test after a
passing build. P04 blocks denied build. P05 rejects a separate ASK on test. P06
uses the exact four-slot read, mutation, build, test budget. P07 repairs once and
reruns the whole plan. P08 binds an explicit A36 baseline to the same build→test
plan and suppresses repair for the matching pre-existing test failure.

Realworld metrics now include plan ID, required/started/passed step counts, failed
step, outcome, tool count, duration, and zero inter-step planning model calls.
At A37, the plan was evaluator-owned for Foundation E04–E07; evaluator setup
performed the CMake configure step, then Forge's production gate executed trusted
`project.build` followed by `project.test`. The independent oracle remains a
separate build→test comparison and cannot mark Forge verified. A36's prior E04
finding arose because immediate test ran before the build prerequisite; explicit
A37 plan semantics resolve that sequencing without changing the task or mutation.
The deterministic accepted A35 line-range delta, reproduced through normal Forge
preview and approval in a disposable E04 workspace (seed 42), acquired the expected
source via lexical bootstrap and a bounded range read, made one mutation, then
passed production build and test. Forge returned `completed_verified`; the
independent oracle also passed, and canonical Foundation hashes were unchanged.

## project-configure-v1

A38 adds eight deterministic production-orchestration cases using configured
subprocesses in disposable workspaces. C01 verifies configure→build→test from a
clean workspace. C02 stops after nonzero configure without code repair. C03/C04
exercise distinct CONFIGURE DENY and ASK rejection. C05 stops after build failure.
C06 ensures configure-created build artifacts remain generated metadata, not
implementation evidence. C07 performs one repair and reruns the full plan. C08
binds a matching pre-existing test failure to the same configure→build→test
baseline and blocks repair. Separate tests cover timeout and process launch
failure. The model does not choose required plan steps, and normal configure
output is compacted before the next model request.

Foundation E04–E07 no longer use evaluator-owned CMake configure setup. Their
trusted task definitions provide a `project.configure` argument array and the
three-step plan. The independent oracle still runs separately after Forge. A38's
accepted E04 line-range delta was reproduced from a clean disposable copy:
lexical bootstrap acquired the expected source, one approved mutation executed,
and Forge's configure, build, and test each passed in order. Forge returned
`completed_verified`; the independent oracle passed; the canonical Foundation
repository hash was unchanged. The historical A37 acceptance script retains its
former evaluator-configured definition for comparison.

In that E04 run, Forge's configure step took 0.57 seconds, build 10.27 seconds,
and test 15.88 seconds; the plan took 26.71 seconds and the complete evaluation
31.04 seconds. Relative to A37, Forge charged one additional tool execution
(six instead of five), while both used three model calls. These are local
observations, not general performance guarantees.

## process-isolation-v1

A39 adds eight deterministic cases. I01 redirects HOME/TMPDIR into the disposable
workspace; I02 excludes a fake parent secret; I03 runs a PATH-resolved trusted
command; I04 proves an unavailable strict adapter fails before execution; I05/I06
exercise outside-write rejection and inside-write allowance through a deterministic
fake adapter; I07 verifies timeout cleanup of a spawned descendant; I08 runs the
existing configure→build→test production plan under `controlled_env` with no
inter-step model call. Additional tests cover policy parsing, NONE compatibility,
approval snapshot currentness, isolation attribution identity, permissions,
symlinked execution-directory rejection, metadata classification, and log redaction.

The installed macOS `sandbox-exec` adapter was probed outside the enclosing
development sandbox. A harmless command and workspace write passed; an attempted
write to a sibling temporary file failed and left it absent. The adapter allows
runtime reads outside the workspace, so this is not complete read confinement.
The optional full strict Foundation E04 run stopped at configure (the independent
oracle still passed); this does not affect the required controlled-environment
acceptance and remains a strict toolchain-compatibility limitation.

The accepted E04 delta under `controlled_env` began in a clean disposable copy,
passed Forge configure→build→test, returned `completed_verified`, and passed the
independent oracle with canonical Foundation unchanged. The final PATH-hardened
repeat took 0.40 seconds to configure, 7.03 seconds to build, and 12.02 seconds
to test; the plan took 19.45 seconds and total evaluation took 22.46 seconds. It
used six tools and three model calls, the same counts as A38;
these local timings do not isolate environment-setup overhead from run variation.

## strict-toolchain-v1

A40's ten-case deterministic suite exercises compiler query, compile-to-object,
link, executable run, workspace/sibling/home-like writes, unavailable-adapter
fail-closed behavior, a strict configure→build→test fixture, and a controlled-env
regression. S01–S09 use explicitly labeled fake adapters inside pytest because the
enclosing development sandbox cannot install nested Seatbelt policy; fake PASS is
an orchestration-contract result, not real strict toolchain acceptance. S10 runs
the real controlled environment.

The final real macOS ladder used `/usr/bin/cc` (Apple clang 21.0.0) and production
adapter `macos-sandbox-exec-toolchain-v2`. Targeted diagnostics isolated Apple
`ld`'s required `sysctl-read hw.pagesize_compat`: its denial caused the observed
`UnsafeHeaderWriter` assertion, while granting that exact read alone restored
linking. Compiler query, compile-to-object, link, executable run, and trivial
CMake configure all passed under the corrected production policy.

The real boundary regression before and after acceptance allowed workspace writes
and denied disposable sibling and home-hierarchy writes. From a clean disposable
Foundation copy, strict configure, build, and test passed, Forge returned
`completed_verified`, and the independent oracle passed with canonical Foundation
unchanged. Deterministic suite aggregate fields explicitly report fake-adapter
contract scope and leave real-platform acceptance unset; real acceptance is
reported only by the separate macOS run.

## multi-file-mutation-v1

A41 adds ten deterministic grouped-mutation scenarios. M01 and M02 cover two-file
exact-text and line-range success. M03 and M05 reject an invalid or no-op child before
preview/application, while M04 changes the second file after preview and proves the
complete compare-before-write pass leaves the first untouched. M06 binds approval to
the exact complete group. M07 injects failure on the second replacement and verifies
exact-byte rollback, modes, generation, and cleanup. M08 injects rollback failure and
requires the fatal workspace-integrity result. M09 runs normal post-group
verification to `completed_verified`; M10 performs grouped primary mutation, failed
verification, fresh reads, grouped repair, and successful reverification while
counting two logical mutations rather than four file writes.

Additional transaction tests cover deterministic path ordering, canonical proposal
identity, the four-file ceiling, duplicate and unauthorized paths, symlink rejection,
LF/CRLF/unterminated UTF-8 files, file-mode preservation, and temporary-state cleanup.
These tests establish application-level rollback for handled failures; they do not
claim crash- or power-loss consistency across several filesystem replacements.

Foundation realworld-v1 E08 historically remained unsupported because its unchanged
request requires one coordinated non-repair change to both `src/clock.c` and
`tests/test_solar.c`, beyond the prior single-file mutation envelope. It requires no
creation, deletion, rename, binary edit, or fifth file, so A41 now represents it with
one two-file grouped line-range mutation and the existing configure→build→test plan.

The qwen-large, seed-42, 8192-context acceptance was attempted unchanged on a
disposable Foundation copy. Baseline verification passed and lexical bootstrap ran,
but the model inspected `src/clock.c`, `src/power.c`, `src/simulation.c`, and
`src/snapshot.c` without acquiring `tests/test_solar.c`; it then emitted a duplicate
tool-call identifier. Forge consequently did not reach grouped MUTATION_READY or
preview, made no mutation, and did not run post-mutation verification. The harness
classified the run as model-quality/protocol failure. Its independent oracle passed
only because the unchanged baseline already satisfies the oracle, not because Forge
applied the requested grouped change. Transactional safety and schema constraints
were not weakened or retried.

## multi-source-acquisition-v1

A42 adds ten deterministic scenarios. Q01 acquires two required candidates without
an intervening model turn; Q02 skips an already-current candidate; Q03 proves stable
path ordering; Q04 and Q05 bound READ denial and a missing file; Q06 protects the
remaining read, grouped mutation, and three verification steps or blocks before
mutation; Q07 invalidates source after a generation change; Q08 reacquires both
original paths before grouped repair; Q09 accepts one corrected unique identifier
after rejecting a duplicate; and Q10 terminates on a repeated duplicate. Additional
tests prove that identical actions with different IDs are legal, IDs remain scoped
across turns/compaction within a session, and independent sessions have fresh scopes.

The unchanged Foundation E08 qwen-large run used seed 42, the 8192-token profile,
LINE_RANGE, and a disposable repository at identity
`26caf0fb85a0efbe2fa709edceca585793071a0a`. The trusted production task definition,
separate from evaluator-only expected/scoring fields, established `src/clock.c` and
`tests/test_solar.c` as required candidates. Forge acquired both through two
orchestrator-required ToolExecutor reads in normalized order, with no model-requested
source reads, and reached grouped MUTATION_READY. Acquisition took 0.0017 seconds.

The model then failed to emit a mutation after the bounded mutation-ready correction,
so there was no proposal, preview, transaction, or post-mutation verification. Forge
truthfully returned `failed_before_mutation` with MODEL_QUALITY; the independent
oracle passed only against the unchanged baseline. The run used two tools and two
model calls in 51.89 seconds. Compared with A41's seven tools, seven model calls,
79.70 seconds, missing test source, and duplicate identifier, A42 acquired both
sources, reached readiness, avoided an observed duplicate, and used five fewer tool
and model calls. These timings and savings are observations from the two acceptance
runs, not general guarantees.

## grouped-protocol-compatibility-v1

A43 separates five layers while holding seed 42, temperature zero, 8192 context,
512 output tokens, task text, source bytes, and normalized source order constant.
G1 asks only for a short implementation/test concept description. G2 uses two
independent evaluator-owned line-range objects. G3 adds a minimal evaluation-only
`edits` group. G4 calls the live `build_mutation_ready_output` function and parses
the actual A41 `multi_file_line_range_edit` envelope. G5 runs unchanged production
orchestration. No diagnostic edit is transformed into a production action.

The three synthetic tasks coordinate exactly two existing files and use executable
oracles: P01 compiles and runs a C17 boundary test; P02 and P03 run Python
implementation/test pairs. qwen-large passed concepts on 3/3, produced valid targets
on 3/3 G2 cases, and passed 2/3 G2, 2/3 G3, and 2/3 G4 oracles. All G3/G4 envelopes
and path sets were valid. qwen-small also passed concepts and target construction on
3/3; it passed 1/3 G2, 1/3 G3, and 2/3 G4 oracles. This shows grouped packaging and
the production envelope are generally usable; semantic edit quality varies by task.

For Foundation E08, G1 recognized both implementation and regression-test concepts.
G2 produced two selectable material edits, but both conservative per-file semantic
checks and the combined oracle failed. G3 and G4 each produced a schema-valid
two-path grouped response without truncation, but A41 rejected a no-op child before
oracle execution. G4 used 153 of 512 output tokens. Thus Foundation first loses at
exact edit construction, not grouped packaging or the production schema.

The production prompt audit nevertheless found a generic contradiction: grouped
LINE_RANGE readiness and premature-final recovery still requested one single-file
`line_range_edit`, and schema-error recovery could ask for a `tool_call` that the
mutation-ready schema did not offer. After representation-aware grouped wording and
bounded incomplete-group recovery were added, the one permitted G5 rerun emitted a
valid two-file grouped action on its first response. Forge created the grouped
preview, executed the atomic transaction, and entered verification. Build and the
independent oracle failed, so Forge truthfully returned
`mutated_verification_failed`. The mutation-ready generation used 4,858 input and
198 output tokens and took 25.32 seconds; it was not truncated. The full-run elapsed
value was not retained by the initial A43 result serializer and is reported as an
instrumentation limitation rather than estimated.

The A42 zero-context observation was a failure-path reporting gap, not an empty
request: direct G1–G4 Foundation requests measured roughly 4,549–4,572 input tokens,
and G5 measured 4,858 actual input tokens. After G5 completed far enough to return
production metrics, its context peak was nonzero. Source presentation was complete:
`src/clock.c` preceded `tests/test_solar.c`, each had an explicit path boundary and
independent line numbering, and their numbered-source estimates were 774 and 3,962
tokens respectively.

## Benchmark integrity

A semantic mutation result is score-eligible only when its evaluator-owned oracle
fails the unchanged baseline and passes a known-good reference mutation. Reference
source and oracle metadata never enter model prompts, candidate metadata, or source
observations. A baseline-passing oracle is classified as non-discriminating and may
still support architectural or protocol observations, but it cannot prove semantic
success. The A44 fixtures also reject an obviously wrong mutation in which the test
changes but the implementation defect remains.

## paired-semantic-planning-v1

A44 compares D0 direct grouped editing with D1 plan-then-grouped-edit for six paired
implementation/test tasks in C17 and Python. Both conditions use seed 42,
temperature zero, an 8192-token context, the live production grouped LINE_RANGE
schema, and a 512-token edit allowance. D1 adds one 256-token call for a concise
externally visible change contract; it does not request hidden reasoning and does
not translate the plan into edits. Plans are limited to four required paths and are
bound to the task, normalized required-path set, workspace generation, and source
hashes. They grant no mutation authority.

Every P01–P06 oracle failed its unchanged baseline, passed its evaluator-only
reference mutation, and rejected a changed test paired with the broken
implementation. P01–P03 retain the A43 source tasks; P04 covers a retry-count
boundary, P05 a C parser-token maximum, and P06 a validation minimum.

For qwen-large, D0 produced 6/6 structurally valid groups and 5/6 semantic passes.
D1 accepted 3/6 plans, produced 3/6 structurally valid groups, and retained 2/6
semantic passes. It used nine model calls, 2,635 input tokens, 1,419 output tokens,
and 50.18 seconds, versus D0's six calls, 1,281 input tokens, 820 output tokens, and
28.99 seconds. For qwen-small, D0 produced 6/6 structurally valid groups and 2/6
semantic passes. D1 accepted 3/6 plans, produced 2/6 structurally valid groups, and
passed 1/6 semantic oracles. It used nine calls, 2,876 input tokens, 1,268 output
tokens, and 62.15 seconds, versus D0's six calls, 1,407 input tokens, 503 output
tokens, and 26.44 seconds.

Planning improved qwen-small P02 but regressed qwen-small P01 structurally and did
not improve qwen-large on any task. P04–P06 plans missed required deterministic
concepts for both models and therefore conferred no authority to proceed. The
intervention is neither broadly beneficial nor structurally neutral, so A44 makes
no production planning change. D2 was not run because D1 was already materially
worse while adding cost; an additional call could not justify production adoption
under the milestone standard.

## Foundation E08 semantic integrity

The unchanged Foundation E08 configure/build/test oracle passes the unchanged
repository. Its task asks for "a clock boundary" and a corresponding source fix but
does not identify which clock boundary or specify its intended behavior. The oracle
only establishes that the project builds and existing tests pass; it does not
establish either required semantic delta. E08 is therefore retained unchanged as a
legacy architectural benchmark for multi-file acquisition, grouped readiness,
protocol, transaction, and verification-path observations, but is classified
`SEMANTICALLY_UNDER_SPECIFIED` and excluded from semantic success rates. No E08v2
was created because doing so would require silently inventing task semantics, and
no additional E08 model run was performed.

## realistic-semantic-v1

A45 adds a frozen, non-packaged mixed Python/C17 service-engine snapshot with 25
non-test source/tool files, 10 tests, 436 source/test LOC, and 36 files overall.
The repository includes retry, configuration, state, header validation, parsing,
quota, window, status, storage, service, checksum, and clock code so relevant edits
are surrounded by plausible adjacent implementation. Every model cell uses an
independent disposable copy; the canonical snapshot identity is
`99e6155e9350711bbcafcb72f6d401a76a58f56ecef1bd30a7f32ece559f37a5`.

R01-v1 through R04-v1 are discovery-required single-file maintenance tasks covering
retry accounting, configuration interpretation, complete-input parsing, and quota
underflow. R05-v1 through R08-v1 are path-known paired implementation/regression
tasks covering terminal state transitions, header validation, exclusive event
limits, and error propagation. The paired paths are legitimate trusted task
metadata and use unchanged A41–A43 grouped acquisition and atomic mutation. All
tasks run unchanged production orchestration with LINE_RANGE, 8192 context,
temperature zero, 512 output tokens, and the existing bounded repair opportunity.

Integrity is established before model execution. Each post-setup baseline fails its
evaluator-only semantic oracle, restoration from the canonical reference passes,
and a plausible wrong mutation fails. Hidden behavior assertions are external to
the workspace. Paired-task oracles additionally mutation-test the submitted
regression against the known defect, allowing alternative correct implementations
while rejecting tests that do not detect the behavior. Neither canonical reference
bytes nor hidden assertions enter prompts, retrieval, candidate evidence, or the
wheel.

qwen-large ran all tasks at seeds 42 and 43. Results were identical at temperature
zero: R05, R07, and R08 passed mutation, configured verification, and the semantic
oracle; R01–R04 reached mutation readiness but emitted no valid edit; R06 failed the
grouped protocol before preview. Thus each seed produced 3/8 first-pass and final
semantic passes, with no repairs. The repeated result demonstrates stability but
also stable failure rather than seed robustness.

qwen-small ran seed 42 only as the cost-control matrix. R01, R04, R06, and R07
passed the semantic oracle. R01, R04, and R07 also passed configured verification;
R06 is a verification-only failure because the mutation passed the independent
semantic oracle while project verification failed. R02 and R03 failed verification,
R05 failed during repair-phase source reacquisition after a failed first mutation,
and R08 failed edit construction. Three repairs were attempted and none succeeded.
The result is 4/8 semantic passes, all attributable to the first mutation rather
than repair.

Across the 24 cells Forge reached mutation readiness every time and had no primary
discovery failure. qwen-large used 32 recorded model calls and 418.38 seconds;
qwen-small used 29 calls and 318.43 seconds. Average wall time was 30.70 seconds per
cell. Successful/returned-result cells recorded at least 57,472 input and 5,102
output tokens in aggregate. Existing failure-path usage reporting leaves token
fields null for some pre-response and orchestration-error cells, so those token
totals are explicit lower bounds rather than fabricated estimates.

Structural and semantic evidence remains separate: a semantic pass requires an
actual mutation plus independent-oracle PASS, while readiness, schema, preview,
transaction, and verification are reported independently. First-pass and repaired
success are also distinct. The dominant qwen-large limitation was edit construction
and grouped protocol compliance rather than discovery; qwen-small showed more
successful edit construction but costly unsuccessful repair and verification
failures. This evidence favors an alternative local coding-model benchmark next,
not new planning, retrieval, mutation, context, or retry architecture.

## alternative-model-bakeoff-v1

A46 consumes the frozen `realistic-semantic-v1` task definitions and A45 result
artifacts without copying or changing the benchmark. Candidate enumeration reads
only trusted configured model profiles. Repository files cannot register model
artifacts. Static eligibility requires the existing generic llama.cpp backend, a
local GGUF artifact, the unchanged 8192-token context, instruction/coding use, and
no model-specific production parser or orchestration branch.

The comparison schema keeps semantic success, single-file and multi-file success,
mutation readiness, valid schema, preview, transaction, verification, repair,
failure layer, model calls, token availability, and elapsed time separate. Artifact
identity supports exact filename, byte size, SHA-256, GGUF version, architecture,
quantization, declared context, and chat-template provenance. Load and protocol
smoke results are represented independently from semantic scores. A45 baseline
reuse validates suite/schema versions, repository identity, context 8192, output
512, temperature zero, and canonical safety before aggregation.

The authorized candidates are DeepSeek-Coder-V2-Lite-Instruct Q4_K_M and
Codestral-22B-v0.1 Q4_K_M. Their local SHA-256 values are respectively
`603bd3f8a0281d16571da7c08bd661ee17ff0d1be6fcbd1b42242da257ef0bb8`
and `003e48ed892850b80994fcddca2bd6b833b092a4ef2db2853c33a3144245e06c`.
The model repositories report `deepseek-license` and `mnpl`. DeepSeek supplies an
embedded GGUF chat template. The Codestral GGUF does not, so its trusted profile
uses llama-cpp-python's existing generic `mistral-instruct` format. Both created an
8192-token Metal context and passed deterministic generation. Both passed the
single-file protocol smoke and emitted the grouped action type, but used an
incorrect grouped path set.

DeepSeek's R01 artifact was lost to the original runner's all-or-nothing process
boundary and was not rerun. Its non-duplicated R02–R08 evidence contains seven
valid readiness states, six valid mutations, no semantic passes, four repair
attempts, and no repair success. Three tasks failed verification, three ended with
source-acquisition diagnostics after initial source readiness, and R08 failed edit
construction. It used 27 model calls and 552.90 task-seconds for those seven cells.

Codestral completed the full primary matrix through per-cell checkpoints. At both
seeds 42 and 43 it passed R04, R05, R07, and R08 and failed verification on R01,
R02, R03, and R06. Each seed therefore produced 4/8 first-pass semantic passes,
1/4 single-file success, 3/4 multi-file success, 8/8 valid mutations, two repair
attempts, and no repair recovery. Seed 42 used 32 model calls and 797.64 task-seconds;
seed 43 used 32 calls and 668.75 task-seconds. Codestral is stable and highly
protocol-compatible, but it only ties qwen-small's overall semantic count and is
substantially slower. It is classified `SIMILAR`, with stronger coordinated-edit
behavior rather than a clearly stronger overall capability envelope.

The matrix exposed an evaluation-only classification defect: successful verified
mutations could retain an incidental source-read failure label. Terminal semantic
and verification success now correctly classifies as `PASS`. It also exposed the
need for per-cell result durability, so the evaluation runner checkpoints each
candidate cell without changing production orchestration. No default model,
benchmark, prompt, budget, repair limit, or production mutation behavior changed.

## repair-effectiveness-v1

A47 derives a source-free repair-case corpus from the immutable A45 qwen-small and
A46 Codestral seed-42 artifacts. It contains eight failed executed model cells,
seven of which retain legitimate production-visible repair eligibility: three
qwen-small cells and four Codestral cells. The corpus covers Python and C17,
single-file and grouped mutations, build/compiler and test failures, one
verification-fail/oracle-pass disagreement, and one ineligible repair-source
reacquisition failure. Hidden oracle outcomes classify results only after execution;
they never enter repair evidence or prompts.

The paired evaluation compares R0, the existing current-source repair prompt, with
R1, which inserts the exact accepted primary `MutationPreview` diff between the
trusted verification diagnostic and freshly reacquired current source. Diff history
is immutable, bound to the successful primary generation and authorized primary
paths, and limited to 4,096 characters with an explicit middle-truncation marker.
R1 replays each R0 primary response sequence up to repair readiness, so the task,
mutated workspace, source, seed, representation, verification evidence, output
budget, and mutation ceiling remain fixed. Both conditions reuse the production
single/grouped mutation schemas, A41 transaction, A42 source currentness, A36
attribution, and the complete A37 configure/build/test rerun.

qwen-small completed three paired cases. R0 and R1 each produced 3/3 structurally
valid executed repairs, zero verification passes, and zero semantic recoveries. R1
raised mean repair input from 1,316.7 to 1,515.3 tokens and generation latency from
9.68 to 11.08 seconds without benefit. Codestral completed four paired cases. R0
produced 3/4 structurally valid repairs and one semantic recovery; R1 produced 2/4
structurally valid repairs and one semantic recovery. R1 recovered R01, which R0
did not, but regressed the already-recoverable R06 and made R02 structurally invalid;
R03 became structurally valid but still failed verification. Mean Codestral repair
input rose from 1,162.8 to 1,418.3 tokens and generation latency from 22.82 to 26.95
seconds.

The accepted-diff intervention therefore changes which Codestral case recovers but
does not increase aggregate recovery, does not reproduce across qwen-small, and
materially reduces Codestral structural validity. Under A47's adoption rule this is
weak evidence. Forge retains the bounded immutable history representation and the
evaluation switch, but the production default remains R0. No repair retry, path,
permission, model, context, output, tool, or mutation budget changes.

## verification-alignment-v1

A48 measures the unchanged configured verification plan against evaluator-only
semantic truth. For each frozen R01–R08 task it constructs BASELINE, REFERENCE, and
WRONG independently from the canonical repository, revalidates oracle FAIL/PASS/FAIL,
and executes the same configure/build/test plan and isolation policy used by
`realistic-semantic-v1`. The 24-state run preserved repository identity
`99e6155e9350711bbcafcb72f6d401a76a58f56ecef1bd30a7f32ece559f37a5`.
References, wrong mutations, and hidden assertions remain evaluator-only and are
absent from production prompts, repair evidence, and installed package data.

R01–R04 are `FULLY_DISCRIMINATING`: their defective baseline and plausible wrong
mutation fail the visible task-relevant test, while the reference passes. R05–R08
are `BASELINE_ACCEPTING`: each paired task setup introduces the implementation
defect while replacing the canonical regression assertion with an unrelated passing
assertion. Consequently the defective baseline passes the full plan. Restoring only
the canonical visible test creates the WRONG state and exposes the still-defective
implementation, so all four WRONG states fail. This is a
`MISSING_BEHAVIOR_ASSERTION` gap in the configured baseline, not an infrastructure,
configure, build, stale-reference, or unrelated-failure problem. All eight reference
states pass; there is no verification rejection of a known-good state in the
deterministic matrix.

The confusion matrix is 8 semantic-correct/verification-pass, 0
semantic-correct/verification-fail, 4 semantic-wrong/verification-pass, and 12
semantic-wrong/verification-fail. Thus 12 semantic failures are legitimately
observable by production verification, while four defective paired baselines are
invisible. No semantically correct deterministic state would spuriously trigger
repair. These terms are deliberately phrased as verification acceptance of a
semantically wrong state and verification rejection of a semantically correct state,
avoiding ambiguous false-positive/false-negative labels.

The existing-test inventory preselected one visible task-named test before observing
outcomes: `tests/test_retry.py`, `tests/test_config.py`, `test_parser`, `test_quota`,
`tests/test_state.py`, `tests/test_headers.py`, `test_window`, and `test_status`.
Every test exists, directly exercises the source behavior, and is already run by the
full plan. T1 reproduced T0 for all eight tasks: four fully discriminating and four
baseline accepting. No subset improved or regressed discrimination, and no hidden or
generated test was used. The full plan remains authoritative.

The 24 full plans took 30.29 seconds in aggregate: 9.88 seconds for BASELINE, 12.27
for REFERENCE, and 8.14 for WRONG. The 24 targeted invocations took 0.73 seconds.
Targeted execution is much cheaper here but supplies no additional semantic
discrimination, so A48 does not adopt it in production.

Historical executed mutations map to 17 `V_PASS / S_PASS`, 17
`V_FAIL / S_FAIL`, one `V_FAIL / S_PASS` (qwen-small R06), and one
`V_PASS / S_FAIL` (DeepSeek R07). The latter is the prominent production-invisible
semantic bug. Six of A47's seven eligible original failures aligned verification
failure with semantic failure; qwen-small R06 was the sole potentially harmful
repair trigger because its mutation was semantically correct. Its source-free A45
artifact records `project.test` failure but intentionally retains no full log or
submitted source, so exact failure fingerprint and same-workspace attribution cannot
be reconstructed. The deterministic A48 R06 reference passes the same full plan,
which rules out a generally stale canonical expectation but cannot distinguish an
alternate-correct implementation/test conflict from an unrelated submitted-test
failure. Repair was therefore not semantically appropriate.

Codestral's original R01 and R06 cells were both `V_FAIL / S_FAIL`, so their A47
repair triggers correlated with semantic incorrectness. R1 recovered R01 and R0
recovered R06 to `V_PASS / S_PASS`; these are aligned semantic recoveries even though
the intervention did not improve aggregate effectiveness. A48 changes no production
verification, repair prompt, retry, model, budget, benchmark, or mutation behavior.

## acceptance-test-synthesis-v1

A49 asks whether a local model can synthesize one bounded behavioral acceptance test
using production-visible information only. C0 supplies the task plus minimal trusted
language/test conventions. C1 additionally supplies the task's relevant current
source and existing visible test source. Neither condition contains the hidden
oracle, reference or wrong mutation, semantic labels, expected implementation, or
A48 classification. The model returns only a constrained test name and test body;
execution commands remain evaluator-owned argument arrays.

Static validation bounds source size and rejects evaluator leakage, path escape,
network/process/file access, unsupported Python imports/calls, unsupported C
includes/operations, and source-text patch matching. Python candidates must parse;
C candidates must supply a standalone `main` and compile as C17 against the trusted
task implementation. Valid candidates run with a ten-second timeout and a restricted
environment in fresh independent BASELINE, REFERENCE, and WRONG workspaces. A
candidate qualifies only for FAIL/PASS/FAIL and an identical second REFERENCE run.
Standard JSON stores only source hash, size, classifications, and metrics—not the
generated body. Evaluator materialization grants no production write authority.

The real matrix used seed 42, temperature zero, context 8192, and output 512, with
one generation per cell. Both qwen-small and Codestral ran R05–R08 under C0/C1 plus
R01/R02 controls. qwen-small qualified 0/6 C0 cells and 2/6 C1 cells: grounded R01
and grounded R05. Its R05 candidate also passed an evaluator-owned behaviorally
equivalent implementation. R06 exhausted the bounded schema output in the task-only
condition and emitted syntax-invalid Python when grounded. R07/R08 produced one
compile-invalid and three unsafe C artifacts.
Codestral qualified 0/6 in both conditions. Its Python candidates either missed the
baseline or rejected the reference; its C0 outputs were schema-invalid and its C1
outputs lacked a standalone `main`.

For the primary verification-blind tasks, only R05 was recovered by any model and no
task was recovered by both. R06–R08 remained uncovered. No qualified candidate was
flaky, leaked evaluator data, inspected source text, or mutated the canonical
repository. Model-free tests additionally prove behavioral tests accept alternative
correct R05 and R06 implementations. Qualified tests compose with A48 full
verification using AND semantics: R05 changes from baseline-accepting PASS/PASS/FAIL
to FAIL/PASS/FAIL without replacing the full plan.

Across six cells per condition, qwen-small generation took 56.88 seconds for C0 and
65.81 for C1; candidate qualification took 2.63 and 1.38 seconds. Codestral generation
took 122.89 seconds for C0 and 75.79 for C1; qualification took 7.94 and 7.43 seconds.
The 15 executed candidate qualifications took 19.38 seconds total, including four
runs per candidate, versus A48's 30.29 seconds for 24 full-plan states. Focused tests
are cheap to execute, but generation dominates and reliability is inadequate.

Cross-model testing against A45/A46 patches was investigated but could not be run:
the durable historical artifacts intentionally retain classifications and metrics,
not submitted source or reconstructable workspaces, while A49 standard artifacts
likewise omit raw test bodies. No patch model was rerun and no model-coupling claim is
made. The reference/wrong qualification itself remains generator-independent, and
the qualified R05 candidate accepted an independently authored alternative correct
implementation.

The production qualification gap is decisive. Evaluation knows a good reference and
a plausible wrong state; production knows only the current repository, task, and a
future mutation. Baseline failure is necessary evidence that a test adds signal, and
post-mutation success is necessary evidence of compatibility, but neither proves the
test expresses the requested behavior. A single model can generate a mutually
consistent wrong patch and wrong test, and even independent-model review cannot
equal B/R/W qualification. Human approval, structural/safety checks, baseline fail,
post-mutation pass, and full regression verification could form a conservative
future gate, but no automated trustworthy replacement for evaluator qualification
was established. A49 therefore makes no production adoption, repair integration,
test-file creation, verification-plan, prompt, retry, model, or budget change.

## evaluation-replay-v1

A50 separates source-free standard evaluation results from local replay bundles.
Standard JSON contains only bounded identities, classifications, hashes, timings,
usage, verification/oracle outcomes, and replay artifact IDs. Raw mutation text and
generated test bodies exist only beneath the operator-selected
`eval-results/replay/` location, which is ignored, outside package source, and never
emitted through normal production telemetry.

Replay bundles are evaluator-owned JSON with an explicit version, canonical payload
SHA-256, frozen benchmark/repository identity, task version, defective source-state
identity, model profile, seed, and one bounded type: `MUTATION_PROPOSAL` or
`GENERATED_ACCEPTANCE_TEST`. Mutation replay validates every group member before any
write and rejects stale hashes/ranges and path or symlink escape. Generated tests
are rechecked by the current A49 structural and safety validator before trusted
execution. Replay only materializes disposable frozen-benchmark workspaces, creates
no production `ToolExecutor` authority, and cannot target user repositories.

Artifacts and the payload-free manifest use same-directory temporary files and
atomic replacement. A committed, hash-valid manifest entry is skipped on resume. A
cell that returns no artifact gets an atomic unavailable checkpoint and is also
skipped, preventing success-seeking retries. An interrupted cell without either
commit may run once again as the same planned execution. Local bundles are retained
manually and carry `LOCAL EVALUATOR DATA — MAY CONTAIN MODEL-GENERATED SOURCE.` They
must not be uploaded or packaged automatically.

The fixed seed-42 R05–R08 matrix generates fresh qwen-small and Codestral patches
through unchanged production orchestration and fresh grounded C1 tests. Available
tests run independently on BASELINE/REFERENCE/WRONG, while hidden semantic truth is
computed separately for patches before all same-model and cross-model pairings. The
standard result links both artifact IDs and hashes without source. Predeclared
equivalent implementations for R05–R08 distinguish behavioral generalization from
reference-shape matching. Positive replay evidence still cannot close production's
missing-reference gap, so A50 does not adopt generated tests.

The fresh A50 seed-42 run committed 15 replay bundles: seven accepted patch states
and eight generated test bodies. qwen-small yielded semantically wrong R05/R06
patches, a correct R07 patch, and no accepted R08 patch; Codestral yielded four
semantically correct patches. One of eight tests qualified independently: the
qwen-small R05 C1 test. It rejected qwen-small's own wrong R05 patch, accepted
Codestral's correct R05 patch, and accepted the predeclared alternative-correct R05
implementation. Of 14 available test/patch pairings, one accepted a correct patch,
two rejected correct patches, and three rejected wrong patches; the rest were unsafe
or unexecutable rather than behavioral rejections. No wrong patch was accepted by
an executable candidate. The single qualified test shows cross-model behavioral
generalization on R05, not systematic same-model preference. Seven other tests did
not qualify, so the production trust gap remains decisive.

## human-gated-acceptance-v1

A51 adds H01–H16 model-free trust-boundary checks and a replay-backed evaluator
runner, `scripts/run_human_gated_acceptance_v1.py`. It reads the A50 qwen-small
R05 test from local replay and creates only disposable benchmark copies. An
explicit simulated review is followed by a correct paired mutation, a wrong
patch, and the independent Codestral patch. The approved test passes both
correct states and rejects the wrong state; both correct states also pass the
unchanged configure/build/test plan. For R06–R08, the evaluator-only ideal
policy rejects every unqualified A50 candidate. That qualification is not
imported into production and cannot auto-approve a test.

H01–H16 cover reviewability, exact preview and approval, invalidation,
capability separation, optional rejection, postmutation gating, full-plan
authority, headless/autonomy denial, and repair isolation. The suite demonstrates
a guarded lifecycle, not general semantic reliability for generated tests.

## ephemeral-acceptance-isolation-v1

A52 adds I01–I18 model-free contract cases and
`scripts/probe_ephemeral_isolation_v1.py` for explicit real macOS acceptance.
Deterministic fake adapters test lifecycle, policy identity, failure taxonomy,
and source-integrity contracts; they are not presented as OS-containment proof.
The real Seatbelt probe uses disposable workspaces and verified: ordinary Python
and project imports PASS; private temp writes PASS; project and sibling writes,
real-HOME read/write-open, loopback socket bind, external and same-interpreter
subprocesses, and symlink escape are denied. A known existing README was opened
read-only or write-open only to test authority; no real HOME file content was
modified. The A50 qualified R05 test was replayed under the new strict profile:
baseline assertion FAIL, correct and Codestral states PASS with full
configure/build/test PASS, and the wrong state FAIL before full verification.
No new model generation or evaluator qualification rule was introduced.

## controlled-file-creation-v1

A53 adds C01–C16 deterministic checks for exact new-path authority, preview and
approval binding, one/two-file creation, protected paths, content and parent
validation, no-overwrite prechecks, owned-file rollback, and one-generation
accounting. C03 explicitly rejects mixed edit/create because v1 retains the
existing A41 edit transaction unchanged. Tests use disposable directories and
injected later-child/rollback failures; they do not imply crash atomicity.

The realistic A53 create task uses a disposable Python package with an absent
module required by an existing caller. The baseline behavioral oracle fails,
and an evaluator-held reference file passes. Real-model acceptance, when run,
uses qwen-small at seed 42, temperature 0, 8192 context, and 512 output tokens;
the milestone does not depend on semantic oracle success to prove transaction
safety. The qwen-small host run reached CREATE_READY, proposed the authorized
new module, displayed and approved its CREATE preview, applied the transaction,
passed configured project testing, and passed the independent oracle. The
fixture is not production guidance and does not enter the wheel.

## mixed-file-transaction-v1

A54 adds deterministic M01–M18 cases in `tests/test_mixed_file_transaction.py`.
They cover mixed success and four-child bounds, canonical ordering, exact edit and
create authority, duplicate paths, all-before-write stale/appearance checks,
missing/protected parents, reverse-order rollback, terminal integrity failures,
one-generation/one-mutation accounting, fresh repair of a created file, legacy A9
bypass closure, and ephemeral-test non-authority. A41 edit-only and A53 create-only
tests remain separate compatibility checks.

`scripts/run_mixed_file_transaction_v1.py` creates a disposable Python application
where the baseline behavioral oracle fails and an evaluator-held two-file reference
passes. Its optional qwen-small attempt uses seed 42, temperature 0, 8192-token
profile context, and 512 output tokens. The task authorizes one existing-file edit
and one new module path through trusted setup; evaluator reference text never grants
runtime authority. Model semantic failure is reported separately from transaction
mechanics. No benchmark fixture or model artifact enters the package.

The real qwen-small attempt outside the Metal-restricted sandbox loaded normally
and reached mixed readiness. It emitted four duplicate CREATE children for the
single authorized new path, then repeated that shape after one generic correction.
The complete-group validator rejected both before preview, approval, mutation, or
verification. This is a model proposal/path-set failure, not a transaction
failure; the disposable baseline/reference oracle remained FAIL/PASS.

## mixed-operation-protocol-v1

A55 isolates mixed-operation reasoning from production orchestration. The
evaluator-only corpus in `scripts/mixed_operation_protocol_v1.py` has six cases:
four Python caller/registry/parser/package tasks and two C dispatcher/unit tasks.
For every case, the unchanged baseline and both one-operation-only reference
states fail, while the paired MODIFY+CREATE reference passes. The unchanged A54
task is an external A54-R1 case. Reference code is never placed in model input.
No A55 fixture or result is a wheel/sdist runtime resource.

`scripts/run_mixed_operation_protocol_v1.py` runs M1 bounded intent roles, M2
independent LINE_RANGE and create actions, M3 a minimal heterogeneous schema,
and M4 the live `build_mutation_ready_output` schema plus `parse_model_output`.
All valid pairs are validated and applied through the A54 transaction in a
disposable workspace before the independent semantic oracle. M5 invokes the
actual repository session separately. Seed 42, temperature 0, 8192 context,
512 output tokens, and LINE_RANGE are held fixed; no planning or extra retry is
introduced. Metrics retain child types/paths, counts, tokens, latency, and
truncation, never generated source content.

Across X01–X06 and A54-R1, all three required models passed M1 role/path
classification on 7/7 cases and emitted valid independent M2 child shapes on
7/7. Semantic M2 passes were qwen-small 1/7, qwen-large 4/7, and Codestral
2/7. M3 structural pairs were 7/7, 6/7, and 6/7 respectively; the two losses
were unauthorized leading-slash paths, not duplicate operations. M4 emitted
the exact two authorized child roles and paths on 7/7 for each model. M4
semantic passes were 1/7, 4/7, and 3/7 respectively. No M1–M4 output reached
the 512-token ceiling (largest reported output: 144 tokens), and no M4 bounded
correction was needed. The first general model limitation in this corpus is
semantic child construction, not conceptual role recognition or mixed schema
compatibility.

On A54-R1 specifically, qwen-small passed M1, produced structurally valid M2
and M4 pairs with failing semantics, and passed the M3 semantic oracle.
qwen-large passed all four layers. Codestral passed M1, M2, and M4 including
their semantic oracles, but its M3 code failed semantics. Thus the historical
duplicate-CREATE production result does not reproduce in isolated M4 and is
attributable to full-session framing/continuity, not inability to express the
production schema. The production audit found that the mixed anchor listed only
existing edit candidates under “authorized mutation targets” and that generic
correction text omitted exact role/path bindings. A generic correction now
names both roles and paths, exact operation count, one child per path, and
LINE_RANGE/CREATE child types; the anchor includes both paths. Schema,
authority, budgets, correction count, and transaction mechanics are unchanged.

The one valid post-fix A54-R1 M5 attempt per model used LINE_RANGE. qwen-large
produced MODIFY app.py plus CREATE helper.py, reached preview and transaction,
and passed configured verification and the independent oracle. Codestral
produced the same valid pair and reached preview and transaction, but failed
verification and oracle (semantic quality). qwen-small's production session
returned, but the evaluator failed serializing its enum status before writing
the result; that run is not scored and was not repeated under the one-run
limit. Two earlier Qwen probes accidentally used the session's EXACT_TEXT
default rather than the A55 LINE_RANGE control; both duplicated CREATE and
are excluded from M4–M5 comparison. The evaluator serializer and representation
setting were corrected; the invalid probes are retained only as an explicit
instrumentation deviation. A55 makes no claim of universal qwen-small M5
success or broad semantic coding reliability.
