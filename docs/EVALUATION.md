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

The real macOS ladder used `/usr/bin/cc` (Apple clang 21.0.0) and the existing
`macos-sandbox-exec-v1` adapter. Strict compiler query passed in 0.06 seconds,
compile-to-object passed in 0.06 seconds, and link failed in 0.09 seconds with
Apple `ld`'s header-alignment assertion. Executable run was not possible because
the strict link produced no executable. Trivial CMake configure and clean
Foundation configure failed at the same compiler-link stage. No specific linker
sandbox denial was observed in bounded stderr or the available macOS logs; these
results are classified `strict_unknown_failure`. No permission was expanded.

The final real macOS boundary regression still allowed a workspace write and
denied disposable sibling and home-hierarchy writes. A40's strict Foundation
acceptance remains blocked at configure; no build/test step or unisolated fallback
was attempted. The A39 `controlled_env` Foundation run remains the accepted path.
