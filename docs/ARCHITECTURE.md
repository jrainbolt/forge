# Forge Architecture

This document records the architectural invariants that apply to Forge today.
It complements the development sequence in `ROADMAP.md`; it does not define
future APIs.

## Package layout and runtime resources

Forge uses a standard setuptools `src/` layout with explicit `package-dir` and
recursive package discovery. The `forge-coding-assistant` distribution installs the
`forge` import package and `forge = forge.cli:main` console entry point. Package
version metadata is derived from `forge.__version__`, which is also used by the CLI.
Neither the console script nor `python -m forge` changes `sys.path` or the process
working directory.

The controlled TinyQueue evaluation workspace is deliberate package data under
`forge.evaluation`. Runtime lookup uses `importlib.resources`, so installed evaluation
suites do not depend on the repository's `tests/` tree, README, scripts, or invocation
directory. User-selected `--workspace .` and relative configuration paths retain their
ordinary current-working-directory semantics.

Base Forge has no runtime dependencies and importing it does not import llama.cpp.
Pytest, Ruff, and the standards-based build frontend belong only to the `dev` extra;
`llama-cpp-python` belongs only to the separately requested `llama` extra. Wheels and
source distributions contain no model weights or user configuration.

## Model independence

Forge Core must not depend on a particular LLM or model family. Model-specific
behavior belongs outside the core.

## Backend independence

Inference implementations sit behind an abstraction so the core does not care
whether inference is local, remote, third-party, or eventually provided by
Forge Runtime.

## Forge Model API

Forge Core communicates with language models only through the synchronous
`Model` abstraction in `forge.models`. A request contains ordered conversation
messages and generic generation configuration. A response contains generated
text, a generic finish reason, model identity, and optional token usage.

Model objects are already initialized when presented to Forge Core. They expose
identity and capabilities without triggering loading, generate complete
responses synchronously, and provide an explicit, idempotent `close` operation
for releasing implementation-owned resources.

## Model vs Backend

A model is the language model serving a request. A backend is the engine that
executes that model. `ModelIdentity` preserves both identifiers as distinct
values so Forge can observe each without collapsing the concepts.

## Backend-specific data

Backend configuration, prompt rendering, native response dictionaries, model
file details, device settings, tokens, and runtime handles do not cross the
generic model boundary. Implementations translate between their native data
and Forge's request and response value objects.

## Model selection and construction

`ModelProfile` gives a stable name to a model identifier, backend identifier,
and opaque backend-owned configuration. `ModelCatalog` provides immutable
lookup and constructs only the selected profile. Listing profiles and reading
their metadata never loads model weights.

`BackendRegistry` is an explicit immutable dependency, not a process-global
service locator. Each `BackendDefinition` owns both validation of its native
settings and construction of its `Model` implementation. The default registry
is created afresh at the application composition boundary. Adding a backend
there does not add backend conditionals to Forge Core.

```text
application composition -> ModelCatalog -> BackendRegistry -> Model
Forge Core -----------------------------------------------> Model
```

TOML is parsed with the Python standard library. Top-level application
configuration (including the `FORGE_CONFIG` path override) remains separate
from generic generation settings and backend-owned execution settings.

## Conversation

`Conversation` owns ordered generic `Message` values as complete user/assistant
turns plus an optional system message. It creates `ModelRequest` values but does
not execute models or know about llama.cpp. Snapshots exposed to callers are
immutable tuples, and `/clear` removes turns while retaining the configured
system message.

## Chat session

`ChatSession` owns one generic `Model`, one `Conversation`, and generic
`GenerationConfig` for its lifetime. It exposes identity and context metadata,
executes requests synchronously, and closes the model explicitly. The CLI owns
terminal interaction and profile composition; the session owns neither input
nor output behavior.

System messages are sent only when the model declares
`ModelCapability.SYSTEM_MESSAGES`. Otherwise the session logs that the message
was omitted. The default text is deliberately general and describes no tools
or repository capabilities.

## Context policy

The generic `Model.context_capacity` property reports the configured capacity
when known without exposing a backend tokenizer or native context object. Forge
does not yet have a generic exact pre-generation tokenizer, so A4 labels its
input accounting as a conservative estimate: UTF-8 byte length divided by
three, rounded up, plus four tokens of per-message overhead.

The available input budget is configured context capacity minus the requested
maximum output and a 64-token safety reserve. A conservative 4,096-token
capacity is used when a backend cannot report one. The system message and
current user message must fit. Forge then includes the largest contiguous
suffix of recent complete turns that fits; it never splits messages, truncates
text, or leaves an orphan assistant response. After successful generation,
omitted oldest turns are removed from stored in-memory history. `/info` reports
the estimate method and number of omitted turns rather than claiming exact
token precision.

## Conversation failure and persistence

Turns are transactional. Forge constructs a request and calls the model before
committing either pending message. If generation fails, neither the user nor an
assistant message is added, so history contains only completed exchanges.

A4 conversation state is ephemeral and process-local. It is never written to
disk and cannot be resumed. There is no streaming, summarization, retrieval,
long-term memory, or agent state in the normal chat session layer.

## Tool execution boundary

Forge tools are inference-independent capabilities. Models and future agents
may eventually propose `ToolInvocation` values, but they never call a tool
implementation directly. Every invocation follows one central sequence:

```text
registry lookup
    -> argument validation
    -> permission decision
    -> invocation-specific approval enforcement
    -> tool execution
    -> structured result
```

`ToolRegistry` is explicitly constructed, rejects duplicate stable names, and
lists immutable safe metadata without execution. There is no global registry,
module scanning, entry-point discovery, or implicit capability import.

## Tool arguments and results

`ArgumentSchema` supports a deliberately small contract of required or optional
string, integer, and Boolean arguments. Unknown fields, missing required fields,
and incorrect types fail before permission evaluation or execution. Validation
returns a new immutable mapping and never mutates invocation-owned data.

Every invocation carries a caller-visible correlation ID, stable tool name, and
structured arguments. `ToolResult` preserves that correlation and distinguishes
success, execution failure, permission denial, and approval required. Expected
tool failures and unexpected implementation exceptions become structured
failure results at the executor boundary; native exception objects do not cross
it. Outputs are recursively immutable JSON-like values: scalars, sequences,
and text-keyed mappings may be nested without exposing mutable tool-owned data.

## Tool permission and approval

`PermissionPolicy` returns exactly `ALLOW`, `ASK`, or `DENY`. Rule-based policy
is deny-by-default: a tool without an explicit rule cannot silently execute.
`DENY` never executes. `ASK` also never executes unless the caller supplies an
explicit `InvocationApproval` matching the exact invocation ID, tool name, and
reviewed arguments. Approval does not mutate policy state, persist, authorize a
tool generally, cross invocation IDs, or survive changed arguments.

## Tool execution context and observability

`ExecutionContext` contains an explicit, normalized, absolute workspace
directory. It is immutable and independent of later process working-directory
changes. Repository paths are relative to that workspace. A central resolver
rejects absolute paths, resolves existing paths and symlinks strictly, and
uses path ancestry (not string prefixes) to reject traversal or symlink escape.
Internal symlinks may be followed only when their final target remains within
the workspace. Results expose workspace-relative paths.

The synchronous executor records the permission decision and monotonic duration
in generic metadata. Logs identify the tool, invocation, decision, and outcome,
but omit argument payloads by default because future arguments may contain
sensitive paths, contents, or commands.

The security invariants for every future tool are:

1. Models never directly execute tools.
2. Missing permission rules deny execution.
3. Approval is explicit and invocation-specific.
4. Arguments are validated before permission and execution.
5. A workspace is supplied explicitly rather than inferred globally.
6. `DENY` and unapproved `ASK` decisions cannot execute tools.
7. Results and failure states are structured and correlated.
8. Sensitive argument payloads are not logged by default.
9. Repository filesystem paths are resolved and confined before access.
10. Built-in A6 capabilities are explicitly classified as read-only.
11. User text never directly triggers tool execution.
12. Only strictly parsed model tool calls enter the executor.
13. Repository results are untrusted data and cannot alter policy.
14. Internal tool transcripts never enter durable conversation history.
15. Failed repository-aware turns commit no partial turn.
16. Repository orchestration has explicit step, tool, and repetition limits.
17. Unknown names never become executable capabilities.
18. Normal chat remains tool-free without an explicit workspace.

## Read-only repository capabilities

A6 supplies an explicitly composed registry containing only
`repository.list_directory`, `repository.read_file`,
`repository.search_files`, `git.status`, and `git.diff`. These capabilities
still pass through registry lookup, argument validation, policy evaluation,
approval enforcement, and the central executor. There is no arbitrary shell
tool, dynamic command construction, network capability, or write operation.

File reads and lexical search are bounded to 256 KiB per file. Search returns
at most 100 matches, truncates displayed matching lines at 500 characters,
skips common generated or metadata directories, and reports skipped files.
Directory listings are non-recursive and deterministic. Missing paths, special
files, invalid UTF-8, and paths that resolve outside the workspace become
structured tool failures.

Git tools use fixed argument arrays with `shell=False`, disable optional Git
locks and interactive prompting, disable pagers, and bound captured output to
256 KiB. Callers can choose only working-tree versus staged diff; they cannot
supply Git flags or commands. Git failures, including non-repository
workspaces, become structured tool failures.

Resolved-path confinement prevents stable traversal and symlink escapes. As
with ordinary path-based APIs, a hostile process that can replace filesystem
objects between validation and opening creates a time-of-check/time-of-use
race; stronger descriptor-relative operating-system primitives would be
required to defend against that concurrent mutation threat.

## Repository-aware orchestration

Supplying an explicit workspace selects `RepositoryChatSession`; omitting it
retains the tool-free `ChatSession`. Repository chat composes the generic
`Model`, `Conversation`, A6 registry, exact read-only policy, `ToolExecutor`,
and immutable `ExecutionContext`. It contains no backend- or model-family-
specific syntax and does not use native function calling.

Each model response is exactly one strict JSON envelope:

```json
{"type":"tool_call","id":"call-1","tool":"repository.read_file","arguments":{"path":"README.md"}}
```

Final answers use `{"type":"final","answer":"..."}`. The dedicated parser
requires the JSON object to occupy the entire response, validates its exact
fields, bounds payload size, and never executes anything. Prose, code fences,
multiple objects, user-entered protocol text, and model text resembling a tool
result do not become calls. Invocation IDs are validated and unique within a
turn, but carry no authority. One malformed or premature response may receive
one deterministic ephemeral correction; it consumes a normal orchestration
step, and any second violation fails the turn. Forge never chooses a tool on
the model's behalf.

Structured output is a generic model capability. A `ModelRequest` may carry an
immutable, backend-neutral JSON output specification. Repository orchestration
requires that capability and constructs its response schema from the actual
tool registry. The schema permits only registered tool names and their declared
argument types. It withholds final answers until evidence requirements are met,
withholds file reads until discovery returns candidate files, and constrains
paths and search terms to discovered or question-derived provenance. The
llama.cpp adapter alone translates this specification to its native
`response_format`; ordinary text requests remain unchanged.

Tool descriptions are rendered deterministically from actual registry metadata
and schemas. Executor results are rendered as deterministic JSON with
the invocation ID, tool, status, and structured output or safe failure. Native
exceptions and Python representations are not exposed. Only the orchestrator
creates trusted result messages.

## Internal repository transcript and transactions

Model tool calls and Forge tool-result messages form an ephemeral current-turn
transcript using ordinary assistant and user roles for backend portability.
They are included in subsequent model requests but never committed to
`Conversation`. A successful turn commits only the original user question and
final answer. Model, protocol, tool-loop, or context failures commit nothing
and preserve earlier completed turns.

Repository orchestration permits at most 12 model steps, 8 executor attempts,
and 2 identical calls per turn. Duplicate IDs fail immediately. All calls,
including unknown, denied, approval-required, validation-failing, and
execution-failing requests, still flow through `ToolExecutor`; ASK is never
auto-approved. The default policy explicitly allows exactly the five A6 tools
and denies every other name.

## Repository context budgeting and trust

The conservative A4 estimate includes the repository system instruction, tool
metadata, current question, retained completed turns, and the entire ephemeral
tool transcript. Older completed turns are omitted before current-turn
evidence. The question and tool framing are never pruned; if required evidence
cannot fit with output and safety reserves, the transaction fails clearly.
A6 file, search, and diff bounds remain the first line of context protection.

Model-visible results add tighter deterministic caps for file and diff text,
search matches and line text, and directory entries, with explicit truncation
flags. Search results prefer distinct files so one noisy file cannot consume
the entire evidence window.

Tool metadata classifies evidence as discovery, source content, Git working
state, or none. Directory listings and searches locate candidates but do not
prove implementation behavior. Git status and diffs describe only current
working state. Successful reads of relevant implementation source provide the
evidence required for a final answer; documentation reads remain discovery.
Questions about mechanisms or safety require distinct relevant source files so
the answer traces behavior across the implementation rather than relying on a
single incidental match. Failed reads and fabricated model text never count.

Repository contents and Git output are untrusted data. The system instruction
labels them accordingly, but the security guarantee is outside the model:
registry composition, explicit policy, executor enforcement, and A6 workspace
confinement remain authoritative even if repository text attempts prompt
injection.

A7 remains read-only. It adds no write, patch, shell, build, test, network,
parallel-call, planning, autonomous-agent, or persistence capability.

## Evaluation boundary

The A8 evaluation package is separate from normal chat and orchestration. Its
runner measures the production `RepositoryChatSession` against a controlled
fixture, so model requests still travel through structured output, provenance,
the central executor, A6 read-only tools, permission policy, and workspace
confinement. There is no evaluator-only repository access or relaxed benchmark
mode.

`coding-v1` defines immutable tasks with private, explicit ground truth:
required inspected files, normalized answer facts, expected file and symbol
references, and coarse tool thresholds. Prompts do not disclose those criteria.
Scoring is deterministic and reports correctness, grounding, localization,
efficiency, and completion independently. Forge does not use an LLM grader,
semantic grading, embeddings, or inferred evidence from answer prose.

One loaded generic model is reused across a run. The repository conversation is
cleared between independent tasks, and a failure is recorded without aborting
later tasks. Reports retain bounded final answers and execution metadata but not
source contents or internal transcripts. Versioned JSON is written only to an
explicitly requested path.

A8 establishes a local pre-write-access baseline for comparing compatible
models and future orchestration changes with identical prompts and scoring.
Normal CI exercises only deterministic scripted-model harness tests; it loads
no GGUF model and enforces no real-model score threshold.

## Controlled write capability boundary

A9 adds exactly `repository.write_file` and `repository.apply_patch`. They are
ordinary tool implementations behind registry lookup, argument validation,
permission evaluation, invocation-specific approval, structured results, and
the explicit workspace context. There is no generic filesystem writer, delete,
rename, directory creation, shell, build/test execution, or Git mutation.

Ordinary repository chat still composes only the five read-only tools. Explicit
assist mode composes those reads plus the two mutation tools under a separate
policy: reads are `ALLOW`, writes are `ASK`, and everything else is `DENY`.
Reading establishes provenance but never permission or approval. The terminal
shows a deterministic diff and target before asking `Approve? [y/N]`; only
explicit `y` or `yes` creates an approval bound to the exact invocation ID,
tool, and frozen arguments. Rejection, EOF, interruption, absent callbacks, and
changed invocations perform no mutation.

Existing-file replacement and patching require the lowercase SHA-256 of the
exact bytes returned by a successful current-turn read of that same path.
`repository.read_file` now reports this hash without changing its previous
fields. Stale or mismatched hashes fail before mutation. Creation has an
explicit `create` mode, rejects existing targets, requires an existing parent
and current-turn parent or related-source inspection, and never creates
directories. Replacement has an explicit `replace` mode and rejects missing or
non-regular targets. Both modes accept exact UTF-8 text up to 256 KiB without
newline normalization.

Patch arguments contain an ordered non-empty sequence of `{old, new}` text
edits for one file. Each old value must be non-empty and occur exactly once in
the incrementally proposed content. Missing, ambiguous, unchanged, malformed,
or oversized edits fail as a whole; there is no whitespace tolerance, fuzzy
matching, semantic patching, or multi-file transaction.

New files use exclusive creation. Existing files and patches are fully rendered
to a securely named same-directory temporary file, flushed, assigned the prior
file's mode bits, and atomically replaced with `os.replace`. Temporary files are
cleaned on failure. Forge then rereads the destination and verifies both exact
bytes and the new SHA-256 before returning success. Ownership, ACLs, and extended
attributes are not explicitly preserved. Common races are reduced by exclusive
creation and compare-before-write, but A9 does not claim descriptor-relative or
adversarial concurrent-filesystem hardening.

Write resolution extends the A6 path boundary for nonexistent final targets by
strictly resolving the existing parent and proving its ancestry. Absolute
paths, traversal and similar-prefix escapes, external or nested symlink-parent
escapes, final symlink targets, directories and special files are rejected.
Internal symlink parents may be used only when their resolved target remains in
the workspace. Any lexical or resolved path under `.git` is protected from
repository tools.

Successful mutation evidence is distinct from inspected source evidence. A
write invalidates prior read evidence and hash provenance for its path; a later
source explanation must reread it. At most one mutation proposal is exposed per
turn. The successful filesystem change is real even if later model generation
fails, while the conversation remains transactionally uncommitted; the REPL
reports that persisted state. A9 verifies file bytes and hashes only. It cannot
claim compilation, tests, or code correctness.

## Project command boundary

A10 adds exactly `project.build` and `project.test` as named verification
capabilities. Their zero-argument model schemas cannot carry a command, argv,
working directory, environment, timeout, or executable. Optional definitions in
the same trusted local TOML configuration use immutable argument arrays and a
positive bounded timeout. Missing definitions fail as `command_not_configured`;
Forge never discovers a build system or invents a default command.

Trusted configuration is user-owned local authority. Repository contents and model
output remain untrusted and cannot edit or replace the parsed command snapshot.
Forge discourages configuring shell executables but does not pretend trusted users
cannot do so. There is no generic shell, arbitrary named-command tool, package
installer, or model-controlled configuration interface.

## Controlled execution safety

Both tools run through the central `ToolExecutor` with the selected
`ExecutionContext.workspace` as `cwd`, an argv array, `shell=False`, and
`stdin=DEVNULL`. A copied environment sets noninteractive pager, Git-prompt, CI,
and unbuffered-Python controls; the model cannot inject environment variables.
The environment is not a security sandbox and may contain process-level values,
which Forge does not include in results. Forge itself initiates no network action,
but configured programs may have their own network behavior; A10 makes no network
sandbox claim.

On macOS and Linux each command starts a new session. Timeout sends SIGTERM to its
process group, waits briefly, then uses SIGKILL if necessary, covering descendants
that remain in that group. Every command has a configured or conservative default
timeout. Start failure, timeout, nonzero exit, and zero exit are distinct structured
outcomes; only a launched, non-timed-out zero exit is verification success.

Dedicated readers continuously drain stdout and stderr to avoid pipe deadlock while
retaining only the last 256 KiB of each stream. Truncation flags make discarded
prefixes explicit. Diagnostic bytes decode as UTF-8 with replacement, and common
ANSI control sequences are stripped. Results include operation, outcome, exit code,
timeout, duration, bounded streams, and per-stream truncation. Full output is not
logged; only execution metadata is logged.

## Execution permission and snapshot

Ordinary repository mode still registers only the five A6 read tools. Assist mode
registers reads, A9 writes, and the two A10 project tools: reads are `ALLOW`; writes,
build, and test are `ASK`; everything else is `DENY`. The terminal previews the exact
configured argv, workspace, and timeout. Only `y` or `yes` approves; default input,
rejection, EOF, interruption, missing callbacks, or a mismatched invocation starts no
process. Exact A5 approval remains bound to invocation ID, tool name, and frozen empty
arguments. The tool's immutable parsed command is the preview/execution snapshot, so
configuration is not reparsed between approval and launch.

## Verification evidence and staleness

`build_result` and `test_result` are distinct from source, Git-state, and mutation
evidence. Failed execution remains observed diagnostic evidence but never verifies
the project. Successful build/test evidence records the current Forge mutation
generation. Every successful Forge write or patch advances that generation and marks
prior verification stale; a later successful run verifies the new generation.
External human edits cannot all be detected, so this is not workspace snapshot
isolation. Existing write hash preconditions still guard Forge's own mutations.

Process output is untrusted data exactly like repository text. Strings resembling
instructions or tool calls cannot grant permission or directly trigger a tool; only a
subsequent valid model envelope can propose another registered operation, still under
policy and approval. Final-answer guidance distinguishes file mutation, build success,
test success, and failure, and permits verification claims only from successful
current-generation evidence.

A10 permits sequential user-approved actions within one bounded assist turn, such as
a write followed by a proposed test. It does not automatically run verification,
repair failures, retry, plan, or create an autonomous fix/test loop; those remain
later milestones.

## Single-step coding task

A11 upgrades explicit assist mode into a reusable single-step coding workflow through
`RepositoryChatSession.execute_task`. Normal chat remains tool-free and ordinary
workspace chat retains only the read registry. Each assist user request creates a new
small `CodingTaskState`; selecting assist mode is the deliberate coding-capability
boundary and does not introduce an agent or autonomous loop.

The state phases are `INSPECTING`, `AWAITING_MUTATION_APPROVAL`, `MUTATED`,
`VERIFYING`, `COMPLETED`, `FAILED`, and `REJECTED`. Terminal task statuses distinguish
verified, unverified, read-only, rejected, failed-before-mutation,
mutation-plus-verification-failure, and mutation-plus-later-task-failure outcomes.
Actual tool results populate mutation path/hashes, build/test records, generation, and
tool sequence; model prose does not populate external-state facts.

One user task may successfully execute exactly one `repository.write_file` or
`repository.apply_patch`. Reads remain bounded by the existing loop and are allowed
before and after mutation. Existing A9 hash and parent-context provenance, exact
preview, and approval remain unchanged. Rejection terminates the mutation phase, and a
second mutation request is denied by state even if a model ignores the dynamic schema.
The allowance resets for the next user request or `/clear`; neither operation rolls
back an already approved write.

Configured build and test operations retain separate approval. Each operation may run
once per workspace mutation generation, allowing old verification to be observed and
then invalidated by the single mutation. A failed build or test terminates the task;
subsequent verification or mutation requests are denied. A successful current-
generation result is required for verified status. Missing or rejected verification
produces an honest unverified result. There is no automatic rerun, repair, package
installation, command discovery, rollback, or second edit.

Only the final user/assistant messages enter conversation history. Tool protocol stays
ephemeral, while a separate immutable `CodingTaskResult` preserves bounded mutation
and verification metadata without source or process contents. Approved filesystem
changes persist if later model or orchestration work fails. The REPL prints an
authoritative `Change`/`Build`/`Tests`/`Status` footer, and also prints the last task
footer on a post-mutation exception, so dishonest or failed model text cannot override
external truth.

A11 preserves the no-shell, no-Git-mutation, no-package-install, and untrusted-tool-
output invariants. It does not plan, decompose tasks, perform multiple mutations, or
iterate after verification failure; those remain later milestones.

After mutation, coding state tracks a distinct verification decision:
`NOT_DECIDED`, `REQUESTED`, `COMPLETED`, or `DECLINED`. When at least one configured
verification capability is available, the first premature model final receives one
bounded structured correction asking for an appropriate `project.build` or
`project.test` request, or an explicit reason to decline. Forge never launches the
operation itself. A second final may deliberately decline verification; user rejection
also completes the decision without another prompt. With no configured verification,
an unverified final remains immediately valid. Post-mutation schemas continue to omit
both write tools.

Read-only repository chat retains its eight-tool limit. Assist coding tasks use a
modestly larger ten-tool limit to accommodate discovery, multiple relevant reads, one
mutation, an optional reread, and verification while preserving a hard bound.

## Agent loop v1

A12 adds an explicit foreground `--agent` mode without changing the authority of any
tool. The four user-visible autonomy modes are ordinary `CHAT`, repository `READ`,
single-step `ASSIST`, and bounded multi-step `AGENT`. Agent mode reuses the production
structured protocol, registry, executor, workspace confinement, provenance rules,
approval snapshots, coding-task state, and one-successful-mutation invariant.

`AgentTaskState` records phases, iterations/model calls, tool requests, unique source
files read, approvals, no-progress cycles, and the underlying mutation and verification
facts. A completed `AgentTaskResult` exposes those facts plus a machine-readable stop
reason. Stop reasons distinguish completion, rejection, cancellation, model/context/
protocol/tool failures, verification failure, repeated calls, no progress, independent
model/tool/iteration limits, and a blocked second mutation. Only model calls count as
iterations; every requested tool counts against the tool budget even when it is denied
or fails.

The default agent budgets are 16 iterations, 16 model calls, and 12 tool calls. Three
consecutive observations without new evidence stop the task, as does a third identical
tool request. Counters and progress history are fresh for each user task and remain
separate from bounded conversation history. The loop is sequential and synchronous;
there is no background scheduler, planner service, or parallel execution.

Writes and configured build/test commands remain `ASK`. The terminal presents the
same exact immutable preview used by assist mode, and Ctrl-C while awaiting approval
cancels the current agent task. Cancellation or rejection grants no authority. One
approved mutation exhausts the task's mutation allowance. After failed verification,
the agent may use remaining read-only calls to inspect and explain evidence, but repair,
retry, rollback, command discovery, shell, Git mutation, package installation, and
network access remain unavailable.

## Repair loop

A13 adds repair as an explicit `--agent --repair` capability. `--assist` and ordinary
`--agent` retain their A12 one-mutation behavior. Repair mode uses 24 iterations/model
calls and 18 tool requests while retaining no-progress and repeated-call guards. It
remains foreground and sequential.

Coding-task phases extend through `DIAGNOSING`, `AWAITING_REPAIR_APPROVAL`, `REPAIRED`,
and `VERIFYING_REPAIR`. Only a launched current-generation build/test ending in
`nonzero_exit` or `timeout` grants repair eligibility. `command_not_configured`,
`process_start_failed`, rejection, write failure, read failure, and model
reconsideration do not. Failure output remains untrusted diagnostic evidence.

Repair mutation #2 is the only corrective proposal. All observed hashes are cleared
after mutation #1, so the model must reread the current target and use its new SHA-256.
The repair may target the same or another freshly evidenced file. It receives the
normal exact A9 preview and invocation-specific approval. Rejection consumes the
repair opportunity; mutation #1 remains on disk.

Build and test attempts are tracked separately and capped at two per operation. A
command repeated after mutation generation changes is legitimate; repeating it in the
same generation is denied. Reverification receives a separate A10 approval. Passing
produces `COMPLETED_REPAIRED_VERIFIED`; rejection produces an honest unverified repair;
failure produces `REPAIR_VERIFICATION_FAILED` and ends the loop. Mutation #3 and
verification attempt #3 are impossible.

Agent results retain bounded mutation records, verification histories, the eligibility
outcome, approval counters, and repair outcome without full source or process content.
The deterministic footer reports both mutations and verification stages independently
of model prose. Superseded observations are discarded at mutation boundaries. There
is no transaction or rollback across writes: every approved successful mutation
persists after rejection, cancellation, context failure, or failed retest.

Repair adds no shell, Git mutation, package installation, command discovery, network,
automatic approval, background work, subagents, or parallel repair candidates.

## Verification failure diagnosis

In explicit repair mode, a launched current-generation verification ending in the
existing repair-eligible `nonzero_exit` or `timeout` outcome enters deterministic
failure diagnosis. The bounded A10 result—including configured operation identity,
exit code, timeout state, and truncated stdout/stderr—is retained as a trusted
`VERIFICATION_DIAGNOSTIC` observation. Its text remains untrusted data and grants no
tool, policy, or mutation authority.

## Fresh repair source

Forge performs at most one automatic, policy-allowed `repository.read_range` of the
successfully changed path for each eligible failed generation. The range uses the
A28 exact edit location with bounded surrounding context when available. The result
must establish the current full-file hash and generation; pre-primary source is stale
and cannot authorize repair.

## Repair-ready transition

`REPAIR_READY` requires both the bound verification diagnostic and the fresh current
source observation. Together they form immutable `RepairEvidence` containing only
observation identities, path, generation, and mutation index. The next model request
prioritizes the original task, diagnostic, and current excerpt while exposing only
the A28 structured edit and its narrowly authorized reread alternative. Broad
discovery is corrected without execution.

## Repair authority

A13 remains the sole authority for entering repair: ordinary assist failures,
process-launch errors, missing commands, permission failures, and rejected execution
do not reach `REPAIR_READY`. The source and diagnostic establish provenance, not
permission. Repair still receives a new exact preview and invocation-specific
approval and remains bound to the changed primary path in A30 v1.

## Repair reverification

After the sole corrective mutation, A29 immediately runs the configured verification
again without another model decision. Success produces the existing repaired-verified
status; failure is terminal. A third mutation is impossible, and no rollback is
performed.

## Autonomy and permission profiles

A14 separates two immutable axes. `AutonomyMode` selects orchestration and defines the
maximum capability set: `CHAT` has none, `READ` has repository reads, `ASSIST` has the
single-step one-mutation task, `AGENT` has bounded multi-step one-mutation behavior,
and `REPAIR` has A13's conditional two-mutation loop. `PermissionProfile` independently
maps explicitly classified READ, WRITE, BUILD, and TEST tools to A5 `ALLOW`, `ASK`, or
`DENY` decisions.

Autonomy is always the ceiling. A profile cannot expose a category absent from its
mode, so CHAT stays tool-free and READ stays mutation/execution-free even under
`trusted-exec`. Tools permanently denied by the selected profile are omitted from the
fixed registry and structured schemas. ASK and ALLOW tools remain visible subject to
the task phase. Every production tool declares a semantic capability; unclassified
tools deny by default.

Built-ins are `safe` (READ allow; all else deny), `confirm` (READ allow; WRITE/BUILD/
TEST ask), and `trusted-exec` (READ/BUILD/TEST allow; WRITE ask). No built-in profile
automatically allows writes. ALLOW removes only the approval prompt: registry lookup,
argument validation, workspace confinement, coding/repair state, generations, attempt
budgets, timeouts, output bounds, and termination guards remain authoritative.

`InteractionPolicy` is the immutable composition passed to the registry, session, and
executor. It is selected only by the application/user, logged at session creation, and
never reread during a task. Model output, repository text, command diagnostics, or a
later configuration-file edit cannot alter the active snapshot. Config-file defaults
and custom profiles are intentionally not added in A14; CLI explicit values take
precedence over conservative mode defaults (`safe` for CHAT/READ, `confirm` otherwise).

The preferred CLI is `--mode MODE --permissions PROFILE`. Legacy workspace/assist/
agent/repair flags translate once at the application boundary and cannot be mixed with
`--mode`. Trusted execution applies only to predefined immutable project commands; it
adds no shell, model-controlled argv, network, package, Git-write, or filesystem-delete
capability.

## Repository analyzers and symbol intelligence

A15 adds a small model-independent analyzer boundary. `PythonAnalyzer` uses the
standard-library AST to extract source-ordered class, function, async-function,
method, and nested-definition locations. Non-Python UTF-8 files receive only an
explicit generic text classification; Forge does not apply regex language parsers or
claim structural support for them.

Four read-only tools expose this boundary. `repository.file_outline` returns bounded
definitions and actual AST line ranges. `repository.find_symbol` performs exact simple
or file-local qualified-name matching. `repository.find_references` returns Python AST
reference candidates with bounded snippets and containing definitions; it is not a
complete semantic call graph and does not promise cross-module type or alias
resolution. `repository.read_range` reads an inclusive 1-based line interval and
returns the SHA-256 of the complete current file.

Outline, definition, and reference results are discovery evidence. Their returned
paths may authorize a subsequent confined source read, but locations alone do not
satisfy implementation grounding. A successful range read is source-content evidence,
and its full-file hash may establish the same current-turn write provenance as a full
file read. Existing compare-before-write checks still reject stale hashes.

All paths use the A6 resolver and walker, including ignored-directory and symlink
rules. Structural requests scan at most 2,000 files, parse at most 512 KiB per file,
return at most 500 outline definitions, 100 symbol matches, or 200 reference
candidates, and bound snippets to 300 characters. Range reads permit at most 400 lines
and 128 KiB of returned text. Parser failures are explicit for direct outlines and are
counted and skipped during workspace searches.

Milestone A17 persists the A15 metadata in a versioned SQLite database beneath the
platform cache directory. A SHA-256 of the canonical workspace path supplies the
workspace identity. The schema stores file-relative paths, size/mtime diagnostics,
content hashes, parse status, symbol ranges, and syntactic reference locations; it
never stores complete source text or snippets.

The first structural query builds the index lazily. Every later structural query
walks the confined workspace and hashes eligible Python files, reparsing only added or
content-changed files and deleting vanished entries in one transaction. A valid build
is replaced atomically. Schema mismatch or corruption causes a derived-cache rebuild;
cache or lock failures cause the structural tool to use its existing bounded direct
scan. Successful Forge writes invalidate the affected row, but write success never
depends on index maintenance.

Indexed outline, definition, and reference results remain discovery evidence. Source
files and their current hashes remain authoritative, `repository.read_range` stays a
direct source read, and indexed data cannot satisfy write provenance. All four tools
declare `ToolCapability.READ`; CHAT remains tool-free, structural discovery grants no
mutation or execution authority, and model or repository instructions cannot expand
the fixed policy.

## Context planning and admission

A18 separates complete bounded task events from the active model context.
`ContextPlanner` assigns each tool observation a deterministic type, generation,
paths, estimated token/byte/line cost, and retention priority. Before each repository
model call it builds an active snapshot inside the existing output and safety reserves.
The generic estimator remains explicitly approximate; no backend tokenizer or second
model is required.

The tool pipeline is state legality, context preflight, permission/approval, execution,
observation registration, and deterministic compaction. Whole-file and range reads are
never silently transformed. Reads that cannot fit receive a structured failure with
targeted alternatives and any known structural window. Small whole-file reads remain
legal. Permission and context decisions are independent.

Successful targeted source compacts broad search payloads. Mutation removes stale
pre-mutation source text while existing hashes and task provenance remain authoritative.
Successful verification is retained as bounded metadata; current failure diagnostics
remain high priority until a later-generation result supersedes them. Required current
source, mutation, and verification evidence is protected from budget-pressure dropping.
CHAT continues to use the ordinary A4 conversation path without repository planner
state.

## Capability discovery

Callers inspect `ModelCapabilities` and query explicit `ModelCapability` values
instead of assuming that every model supports the same behavior. A1 declares
only chat, system-message, and seeded-generation capabilities.

## Model adapter

`LlamaCppModel` is the first real implementation of the generic Forge model
API. It translates Forge messages and generation settings to
`llama-cpp-python`, then normalizes generated text, finish state, identity, and
available token counts into a `ModelResponse`.

The dependency direction is strictly Forge Core to the generic Model API to the
llama.cpp adapter. Forge Core never imports `llama_cpp`, and backend-native
objects do not cross the model boundary.

## Backend configuration

`LlamaCppConfig` owns execution-specific settings: an explicit local model
path, context size, GPU-layer request, optional thread count, diagnostics, and
optional explicit model identifier. None of these settings are part of generic
generation configuration. The adapter reads the supplied model file but does
not discover, download, move, or modify models.

## Chat formatting

The adapter sends structured Forge messages to llama.cpp's chat-completion API.
Chat-template adaptation belongs to the backend and model metadata. Forge does
not render model-specific prompt markers, and the adapter rejects a GGUF model
without chat-template metadata instead of allowing llama.cpp's fallback format.

## Optional inference dependency

`llama-cpp-python` is isolated in the `llama` installation extra and imported
only when constructing the adapter. Base Forge installations, the CLI, unit
tests, and hosted CI remain usable without the native inference dependency.

## Semantic retrieval

A19 adds a separate generic embedding boundary and an optional lazy llama.cpp
adapter for user-owned local GGUF embedding models. Semantic configuration is
independent from generative model profiles. Query execution is offline and never
downloads a model.

Deterministic structural and line-window chunks are stored in `semantic.sqlite3`
beside, but not inside, the A17 structural index. The semantic database contains
locations, hashes, model identity, and little-endian float32 vectors; it does not
store source text or pickle data. Model identity, dimensions, or chunker-version
changes cause an atomic semantic rebuild without altering structural state.

`repository.semantic_search` is conditionally exposed as READ/DISCOVERY. Its ranked
locations and recommended ranges are navigation hints only. Current source from
`repository.read_range` or `repository.read_file` remains required for grounded
implementation claims and mutation provenance.

## Tool isolation

Models cannot directly access the filesystem, shell, Git, network, compilers,
or other external capabilities. Such access will occur only through explicit,
separately testable Forge tools.

## Explicit permissions

External actions will pass through explicit permission policies enforced
outside model instructions. Authority must not be inferred from model output.

## Workspace confinement

Read-only repository access is bounded to explicitly selected workspaces.
Future code execution and file mutation must preserve or strengthen that
boundary. Access to one workspace must not imply access elsewhere.

## Observable execution

Important model, tool, and agent activity must be inspectable, including
requests, results, changes, errors, timing, and relevant configuration.

## Structured state

Conversation state, task state, model context, repository state, tool history,
and persistent knowledge are distinct concepts. They must not collapse into a
single unbounded message history.

## Local-first

Forge must remain useful without cloud services. Any future network access is
an optional, explicitly controlled capability.

## Portable core

macOS and Apple Silicon are the first target, but Forge Core remains portable.
Platform-specific behavior belongs behind clear boundaries.

## Correctness before optimization

Correct, tested behavior precedes optimization. This is especially important
for the future native inference runtime, where reference implementations must
anchor optimized CPU, SIMD, and GPU implementations.
# Hybrid retrieval and source-aware reranking

Semantic repository search uses a two-stage, model-independent retrieval path.
The embedding backend first supplies a bounded semantic candidate pool (four
times the requested result count, with a floor of 20 and ceiling of 80). A pure
deterministic reranker then combines dominant cosine similarity with small
lexical, structural, and source-kind signals. The fixed policy is versioned as
`retrieval_ranking_version = 1`; it is intentionally not user-configurable.

Lexical evidence is derived from normalized query, path, symbol, qualified
symbol, and current bounded source text. Tokenization handles snake_case,
camelCase, dotted names, and path components without an NLP dependency.
Implementation symbols receive a modest structural/source prior, while semantic
similarity remains strong enough that a much better semantic match cannot be
overturned by metadata alone. Exact locations and identical chunk hashes are
deduplicated, ties are stable, and a three-results-per-file first pass preserves
useful result diversity when alternatives exist.

`SourceKind` is the centralized classifier for implementation, test,
documentation, configuration, generated metadata, and other text. The shared
repository walker excludes package metadata such as `.egg-info`, `.dist-info`,
`PKG-INFO`, and `SOURCES.txt`. Semantic refresh consequently deletes stale rows
for newly ineligible files without rebuilding unchanged embeddings. The SQLite
cache remains source-free; source text is read only for the bounded live
candidate pool and is neither persisted nor returned as discovery evidence.

The repository tool returns the raw cosine as both `similarity` and
`semantic_similarity`, plus `source_kind`. Ranking component scores remain an
internal diagnostic detail. If candidate validation rejects data, the index
falls back to deterministic raw semantic order; unrelated programming failures
are not suppressed.

## Retrieval strategy

Repository orchestration owns a deterministic `RetrievalStrategy` separate from
ranking and context admission. Its compact states distinguish unstarted
retrieval, discovery without candidates, available candidates, an exact target,
acquired source, and exhaustion. Trusted tool results—not repository text or a
model classifier—drive transitions.

The strategy retains at most 32 immutable, generation-aware candidates with
path/range/symbol identity, discovery source, priority, and inspection state.
Exact duplicates are ignored; identical candidate sets provide no novelty.
Successful reads inspect matching candidates, ordinary read failures mark them
failed, and context-budget rejection leaves them unresolved. Mutation marks
changed-path candidates stale without invalidating safe candidates elsewhere.

Dynamic response schemas intersect registry capabilities with routed tools.
Concrete semantic, lexical, reference, directory, or symbol candidates suppress
broad discovery until targeted inspection or refinement occurs. Exact symbols
produce a target-identified state. Candidate failure or exhaustion reopens
discovery. Acquiring one source file does not discard unresolved candidates, so
multi-file evidence requirements remain possible.

Routing can only narrow the autonomy/permission capability ceiling. Candidate
metadata remains discovery evidence, current source reads remain necessary for
grounding and write hashes remain necessary for mutation provenance. A20 ranks
candidates, A21 manages their workflow, A18 decides context admission, and the
permission system remains independently authoritative.

## Task evidence plans and coverage

A22 represents bounded task obligations with an immutable `TaskEvidencePlan`
of at most four `EvidenceGoal` values and a separate mutable
`EvidenceCoverageState`. Production decomposition is deliberately conservative.
Explicit numbered or bulleted items and independent semicolon clauses become
separate goals. The narrowly recognized `How do X and Y work together`,
`interact`, or `connect` form becomes two source goals plus a dependency-only
relationship goal. Ordinary uses of `and` and ambiguous prose remain one goal.
Plans are frozen at task start; explicit trusted plans take precedence.

Discovery marks only discovery state. A successful trusted source read covers
only the active associated goal when its centralized source kind satisfies that
goal. Relationship goals become covered only after their declared source goals.
Coverage metadata stores paths, generation, evidence type, source kind, and
observation identity—not source contents. Changed-path evidence is invalidated
without disturbing unchanged goals.

Goals are processed sequentially. Covering one goal reopens A21 retrieval for
the next, while retained source observations remain available to A18 for final
synthesis. A final response requires both the existing grounding rule and all
required goals; one premature final receives bounded missing-goal guidance.
Coverage proves that evidence was gathered, not that the model interpreted it
correctly, and it grants no tool or write authority.

Discovery candidates are associated with the goal active when Forge receives
the trusted result. A read cannot cover another goal unless that path was also
discovered for that goal. Two bounded empty discovery attempts mark the active
required goal failed and produce an explicitly incomplete result. Source
observations retain the context planner's highest priority for final synthesis
when the model context can hold them.

## Goal-directed bootstrap

A23 gives each unresolved source-bearing evidence goal one optional Forge-owned
discovery execution per workspace generation. A compatible current semantic index
has priority; otherwise A26 selects the fast lexical provider when available.
Eligibility requires no actionable A21 candidates and an effective `ALLOW`
permission. Forge uses the bounded goal description directly as the query and
executes it through `ToolExecutor`; ASK, DENY, missing providers, empty results,
and failures fall back to ordinary model-directed discovery.

Bootstrap results remain discovery-only. They enter A20 candidate ranking, A21
routing, A18 context admission, and A22 goal association through the same paths as
model-requested semantic results. The model still chooses every source inspection.
Bootstrap consumes one normal tool execution but no additional model call, cannot
authorize writes, and never runs for dependency-only relationship goals. A22 owns
obligations, A23 seeds discovery, A20 ranks, A21 routes, and A18 bounds context.

## Fast repository discovery

A26 maintains a rebuildable SQLite repository map separate from the Python-only
structural index and model-bound semantic vectors. It recognizes common source,
header, test, configuration, and documentation formats without parsing a language.
Generated/build directories, binary files, oversized files, and symlinks are
excluded; vendored source is retained with an explicit third-party classification.

## Lexical retrieval

The versioned tokenizer splits paths, punctuation, snake case, and camel case.
The index stores hashes, path/basename tokens, bounded content-token counts,
document frequency, and a few representative line positions. It never stores
repository source text. Ranking combines path and basename coverage, content,
rarity, goal-aware source-kind preference, and deterministic path/line tie breaks.

## Bootstrap provider selection

A compatible semantic index remains the preferred bootstrap provider. A missing,
cold, or incompatible semantic index selects `repository.lexical_search`; absence
of both providers leaves discovery to the model. The lexical index builds lazily
and all providers share the existing one-tool-per-goal-generation and tool-budget
rules.

## Authority

Lexical candidates are hints, not evidence. A21 associates candidates with the
active A22 goal, while A24 finalization still requires a trusted confined source
read. Tokens cannot grant permissions or extend read/write capability.

## Incremental freshness

Per-file SHA-256 values make unchanged refreshes reuse all postings. Only added or
changed files are tokenized, and deleted paths are removed. Incompatible or corrupt
derived state is rebuilt from the authoritative workspace.

## Limitations

Lexical discovery is not semantic understanding, a symbol parser, or a call graph.
Its bounded representative ranges can only direct the model toward source that
must subsequently be read and interpreted.

## Evidence-locked finalization

A24 introduces a repository task phase separate from A21 retrieval state. A
read-only task enters `FINALIZING` only when trusted A22 coverage is complete,
ordinary source grounding is satisfied, and no required goal has failed. The next
generic structured-output request uses a final-only JSON schema with no model-visible
tool metadata. A tool call emitted anyway never reaches `ToolExecutor`; one bounded
correction asks for the final answer and a repeated violation fails safely.

The final synthesis snapshot contains the original task, covered-goal labels and
paths, and one current A18-owned source observation per required source-bearing goal.
Discovery chatter and failed ranges are omitted. Shared active observations are not
duplicated, and failure to fit balanced required evidence raises a context-budget
error. Current hashes are rechecked before read-only finalization; concurrent edits
outside Forge remain limited to this synchronous check rather than filesystem
watching. A24 guarantees evidence availability, not perfect model interpretation.

A22 defines required evidence, A23 seeds discovery, A24 ends retrieval and locks
synthesis, while A18 continues to own context budgeting. Coding, agent, and repair
state machines remain authoritative and retain their established final behavior.

## Coding transition

A27 extends the existing A11 coding state with `MUTATION_READY`. In ordinary
single-step assist tasks, a successful current source read that satisfies the active
evidence requirement can move the task from inspection to mutation preparation.
Discovery metadata and model assertions cannot cause this transition.

## Mutation candidate

A mutation candidate is a bounded immutable path, current SHA-256, workspace
generation, source-observation identity, and optional trusted range. At most four
current candidates are retained. They are separate from A21 retrieval candidates:
only an A9-authoritative source read creates write provenance.

## Tool narrowing

While mutation-ready, the dynamic schema exposes `repository.apply_patch` for the
current candidates and, only for an incomplete trusted range, one exact targeted
`repository.read_range`. Broad search, semantic/lexical discovery, outlines,
references, directory listing, and unrelated reads are withheld and cannot execute.
One premature final or broad-search attempt receives bounded guidance; repetition
fails truthfully.

## Authority

Forge never authors or automatically calls a patch. The model must emit the existing
patch protocol. `ToolExecutor`, compare-before-write hashing, workspace confinement,
the permission policy, exact preview, and invocation-bound approval remain the only
mutation authority.

## Invalidations

Before each mutation-ready model call Forge rechecks candidate hashes. Missing or
changed source invalidates readiness, advances the workspace generation, removes
stale provenance, and returns to evidence acquisition. Successful mutation hands
control back to the unchanged A11 verification and A13 repair state machines.

## Structured mutation proposal

In mutation-ready orchestration the model supplies one immutable,
language-independent `path`/`old_text`/`new_text` replacement. The path must be a
current A27 candidate, the source hash and generation must still match, and the
old text must occur exactly once inside the source-backed candidate range. Each
text is limited to 16 KiB and 200 lines, with a 24 KiB combined limit.

## Deterministic materialization

Forge reads the confined UTF-8 candidate without normalization, verifies the exact
match, and translates the proposal into one existing `repository.apply_patch`
edit. The unchanged A9 preview constructs the unified diff. LF/CRLF, tabs, Unicode,
and all bytes outside the replacement remain untouched.

## Structured mutation authority

The model chooses the replacement code. Forge validates its representation and
provenance, the user or policy approves the exact preview, and A9 `ToolExecutor`
performs the mutation. Raw patch calls remain available to direct A9 callers but
are absent from primary mutation-ready model schemas.

## Structured mutation recovery

One malformed, mismatched, ambiguous, out-of-range, or oversized proposal receives
one correction using retained trusted source. A second invalid proposal terminates
truthfully. Stale source instead invalidates A27 readiness and requires a fresh
read. Repair uses the same validator after its required fresh-source read.

## Structured mutation limitations

Version 1 changes one exact occurrence in one file. It has no regex, AST editing,
free-form diff scraping, insertion primitive, multi-hunk proposal, or multi-file
transaction.

## Mutation intent anchoring

Before a primary or repair mutation proposal, Forge builds a compact model snapshot
that keeps the original requested behavior beside the trusted mutation target and
exact current source observation. Retrieval candidate lists, unrelated source,
failed discovery attempts, documentation results, and routing corrections are not
carried into this snapshot. Repository text remains inert evidence.

## No-op edit and recovery

Exact `old_text == new_text` is classified as `NO_OP_EDIT`. After materialization,
Forge independently requires the resulting source to differ from current source.
Neither failure creates a preview, requests approval, executes a tool, changes the
target, or reopens discovery. One shared structured-edit correction reiterates the
original task, current trusted source, and non-identity requirement; a second bad
proposal fails truthfully. Primary and repair mutations use the same mechanism,
with one recovery opportunity for each mutation attempt.

Forge validates that a delta exists, but never infers, rewrites, or supplies the
semantic change. A28 path, generation, hash, range, exact-match, preview, approval,
and execution authority remains unchanged.

## Real-world evaluation authority

The A25 evaluation harness is outside normal runtime authority. It creates a fresh
non-Git repository copy for each task, applies exact-precondition fixture changes,
injects an approval callback restricted to declared paths and exact project command
snapshots, and independently hashes and verifies the disposable result. Evaluator
oracle subprocesses use fixed argument arrays with `shell=False`; they are not tools
exposed to the model. This adds no shell, write, Git, network, repair, or approval
authority to ordinary Forge sessions.

## Mutation representations

Mutation-ready orchestration explicitly selects either the default exact-text
proposal or the optional line-range proposal. The selection comes from trusted
caller or model-profile configuration; it is never inferred from a model name and
does not grant additional capabilities.

Both representations converge before `MutationPreview`. For a line-range proposal,
Forge resolves the 1-based inclusive range against current trusted UTF-8 source,
derives exact old text internally, and then enters the same A28 validator and A9
preview, approval, permission, compare-before-write, and execution path.

Line numbers are authority only within the current mutation candidate's path,
SHA-256, workspace generation, and observed line bounds. Stale, Boolean, reversed,
oversized, missing, or out-of-observation ranges are rejected. Range selection does
not widen write authority, permit fuzzy matching, or enable multi-range or multi-file
mutation.

## Verification attribution

A36 optionally runs one pre-mutation verification through the same trusted A10
`project.test` (or configured `project.build`) tool, policy, approval, workspace,
timeout, and tool budget as ordinary verification. It is off by default. The
immutable baseline evidence binds the selected operation, argv, workspace, timeout,
effective process environment, and Forge pre-mutation generation. Repository text
and model output cannot supply baseline evidence. A baseline run is a real tool
request, not a free evaluator observation.

After A29 executes post-mutation verification, a passing result needs no
attribution. A failure with a comparable passing baseline is
`MUTATION_ASSOCIATED`; identical bounded failure fingerprints are
`PREEXISTING_OR_UNRELATED`; differing or incomplete fingerprints are
`UNATTRIBUTED`. Unavailable baselines and changed command identity never establish
equivalence. Fingerprints use explicit failure lines plus structured A10 outcome,
exit, and timeout fields; normalization removes only ANSI escapes and the disposable
workspace prefix. Truncated or marker-free output cannot prove a matching failure.

## Attribution truth and repair boundary

A matching pre-existing failure blocks A13 repair eligibility from that failure.
Mutation #1 remains applied, but its terminal status is
`MUTATED_VERIFICATION_BLOCKED_BY_BASELINE_FAILURE`, never
`COMPLETED_VERIFIED`. Mutation-associated and unattributed failures retain the
existing bounded repair rules; A36 grants no new repair authority. The independent
realworld oracle can evaluate this decision but cannot change production status.
Forge's internal generation cannot detect every concurrent external filesystem edit;
baseline comparison assumes the selected workspace remains under the same task's
control until mutation.

## Verification plans

A37 permits an explicit immutable `[project.verification]` plan with an ID and one
to four ordered references to already-configured A10 project operations. A38 adds
the optional `project.configure` prerequisite; a configure-only plan is invalid.
The plan is loaded from trusted local configuration or
an evaluator-owned task definition. Repository text and model output cannot
create a plan or supply commands. Without an explicit plan, A29 continues to
select one configured operation, preferring test.

Each step executes serially through the existing ToolExecutor in the active
workspace, with its own A14 ALLOW/ASK/DENY decision, exact per-operation approval,
timeout, controlled environment, and tool-budget charge. No model turn occurs
between steps. A failed, denied, rejected, or unavailable prerequisite stops the
plan. Successful prerequisite output is compacted to status metadata; the bounded
failing-step output remains available to A30 repair grounding.

The orchestration budget reserves one mutation slot plus the configured number
of verification steps while discovery is still active, without raising the
ceiling. A mutation is `COMPLETED_VERIFIED` only after every plan step actually
passes in the current workspace generation. A repair mutation restarts the full
plan at step one, preserving the two-mutation ceiling. If A36 baseline mode is
enabled, it runs the same ordered plan before mutation; attribution compares a
failed step only when plan identity, command identity, and all preceding passing
steps match. The independent evaluation oracle has no production status authority.

## Trusted project configuration

A38 adds optional `[project.commands.configure]` with the same immutable argv and
bounded timeout format as build/test. It is a distinct, workspace-mutating
`CONFIGURE` capability, not READ or source WRITE. Safe policy denies it, confirm
asks per exact prepared invocation, and trusted-exec allows it. The model cannot
supply argv, cwd, timeout, or environment. Execution uses the existing A10
ToolExecutor and subprocess boundary (`shell=False`, workspace cwd, controlled
environment, DEVNULL stdin, bounded output, and process-group timeout cleanup).
No package-install, clean, hook, or generic shell operation is added.

An explicit plan may now run configure→build→test. Each step consumes one tool
execution and advances only after a pass; successful configure output is compacted
in model context, while bounded failure diagnostics remain available for truthful
reporting. Configure failure does not authorize source-code repair. A later
repair-eligible build/test failure may lead to one repair mutation and a complete
configure→build→test rerun. Optional A36 baseline mode runs the same trusted plan
before mutation and compares only matching plan, step, command, and predecessor
identities. Without a plan, legacy build/test selection is unchanged.

Generated build paths such as `build/` and `CMakeFiles/` are classified as generated
metadata, not implementation evidence merely because configure created them.
Forge fixes subprocess cwd to the authorized workspace and configures no separate
output directory or cleanup authority. In the default A10/A38 mode, trusted
commands are not OS-filesystem-sandboxed; project owners must configure commands
whose outputs stay within the workspace. A39 adds optional isolation below.

## Trusted process isolation

A39 distinguishes trusted command configuration from subprocess isolation.
Optional `[project.execution] isolation` selects `none` (the backward-compatible
default), `controlled_env`, or `strict` for all configured project operations in
that task. The model and repository evidence cannot select or change this policy.
Every configure/build/test still passes through the same ToolExecutor permission,
approval, timeout, and process-group cleanup path; no tool-budget or model-call
allowance changes.

## Environment hardening

`controlled_env` and `strict` construct a bounded subprocess environment instead
of inheriting unrelated parent variables. A fixed allowlist retains PATH, locale,
selected system/toolchain variables, and nonsecret user identifiers. Forge sets
PATH from absolute entries outside the active workspace, dropping relative and
workspace-local entries so repository content cannot supply a PATH executable.
Forge sets
noninteractive values, redirects HOME and TMPDIR/TMP/TEMP to workspace-local
`.forge-exec/home` and `.forge-exec/tmp`, and rejects pre-existing symlinks at
those directory nodes. The prepared approval snapshot binds the effective
environment, isolation mode, adapter identity, workspace, argv, and timeout.
Environment values are not logged. `.forge-exec/` is excluded from discovery and
classified as generated metadata, not source evidence.

## Strict isolation and authority

`strict` requires an available Forge-owned OS adapter and never falls back to
`controlled_env`. The macOS adapter uses the installed Seatbelt `sandbox-exec`
facility with deny-by-default policy: runtime filesystem reads are allowed,
filesystem writes are allowed only below the resolved workspace, and no network
permission is granted. Forge constructs the policy; neither repository content
nor the model supplies it. Where the adapter cannot start, Forge reports an
isolation failure without running the command. This is a filesystem-write and
network boundary, not full read confinement: a strict process may read files
outside the workspace that its user can already read. The adapter is platform-
specific; other platforms report strict unavailable. `controlled_env` redirects
ambient paths and removes secrets from the environment but is not filesystem
sandboxing. Isolation constrains an already-authorized process; it grants no
execution or source-write permission. A36 attribution identity includes mode,
adapter, effective environment, HOME, and TMPDIR so changed conditions are not
treated as comparable.

## Strict toolchain isolation diagnostics

A40 versions the macOS policy as `macos-sandbox-exec-toolchain-v2`. It adds only
the exact `sysctl-read hw.pagesize_compat` capability required by the installed
Apple linker. A permissive diagnostic Seatbelt profile linked successfully while
the former strict profile failed, macOS logs recorded the named denial, and a
one-rule diagnostic profile restored linking and Foundation configure without
weakening the external-write boundary. No arbitrary sysctl, Mach-service,
external-write, or network capability was added.

Forge's strict failure taxonomy recognizes explicit deny markers for read, write,
process, executable mapping, and system-service operations, and distinguishes
sandbox startup failures. The adapter identity is the strict policy version bound
to approvals and A36 attribution, so old-policy approvals and baselines do not
compare as current. Its declared capabilities remain all runtime reads, process
operations, workspace writes, and no network grant. Deterministic fake-adapter
tests exercise strict orchestration contracts but do not prove real compiler
compatibility; suite output explicitly labels its aggregate scope. Real macOS
checks passed query, compile, link, execution, and clean Foundation
configure→build→test. Write regressions before and after acceptance allowed the
workspace and denied sibling and disposable home-hierarchy writes. Strict remains
fail-closed and `controlled_env` behavior is unchanged.
