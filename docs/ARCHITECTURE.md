# Forge Architecture

This document records the architectural invariants that apply to Forge today.
It complements the development sequence in `ROADMAP.md`; it does not define
future APIs.

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
