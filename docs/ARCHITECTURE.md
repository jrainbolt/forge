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
long-term memory, tool execution, or agent state in the session layer.

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
it. Outputs are limited to scalar values or immutable mappings of text keys to
scalar values.

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
changes. A5 establishes this ownership boundary only; path confinement and
repository access arrive with real read-only tools in A6.

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

A5 provides framework machinery and deterministic, harmless test tools only.
There are no production filesystem, repository, Git, shell, process, network,
web, or external-application tools, and chat is not connected to this framework.

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

Future code execution and file mutation must be bounded to explicitly selected
workspaces. Access to one workspace must not imply access elsewhere.

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
