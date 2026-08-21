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

## Capability discovery

Callers inspect `ModelCapabilities` and query explicit `ModelCapability` values
instead of assuming that every model supports the same behavior. A1 declares
only chat, system-message, and seeded-generation capabilities.

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
