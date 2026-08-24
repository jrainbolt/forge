# Forge Architecture & Development Roadmap v1

**Status:** Initial architecture and implementation roadmap
**Primary target:** Apple Silicon MacBook Pro, M4 Pro, 48 GB unified memory
**Primary purpose:** Local-first coding AI and software-development agent
**Secondary purpose:** General conversational AI
**Development model:** Milestone-driven implementation with Codex, architecture review between milestones

---

# 1. Project Vision

Forge is a local-first AI system designed primarily to perform real software-engineering work against local repositories.

Forge should eventually be capable of:

* conversing with the user;
* understanding and navigating software repositories;
* locating relevant code;
* explaining implementations and architecture;
* creating and modifying source code;
* interacting with Git;
* compiling projects;
* running tests;
* analyzing failures;
* iterating on changes;
* performing bounded autonomous coding tasks;
* retrieving project-specific knowledge;
* using interchangeable language models;
* running using an existing inference backend initially;
* eventually running models through Forge's own native inference runtime.

Forge is **not** defined by any particular LLM.

The language model is a replaceable component of the larger Forge system.

---

# 2. Core Architecture

The long-term architecture is:

```text
                         User
                          │
                          ▼
                 ┌─────────────────┐
                 │ Forge Interface │
                 │                 │
                 │ CLI initially   │
                 │ IDE later       │
                 └────────┬────────┘
                          │
                          ▼
                 ┌─────────────────┐
                 │   Forge Core    │
                 │                 │
                 │ sessions        │
                 │ orchestration   │
                 │ permissions     │
                 │ context         │
                 └───────┬─────────┘
                         │
          ┌──────────────┼───────────────┐
          │              │               │
          ▼              ▼               ▼
     Forge Model      Forge Tools    Forge Context
        API                              System
          │
          ▼
    Model Backend
       /     \
      /       \
 existing      Forge Runtime
 backend            │
      │             │
      ▼             ▼
 open-weight     native C
   models        inference
```

---

# 3. Architectural Principles

These principles should be treated as project invariants.

## 3.1 Model independence

Forge Core must never depend directly on a particular model family.

Agent code must not contain assumptions such as:

```text
Qwen behaves like...
Llama requires...
DeepSeek outputs...
```

Model-specific behavior belongs behind model adapters or capability interfaces.

---

## 3.2 Backend independence

The Forge model interface must not care whether inference is provided by:

* llama.cpp;
* another local inference library;
* Forge Runtime;
* a test/mock model;
* potentially a remote backend in the future.

Existing inference software is an implementation dependency, not an architectural dependency.

---

## 3.3 Tool isolation

Models cannot directly:

* read files;
* modify files;
* invoke shell commands;
* use Git;
* access the network;
* run compilers;
* execute tests.

All external actions occur through explicit Forge tools.

```text
Model
  │
  ▼
Forge tool request
  │
  ▼
permission check
  │
  ▼
tool execution
```

---

## 3.4 Explicit autonomy

Agent autonomy is a configurable policy.

Initial conceptual modes:

```text
CHAT
    No external tools.

READ
    Read-only repository tools.

ASSIST
    May propose modifications or commands.
    User approval required for execution.

AGENT
    May execute approved classes of operations
    autonomously inside a defined workspace.

UNRESTRICTED
    Broad tool access.
    Must be explicitly enabled.
```

Exact names and policies may evolve, but autonomy must never be an implicit Boolean hidden inside the agent loop.

---

## 3.5 Workspace confinement

Forge operates against explicitly defined workspaces.

A coding agent authorized for:

```text
~/projects/foo
```

must not silently obtain write access to arbitrary locations elsewhere on the machine.

Filesystem permission enforcement must occur outside the model.

---

## 3.6 Observable execution

Important activity must be inspectable:

* model selected;
* backend selected;
* prompt/context construction;
* tool requests;
* tool results;
* file modifications;
* commands executed;
* test results;
* agent iterations;
* errors;
* timing;
* token usage where available.

Forge should never become a system whose failures can only be explained by saying:

> "The AI decided to do something."

---

## 3.7 Structured state

The following concepts remain separate:

* conversation history;
* active task state;
* repository state;
* model context;
* tool execution history;
* persistent project information;
* long-term memory.

They should not become one unbounded message list.

---

## 3.8 Local-first operation

Forge should function as a useful coding agent without requiring cloud inference or network access.

Network capabilities may later be implemented as explicit tools.

---

## 3.9 Portable core

macOS and Apple Silicon are the first-class target.

However:

```text
Forge Core
Forge Model API
Forge Runtime Core
```

should not unnecessarily depend on Apple-specific APIs.

Platform-specific functionality should remain behind backend boundaries.

---

## 3.10 Correctness before optimization

Especially for Forge Runtime:

```text
correct scalar implementation
        ↓
tests
        ↓
optimized CPU implementation
        ↓
SIMD
        ↓
Metal
```

A fast incorrect inference engine is not progress.

---

# 4. Language Strategy

Forge intentionally uses multiple languages where appropriate.

## Python

Primary responsibilities:

* CLI;
* orchestration;
* model abstraction;
* tool definitions;
* repository operations;
* Git integration;
* subprocess execution;
* context assembly;
* agent loop;
* evaluation infrastructure;
* training/fine-tuning experiments.

## C17

Primary responsibilities:

* Forge Runtime;
* tensor representation;
* model loading;
* tokenizer execution where appropriate;
* transformer inference;
* memory management;
* CPU kernels;
* quantization;
* SIMD optimization.

## Metal / platform bridge

Later responsibilities:

* GPU compute kernels;
* Apple Silicon acceleration.

## TypeScript

Potential future responsibility:

* VS Code extension.

---

# 5. Model Strategy

Forge begins with existing open-weight models.

Initial goals:

* support interchangeable models;
* support local execution;
* support quantized models;
* support coding-oriented models;
* allow general instruction models.

Initial preferred distribution format:

```text
GGUF
```

GGUF compatibility does **not** mean Forge Runtime must permanently mirror llama.cpp internals.

Forge should interpret model formats through an explicit loading layer.

---

# 6. Capability Model

Different models may support different capabilities.

Examples:

```text
text generation
chat templates
structured output
tool calling
context length
system messages
streaming
reasoning modes
specific tokenizer behavior
```

Forge must expose capabilities rather than assume every model supports everything.

Conceptually:

```text
ModelCapabilities
    supports_chat
    supports_tools
    supports_streaming
    context_length
    ...
```

---

# 7. Tool Architecture

Every tool should eventually expose at minimum:

```text
name
description
input schema
permission class
execution handler
```

Initial tool categories:

```text
READ
WRITE
EXECUTE
NETWORK
DESTRUCTIVE
```

Early coding tools should include capabilities equivalent to:

```text
read_file
list_directory
search_files
git_status
git_diff
```

Later:

```text
write_file
apply_patch
run_build
run_tests
run_command
```

And eventually:

```text
web_search
fetch_url
search_documents
```

Tools should be individually testable without invoking an LLM.

---

# 8. Context Architecture

Forge should avoid feeding an entire repository to the model.

Repository understanding should evolve through several levels.

```text
direct file selection
        ↓
repository search
        ↓
symbol-aware search
        ↓
dependency-aware context
        ↓
retrieval/indexing
        ↓
semantic retrieval
```

Context selection is part of Forge intelligence and should be separately observable and testable.

---

# 9. Coding Agent Execution Model

A mature Forge coding task should resemble:

```text
User task
   ↓
understand request
   ↓
inspect repository
   ↓
select relevant context
   ↓
develop plan
   ↓
perform bounded edits
   ↓
build / test
   ↓
inspect failures
   ↓
revise
   ↓
verify
   ↓
present result
```

Autonomous execution must always have:

* a workspace;
* a permission policy;
* an iteration limit;
* a termination condition.

No unbounded agent loops.

---

# 10. Evaluation Strategy

Forge should be evaluated continuously rather than judged subjectively.

A dedicated fixture repository should eventually contain controlled coding tasks.

Example progression:

```text
E01  answer a repository question
E02  locate an implementation
E03  explain a function
E04  identify a known bug
E05  write a unit test
E06  implement a small function
E07  repair a failing test
E08  modify multiple files
E09  perform compile/test/fix iteration
E10  complete a bounded coding task autonomously
```

Evaluation should capture information such as:

```text
success/failure
model
model configuration
elapsed time
tool calls
agent iterations
tokens
files changed
test result
```

This will later allow meaningful comparisons between models.

---

# 11. Forge Repository Roadmap

Two development tracks proceed in parallel but at different times.

```text
TRACK A
Forge Agent
Useful coding AI

TRACK B
Forge Runtime
Native inference engine
```

Track A comes first.

Forge Runtime must not block delivery of a useful coding assistant.

---

# TRACK A — FORGE AGENT

# Milestone A0 — Project Bootstrap

**Goal:** Establish the Forge repository, engineering rules and executable skeleton.

Deliverables:

* project structure;
* Python environment;
* basic CLI;
* test framework;
* linting/formatting strategy;
* architecture documentation;
* configuration conventions;
* logging foundation.

Expected result:

```bash
forge --help
```

works locally.

No LLM integration yet.

---

# Milestone A1 — Generic Model Interface

**Goal:** Define the abstraction between Forge and any model implementation.

Implement:

* model request;
* model response;
* message representation;
* generation configuration;
* model capabilities;
* backend lifecycle;
* deterministic mock backend.

Tests must prove that Forge Core depends only on the generic interface.

No real model required yet.

---

# Milestone A2 — First Local Model Backend

**Goal:** Execute a real open-weight model locally.

Implement an adapter to an existing local inference backend.

Expected interaction:

```bash
forge chat
```

then:

```text
> Explain a mutex.

A mutex is...
```

The exact initial model is deliberately not architecturally significant.

---

# Milestone A3 — Model Selection and Configuration

**Goal:** Demonstrate model interchangeability.

Support:

* configured model paths;
* backend selection;
* model metadata;
* per-model parameters;
* capability validation.

Acceptance requirement:

The same Forge CLI can run two different compatible models without Forge Core modifications.

---

# Milestone A4 — Session and Conversation Model

**Goal:** Establish correct state handling for normal chatbot use.

Implement separation between:

* session;
* conversation messages;
* model request;
* generated response.

Add:

* history truncation strategy;
* context accounting where available;
* streaming if backend support makes it appropriate.

At this point Forge is a usable local chatbot.

---

# Milestone A5 — Tool Framework

**Goal:** Establish generic tool execution without coding autonomy.

Implement:

* tool registry;
* tool definitions;
* structured inputs;
* structured results;
* permission classes;
* validation;
* logging.

Use deterministic test tools before filesystem mutation.

---

# Milestone A6 — Read-Only Repository Tools

**Goal:** Allow Forge to understand a local code repository safely.

Initial capabilities:

```text
list_directory
read_file
search_files
git_status
git_diff
```

Enforce workspace confinement.

Network access remains disabled.

---

# Milestone A7 — Repository-Aware Chat

**Goal:** Answer questions about the active repository.

Examples:

```text
Where is configuration loaded?

Explain this class.

Find where retries are implemented.

Which tests exercise this function?
```

Forge should select repository context rather than receiving the entire repository.

This is the first milestone where Forge becomes meaningfully useful for programming work.

---

# Milestone A8 — Coding Evaluation Harness v1

**Goal:** Create objective measurements before adding write autonomy.

Build a fixture repository and initial evaluation tasks for:

* repository Q&A;
* code localization;
* explanation;
* bug identification.

Evaluation must run against different models.

---

# Milestone A9 — Controlled Write Tools

**Goal:** Allow source modification under explicit user approval.

Introduce:

```text
write_file
apply_patch
```

Requirements:

* workspace restrictions;
* diff visibility;
* permission checks;
* rejection handling;
* atomic or recoverable writes where practical.

Default mode remains non-autonomous.

---

# Milestone A10 — Build and Test Tools

**Goal:** Let Forge validate its own coding changes.

Add controlled execution for:

```text
build
test
selected commands
```

Commands must not initially be arbitrary unrestricted shell access.

Expected workflow:

```text
request
  ↓
Forge proposes patch
  ↓
user approves
  ↓
Forge modifies files
  ↓
Forge runs tests
  ↓
Forge reports result
```

---

# Milestone A11 — Single-Step Coding Tasks

**Goal:** Complete bounded coding changes without iterative autonomy.

Examples:

```text
add this unit test
rename this symbol
implement this small function
fix this obvious defect
```

One planning/execution cycle only.

Evaluation expands accordingly.

---

# Milestone A12 — Agent Loop v1

**Goal:** Introduce bounded iterative execution.

Core loop:

```text
inspect
  ↓
reason
  ↓
act
  ↓
observe
  ↓
reason
  ↓
act
```

Required safeguards:

* maximum iterations;
* maximum tool calls;
* explicit termination;
* clear errors;
* permission enforcement;
* cancellation.

---

# Milestone A13 — Compile/Test Repair Loop

**Goal:** Forge can use compiler and test output to repair its own changes.

Canonical task:

```text
modify code
   ↓
compile
   ↓
failure
   ↓
inspect output
   ↓
repair
   ↓
compile
   ↓
tests
```

This is a major functional milestone.

---

# Milestone A14 — Agent Autonomy Modes

**Goal:** Implement configurable autonomy as a first-class subsystem.

Formalize:

```text
CHAT
READ
ASSIST
AGENT
UNRESTRICTED
```

or revised equivalents.

Permission policy must be enforced independently of model instructions.

---

# Milestone A15 — Planning and Multi-File Tasks

**Goal:** Handle larger coding work.

Add explicit task plans containing concepts such as:

```text
goal
steps
affected files
verification
status
```

Forge should be able to perform a bounded multi-file implementation and verify it.

---

# Milestone A16 — Repository Intelligence v2

**Goal:** Improve context selection for larger projects.

Possible capabilities:

* source symbol indexing;
* definitions/references;
* import/include relationships;
* test relationships;
* tree-sitter or language-aware parsing;
* repository summaries.

This milestone should remain language-extensible.

---

# Milestone A17 — Persistent Repository Index v1

**Goal:** Reuse repository structure across queries without weakening source authority.

Implemented as a local, versioned SQLite cache of A15 file, symbol, and syntactic
reference metadata. It refreshes incrementally, recovers from corrupt or incompatible
derived state, and falls back to bounded direct scanning when the cache is unavailable.
No complete source text is stored, and indexed discovery does not authorize writes.

Future retrieval work may still add abstractions such as:

```text
keyword retrieval
symbol retrieval
embedding retrieval
hybrid retrieval
```

Forge Core remains independent of embeddings.

---

# Milestone A18 — Context Planning & Budget Management v1

**Goal:** Admit, retain, narrow, and compact repository observations deliberately.

Implemented with a deterministic application-layer context planner that accounts for
estimated tokens, bytes, and lines; preflights source reads; recommends symbol and
reference windows; compacts superseded discovery and verification payloads; and keeps
authoritative task/provenance metadata separate from the model-visible snapshot.

The planner adds no retrieval source, model call, authority, embeddings, or semantic
summarization.

---

# Milestone A19 — Coding Evaluation Harness v2

**Goal:** Establish serious model and agent benchmarking.

Expand tests to include:

* single-file changes;
* multi-file changes;
* bug repair;
* test creation;
* compiler-feedback repair;
* autonomous completion.

This becomes Forge's regression suite for AI behavior.

---

# Milestone A20 — External Information Tools

**Goal:** Add optional network access.

Potential capabilities:

```text
web search
URL retrieval
documentation lookup
```

Network use must have its own permissions.

Forge should remain fully usable with network disabled.

---

# Milestone A21 — Retrieval Strategy & Tool Routing v1

**Status:** Implemented and deterministically closed by the production
`routing-v1` orchestration evaluation.

Forge now tracks a bounded, generation-aware queue of discovery candidates and
deterministically narrows repository tool schemas toward targeted inspection.
Failed or exhausted candidates reopen discovery, while successful source reads
retain unresolved candidates for multi-file tasks. Routing only removes tools
from the existing autonomy and permission ceiling; it adds no capability or
model-based planner.

---

# Milestone A22 — Task Decomposition & Evidence Coverage v1

**Status:** Implemented.

Forge tracks bounded deterministic evidence goals independently from retrieval
ranking and routing. Trusted source observations cover only associated goals;
multi-goal plans advance sequentially, dependency and source-kind requirements
are enforced, changed-path coverage becomes stale, and finalization requires
both existing grounding and complete required-goal coverage. The fixed
`coverage-v1` suite validates these behaviors without an LLM planner or judge.
Normal requests conservatively decompose explicit lists, semicolon tasks, and
recognized two-aspect relationship questions into production evidence goals.

---

# Milestone A23 — VS Code Integration

**Goal:** Use Forge directly inside the development environment.

Potential capabilities:

* chat panel;
* selected-code context;
* repository questions;
* diff display;
* approve/reject patch;
* launch agent task;
* view tool activity.

Forge Core remains independent of VS Code.

---

# TRACK B — FORGE RUNTIME

Track B begins only after Forge Agent has a stable model API and useful coding functionality.

---

# Milestone R0 — Runtime Architecture

**Goal:** Define the native inference engine boundaries before implementation.

Specify:

```text
model loader
tensor representation
execution context
memory ownership
tokenizer boundary
backend interface
sampler
generation API
```

Define strict separation between:

```text
runtime core
CPU backend
Metal backend
model format loader
```

---

# Milestone R1 — Tensor Core

**Goal:** Establish correct multidimensional tensor representation.

Implement:

* shapes;
* strides;
* storage;
* ownership;
* views where appropriate;
* scalar data types;
* validation.

No transformer yet.

---

# Milestone R2 — Reference CPU Math

**Goal:** Implement correct scalar numerical primitives.

Examples:

```text
vector operations
matrix multiplication
normalization primitives
activation functions
```

Optimize nothing prematurely.

Compare results against trusted references.

---

# Milestone R3 — Model File Inspection

**Goal:** Read model metadata and tensors from the initial supported model format.

Likely initial target:

```text
GGUF
```

Implement:

* header parsing;
* metadata;
* tensor descriptors;
* validation;
* bounds checking.

Do not execute a model yet.

---

# Milestone R4 — Tokenizer

**Goal:** Correctly transform text into model tokens and tokens back into text.

Build tokenizer tests against a trusted implementation.

---

# Milestone R5 — Minimal Transformer Forward Pass

**Goal:** Execute one transformer layer using reference CPU code.

Implement required operations incrementally.

Correctness tested against a known implementation.

---

# Milestone R6 — Full Model Forward Pass

**Goal:** Compute model logits from input tokens.

Still reference CPU implementation.

No concern yet for useful generation speed.

---

# Milestone R7 — Autoregressive Generation

**Goal:** Generate text.

Implement:

```text
prompt evaluation
logits
sampling
next token
repeat
```

Forge Runtime can now produce model output.

---

# Milestone R8 — KV Cache

**Goal:** Avoid recomputing the complete prompt during every token generation step.

Implement explicit KV-cache ownership and lifecycle.

Validate output equivalence with uncached execution.

---

# Milestone R9 — Sampling System

**Goal:** Support configurable generation behavior.

Potential features:

```text
greedy
temperature
top-k
top-p
seeded sampling
```

Sampling must be independent from transformer implementation.

---

# Milestone R10 — Forge Runtime Model Backend

**Goal:** Connect Forge Agent to Forge Runtime.

This is the convergence milestone.

```text
Forge Agent
      │
Forge Model API
      │
Forge Runtime Backend
      │
native C inference
```

The existing inference backend remains available.

Forge's evaluation suite runs against both.

---

# Milestone R11 — Profiling Infrastructure

**Goal:** Measure before optimizing.

Track:

```text
model load time
prompt evaluation time
tokens/sec
memory use
hot kernels
cache behavior
```

---

# Milestone R12 — CPU Optimization

**Goal:** Improve performance without changing inference semantics.

Potential work:

* memory layout;
* blocking;
* cache-aware kernels;
* reduced allocations;
* threading.

Each optimization must preserve reference results within defined numerical tolerances.

---

# Milestone R13 — Quantized Tensor Support

**Goal:** Load and execute quantized models efficiently.

Start with a limited quantization format rather than implementing everything.

Reference comparisons remain mandatory.

---

# Milestone R14 — Apple Silicon SIMD

**Goal:** Use appropriate CPU vector capabilities on the M4 Pro.

Possible technologies should be evaluated when this milestone is reached rather than assumed now.

Maintain scalar/reference kernels for validation.

---

# Milestone R15 — Metal Backend Foundation

**Goal:** Establish GPU compute execution.

Implement:

* device initialization;
* buffer lifecycle;
* kernel dispatch;
* synchronization;
* CPU/GPU backend abstraction.

Start with isolated tensor operations.

---

# Milestone R16 — Metal Matrix Operations

**Goal:** Accelerate dominant inference kernels.

Benchmark against optimized CPU execution.

---

# Milestone R17 — Metal Transformer Execution

**Goal:** Move major transformer operations to the GPU.

CPU fallback remains available.

---

# Milestone R18 — Hybrid Execution Optimization

**Goal:** Determine efficient CPU/GPU division for Apple unified memory.

Optimize based on measurement rather than assumptions.

---

# Milestone R19 — Runtime Compatibility Expansion

**Goal:** Increase supported model architectures and quantization types.

Only after the initial runtime works end-to-end.

---

# Milestone R20 — Runtime Production Hardening

**Goal:** Make Forge Runtime reliable enough to become Forge's preferred local backend.

Focus on:

* malformed model handling;
* memory failure behavior;
* deterministic tests where applicable;
* model compatibility tests;
* performance regression tests;
* long-context stability;
* repeated load/unload cycles.

---

# TRACK C — MODEL DEVELOPMENT

This track is deliberately deferred.

Forge should first become good at executing existing models.

Possible future stages:

```text
C0  LoRA experimentation
C1  coding-specific fine-tuning
C2  Forge-generated evaluation datasets
C3  synthetic coding-task generation
C4  reinforcement/evaluation experiments
C5  training a small model from scratch
C6  architecture experiments
```

Training a competitive foundation model from scratch is **not** an initial Forge objective.

---

# 12. Long-Term Architecture

A mature Forge installation could ultimately resemble:

```text
                         Forge
                           │
        ┌──────────────────┼───────────────────┐
        │                  │                   │
       CLI             VS Code             Local API
        │                  │                   │
        └──────────────────┼───────────────────┘
                           │
                     Forge Core
                           │
          ┌────────────────┼────────────────┐
          │                │                │
       Agent            Context           Tools
          │                │                │
          │          ┌─────┴─────┐    ┌─────┴──────┐
          │          │ retrieval │    │ filesystem │
          │          │ projects  │    │ git        │
          │          │ memory    │    │ build      │
          │          └───────────┘    │ test       │
          │                           │ network    │
          │                           └────────────┘
          │
          ▼
     Forge Model API
          │
     ┌────┼──────────┐
     │               │
 llama.cpp       Forge Runtime
 backend              │
                      ├── CPU
                      ├── SIMD
                      └── Metal
```

---

# 13. Milestone Development Protocol

Forge will be implemented one milestone at a time.

For each milestone, the architecture/planning step will provide Codex with:

```text
Objective

Current system state

Required architecture

Required implementation

Constraints

Explicit non-goals

Tests

Acceptance criteria

Completion-report format
```

Codex should implement only the active milestone.

The completion report is returned for architecture review.

Review outcomes are:

```text
PASS

PASS WITH NOTES

CORRECTION REQUIRED

ARCHITECTURAL CORRECTION REQUIRED
```

The next milestone does not begin until the current milestone is accepted.

---

# 14. Codex Rules

Codex should generally be instructed to:

* inspect the existing repository before editing;
* preserve established architecture;
* avoid speculative abstractions;
* avoid implementing future milestones;
* run all relevant tests;
* report every file created or modified;
* identify deviations from the prompt;
* identify unresolved concerns;
* avoid claiming success when tests have not been run;
* stop and report genuine blockers instead of improvising architectural changes.

---

# 15. Initial Development Sequence

Although the full roadmap is intentionally long, the immediate sequence is small:

```text
A0  Project Bootstrap
 ↓
A1  Generic Model Interface
 ↓
A2  First Local Model Backend
 ↓
A3  Model Selection
 ↓
A4  Conversation
 ↓
A5  Tool Framework
 ↓
A6  Read-Only Repository Tools
 ↓
A7  Repository-Aware Chat
```

At **A7**, Forge should already be useful.

The next major capability sequence is:

```text
A8   Evaluation
 ↓
A9   Controlled Writes
 ↓
A10  Build/Test
 ↓
A11  Coding Tasks
 ↓
A12  Agent Loop
 ↓
A13  Autonomous Repair
```

At **A13**, Forge becomes a genuine local coding agent.

Only after the agent architecture is established should Forge Runtime become a major parallel development effort.

---

# 16. Definition of Early Success

Forge v0 should not be judged by whether it rivals commercial frontier coding agents.

The first meaningful success state is:

```text
$ forge

> Where is retry handling implemented?

Forge searches the active repository,
reads relevant files,
and explains the implementation accurately.
```

The next success state is:

```text
> Add a test for this edge case.

Forge determines the relevant test file,
proposes a patch,
receives approval,
applies it,
runs the test suite,
and reports the result.
```

The first major agent success state is:

```text
> Fix this failing test.

Forge:
    investigates
    edits
    compiles
    observes failure
    revises
    reruns tests
    verifies success
    presents the final diff
```

The first major runtime success state is:

```text
Forge performs the same task
using Forge Runtime
instead of a third-party inference engine.
```

---

# 17. Current Starting Point

The GitHub repository exists and has already been cloned locally.

Therefore the next implementation task is:

**Forge Milestone A0 — Project Bootstrap**

Milestone A19 adds optional local semantic repository retrieval: a generic embedding
API, deterministic chunking, source-free incremental vector persistence, conceptual
search, context-planner compaction, and explicit semantic index lifecycle commands.
It preserves exact structural/text lookup precedence and requires targeted source
reads before semantic candidates become evidence.

No model should be downloaded and no inference code should be written until A0 establishes the repository structure, architecture documents, Python project configuration, CLI skeleton, tests, and engineering conventions.
