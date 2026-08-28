# Forge

Forge is an early-stage, local-first AI coding assistant and software-development
agent. Its long-term design keeps language models interchangeable and keeps the
core useful without cloud services. The current implementation provides the
project bootstrap, a generic backend-independent model API, named model
profiles, an optional llama.cpp adapter for real local inference, ephemeral
multi-turn chat sessions, and permission-controlled repository-aware chat using
workspace-confined read, controlled-write, and configured build/test tools.

## Development setup

Forge requires Python 3.12 or later. Create an isolated environment and install
the project with its development tools:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

This editable development install uses the standard `src/` layout and requires no
`PYTHONPATH` setting. A regular local installation is also supported:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install .
forge --version
```

The distribution name is `forge-coding-assistant`; the import package and console
command are both `forge`. Forge is not published to PyPI, so these commands install
the local checkout.

Run the CLI:

```bash
forge --help
python -m forge --help
```

Run the quality checks:

```bash
pytest
ruff check .
ruff format --check .
```

The same test, lint, and formatting checks run in GitHub Actions for pushes and
pull requests on Python 3.12.

On macOS, if an editable install reports success but `forge` raises
`ModuleNotFoundError`, inspect the generated `site-packages/__editable__*.pth` with
`ls -lO`. Python ignores a `.pth` carrying the macOS `hidden` flag. This is external
filesystem metadata, not a Forge import requirement; remove that flag from the
affected virtual environment or recreate the venv under a directory that does not
propagate it. Do not set `PYTHONPATH` as a permanent workaround.

## Optional local inference

The base installation remains lightweight and does not install an inference
engine. To use the llama.cpp adapter, install Forge's bounded `llama` extra:

```bash
python -m pip install -e ".[llama]"
```

Forge is validated against `llama-cpp-python` 0.3.35 and accepts compatible
0.3.x updates; the `<0.4` bound prevents an unreviewed breaking release from
changing the adapter contract.

On Apple Silicon, first confirm that both the machine and Python report `arm64`:

```bash
uname -m
python -c 'import platform; print(platform.machine())'
```

The official `llama-cpp-python` project offers Metal wheels for macOS and
Python 3.12. To explicitly select that wheel source when installing, use:

```bash
python -m pip install -e .
python -m pip install "llama-cpp-python>=0.3.35,<0.4" \
  --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/metal
```

Alternatively, a source build can enable Metal with
`CMAKE_ARGS="-DGGML_METAL=on"`. Runtime GPU offload also requires a nonzero
`gpu_layers` value; Forge defaults to `-1`, requesting all supported layers.
Backend initialization logs are the source of truth for whether Metal and GPU
offload are active.

Download compatible GGUF files manually and retain ownership of their location.
Forge neither downloads nor modifies models. The A2 reference is
`Qwen/Qwen2.5-Coder-7B-Instruct-GGUF`, Q4_K_M quantization, but the model and
path are not hard-coded. Local `*.gguf` files are ignored by Git.

Run one non-interactive smoke request:

```bash
python scripts/smoke_model.py /path/to/model.gguf \
  --model-id Qwen2.5-Coder-7B-Instruct --verbose
```

Or run the opt-in integration test:

```bash
FORGE_TEST_MODEL=/path/to/model.gguf pytest -m integration
```

## Named model profiles

Copy [`forge.example.toml`](forge.example.toml) to a location outside the
repository and replace its model paths. The example contains the two A3
reference profiles, `qwen-small` and `qwen-large`. Keep GGUF weights and the
machine-specific configuration outside version control. A recommended local
layout is `~/Models/forge/` for both the weights and an untracked `forge.toml`.

Inspect profiles without loading model weights:

```bash
python scripts/list_models.py /path/to/forge.toml
```

Run the same A3 coding request through either profile. `--config` takes an
explicit path; `FORGE_CONFIG` is the application-level environment override.

```bash
python scripts/smoke_profile.py qwen-small --config /path/to/forge.toml
FORGE_CONFIG=/path/to/forge.toml python scripts/smoke_profile.py qwen-large
```

The catalog validates all profile and backend settings while loading TOML, but
constructs only the selected model. Load and generation timings are written to
standard error.
The first two-profile observations are recorded in
[`docs/MODEL_COMPARISON.md`](docs/MODEL_COMPARISON.md).

## Interactive chat

Launch a local session with an explicit profile and configuration:

```bash
forge chat --model qwen-small --config ~/Models/forge/forge.toml
FORGE_CONFIG=~/Models/forge/forge.toml forge chat --model qwen-large
```

Interactive generation defaults to 256 output tokens and temperature 0.4.
Generic overrides are available through `--max-tokens`, `--temperature`, and
`--seed`; `--no-system` omits the concise default Forge system message.

The REPL supports:

```text
/help   show commands
/clear  clear in-memory conversational turns without reloading the model
/info   show profile, model, context, and estimated-budget information
/exit   close the model and exit
```

Ctrl-D and Ctrl-C also exit cleanly. The selected model is loaded once and
reused for every turn in the session. Responses are synchronous and appear
after generation completes. Conversation history exists only in memory and is
not saved or resumable.

Without an explicit workspace, Forge chat has no tools or repository access.
It remains a general local conversation interface; model answers do not grant
external capabilities.

To enable repository-aware chat, select the workspace explicitly:

```bash
forge chat \
  --model qwen-small \
  --config ~/Models/forge/forge.toml \
  --workspace .
```

Without `--workspace`, chat retains its normal tool-free behavior. With a
workspace, Forge may list directories, read bounded UTF-8 files, perform bounded
lexical searches, inspect Git status, and read working-tree or staged diffs.
The model uses strict backend-independent JSON output specifications, while
every request still passes through the deny-by-default permission policy and
central executor. Discovery results identify candidates; implementation claims
require relevant source-file evidence, and Git output counts only as working-
state evidence.
Internal tool calls and repository contents are ephemeral and are not saved in
conversation history or on disk. Repository mode requires model-requested tool
evidence before accepting a final answer and permits only one bounded protocol
correction per turn.

Repository mode is read-only. Forge still cannot edit files, apply patches, run
configured builds or tests, mutate Git, access the network, or use the web.

Explicit assist mode is Forge's bounded single-step coding mode. It adds previewed,
individually approved file mutations and configured project verification:

```bash
forge chat \
  --model qwen-large \
  --config ~/Models/forge/forge.toml \
  --workspace . \
  --assist
```

Normal `--workspace` mode remains read-only. In assist mode, reads are allowed,
Forge may inspect source, propose at most one successful code mutation for each user
request, reread the result, and optionally propose configured build or test
verification. Every write, patch, build, or test proposal receives a separate default-no
approval prompt. Writes show their target and deterministic diff. Existing files
require the SHA-256 returned by a current-turn read; approving one exact invocation
never approves changed content or another path.

`project.build` and `project.test` accept no model arguments. Their immutable argv
arrays and timeouts come from the trusted local TOML configuration:

```toml
[project.commands.build]
argv = ["python", "-m", "compileall", "src"]
timeout_seconds = 120

[project.commands.test]
argv = ["python", "-m", "pytest"]
timeout_seconds = 300
```

Each approval preview displays the exact argv snapshot, workspace, and timeout.
Execution uses no shell, receives closed stdin, runs inside the selected workspace,
has bounded output and a timeout, and returns exit-status-based evidence. A later
Forge write makes earlier build/test evidence stale. Project configuration is a
trusted user-owned boundary: configured programs may themselves access the network,
and A10 does not provide an environment or network sandbox.

If verification fails, the approved mutation remains on disk, Forge reports the
failure, and the task stops without a corrective edit. A deterministic footer reports
the actual change/build/test status independently of model wording. `/clear` resets
conversation and coding-task state but never undoes filesystem changes.

When build or test is configured, Forge does not accept the model's first immediate
post-mutation final without a verification decision. It gives the model one bounded
opportunity to request an available verification tool or explicitly explain why it is
being skipped. Execution remains separately previewed and approval-gated; Forge never
auto-runs the command.

Forge still has no arbitrary shell execution or automatic fix/test/retry loop.
Mutation success alone is not a code-correctness claim, tests are never run without
approval, and each new user request receives a fresh one-mutation allowance.

Use explicit agent mode when a task needs a bounded sequence of repository reads,
one optional edit, and optional configured verification:

```bash
forge chat \
  --model qwen-small \
  --config ~/Models/forge/forge.toml \
  --workspace . \
  --agent
```

Agent mode is distinct from ordinary chat, read-only workspace chat, and `--assist`.
It runs in the foreground with limits of 16 model calls/iterations and 12 tool calls,
stops after repeated or no-progress activity, and prints a structured task footer.
Every write, build, and test still receives its own exact preview and approval prompt.
Only one successful mutation is permitted; a failed verification may be inspected and
explained but never authorizes a repair edit. Ctrl-C at an approval prompt cancels the
agent task without performing the proposed action. Agent mode adds no shell, Git
mutation, package installation, network access, rollback, or background execution.

Enable A13 repair authority explicitly with `--repair`:

```bash
forge chat \
  --model qwen-small \
  --config ~/Models/forge/forge.toml \
  --workspace . \
  --agent \
  --repair
```

Repair mode permits at most two approved mutations and two attempts per configured
build/test operation. The second mutation becomes available only after the first
change receives a current-generation `nonzero_exit` or `timeout`. Forge must reread
the current repair target, show another exact diff, and receive another approval.
Reverification is separately approved. A second verification failure ends the loop;
process-start and missing-command failures grant no repair authority. Approved
changes persist without automatic rollback.

A14’s preferred syntax selects orchestration and permissions independently:

```bash
forge chat \
  --model qwen-small \
  --config ~/Models/forge/forge.toml \
  --workspace . \
  --mode agent \
  --permissions confirm
```

For automatic execution of trusted configured build/test commands while keeping every
write approval-gated:

```bash
forge chat \
  --model qwen-small \
  --config ~/Models/forge/forge.toml \
  --workspace . \
  --mode repair \
  --permissions trusted-exec
```

Built-in profiles are deliberately small: `safe` allows reads and denies everything
else; `confirm` allows reads and asks for writes/build/tests; `trusted-exec` allows
reads and configured build/tests automatically but still asks for writes. No built-in
profile auto-allows writes. Legacy `--workspace`, `--assist`, `--agent`, and
`--agent --repair` forms remain supported and map to READ, ASSIST, AGENT, and REPAIR.

Repository modes also provide bounded Python symbol intelligence. Forge can inspect a
file outline, find exact simple or qualified definitions, locate structural reference
candidates, and read a targeted line range with the file's current SHA-256. These
operations use a local persistent SQLite index of derived file hashes, definitions,
and syntactic reference locations. The index is stored in the platform cache, never
in the repository, and refreshes synchronously before structural results are used.
It is not a semantic call graph, embedding search, or refactoring engine. Actual
source ranges or files must still be read before an implementation answer or
provenance-constrained write.

Repository workflows also plan active context deterministically. Forge estimates
observation cost, rejects source reads that cannot fit, recommends bounded ranges from
structural locations, and compacts superseded discovery or verification payloads.
Tools still return their real structured results, source reads remain explicit, and
context planning cannot grant permissions or authorize writes.

The derived cache can be inspected or maintained without loading a model:

```bash
forge index status --workspace .
forge index build --workspace .
forge index refresh --workspace .
forge index clear --workspace .
```

Run the controlled read-only coding evaluation locally with:

```bash
forge eval \
  --model qwen-small \
  --config ~/Models/forge/forge.toml \
  --suite coding-v1
```

Use `--suite context-v1` to run five focused structural-navigation tasks and compare
whole-file reads, targeted range reads, returned bytes/lines, grounding, and tool
counts.

An optional versioned JSON report is written only when requested:

```bash
forge eval \
  --model qwen-large \
  --config ~/Models/forge/forge.toml \
  --suite coding-v1 \
  --output eval-results/qwen-large.json
```

Evaluations use the committed fixture and the same production read-only
repository orchestration. Real-model runs remain local and are not part of CI.
See [`docs/EVALUATION.md`](docs/EVALUATION.md) for tasks, scoring, and report
details.

Forge also includes an opt-in `realworld-v1` baseline for an explicitly supplied
local C repository. It runs production repository modes only in disposable copies,
uses constrained evaluator approvals and independent oracles, and records model
failures without treating benchmark quality as a CI gate. See
[`docs/EVALUATION.md`](docs/EVALUATION.md#realworld-v1); the harness never downloads
or clones a benchmark repository.

An opt-in non-interactive repository smoke request is also available:

```bash
python scripts/smoke_repo_chat.py qwen-small \
  --config ~/Models/forge/forge.toml \
  --workspace . \
  --question "Where is the generic Model abstraction defined?"
```

The architectural direction and milestone boundaries are described in
[`docs/ROADMAP.md`](docs/ROADMAP.md). Current invariants are summarized in
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).
Semantic repository search is optional and separately configured. Copy
[`embeddings.example.toml`](embeddings.example.toml), select it with
`--embedding-config` and `--embedding-profile`, and explicitly build its local cache:

```text
forge index semantic-build --workspace . \
  --embedding-config /path/to/embeddings.toml --embedding-profile code
```

Forge never downloads embedding models. Semantic matches are discovery candidates;
the agent must read the recommended source range before using them as evidence.
# Hybrid repository retrieval

Configured semantic search now applies deterministic hybrid reranking: semantic
cosine remains dominant, while path/symbol/source tokens, code structure, and
source kind provide small relevance corrections. Generated package metadata is
excluded from repository indexes, and returned candidates include their source
kind while preserving raw cosine similarity. The model-free `retrieval-v1`
evaluation suite compares raw and reranked top-k quality on six Forge
architecture questions.

Forge narrows repository tools as useful candidates are discovered, helping
local models inspect known source before restarting broad searches. Failed or
exhausted candidates safely reopen discovery, including for multi-file tasks.

For multi-part repository questions, Forge can track which requested areas have
actual source evidence before accepting a final answer.

Decomposition is intentionally conservative: explicit lists, independent
semicolon tasks, and narrowly phrased relationship questions are supported;
ambiguous prose remains one evidence goal.

Forge can build a fast language-agnostic repository map for immediate source
discovery, while using semantic retrieval when a compatible semantic index is
already available. Each unresolved evidence goal receives at most one bounded
bootstrap before the local model chooses source to inspect; lexical and semantic
results remain discovery-only until Forge reads the source.

Once the required source evidence is gathered, Forge locks retrieval and asks the
local model to answer from that evidence instead of continuing to search.

For coding tasks, once Forge has current source evidence for a safe write target,
it narrows the session toward proposing the code change instead of continuing broad
repository discovery. The model still authors the patch and every write remains
subject to normal preview, policy, and approval enforcement.
