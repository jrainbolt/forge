# A95 terminal UX dogfood guide

Forge keeps session history in memory only. Run these scenarios from a disposable
or safely backed-up project when testing MANUAL edits. Forge does not invoke Git.

1. From a repository containing a common project marker, run `forge`. Confirm the
   detected workspace and `[PLAN]` zero-write badge.
2. Chat naturally: `I'm considering changing the retry mechanism.` Confirm that
   conversation alone does not request a mutation.
3. Ask `Inspect the current retry mechanism and explain it.` Confirm concise search
   and read events plus trusted source paths.
4. Run `/mode plan`, then ask for an implementation plan. Confirm no write preview
   or approval is offered.
5. Run `/trace on`, ask another repository question, and inspect the bounded tool,
   grounding, timing, model, and context details. Hidden prompts and reasoning must
   not appear.
6. Run `/trace off` (or `/display normal`) and confirm concise output returns.
7. Exit and start `forge chat --mode manual --workspace .`. Request one small edit,
   inspect its preview, and explicitly approve or reject it. Confirm `[ASK WRITE]`.
8. Observe the per-turn elapsed time, tool/file counts, output tokens when supplied
   by the backend, and the context gauge. Unknown values must remain unknown.
9. Run `/multi`, enter several lines (including indentation), then enter a line
   containing only `.`. Confirm the complete paragraph is treated as one request.
10. During a long generation press Ctrl-C. Forge cancels the active turn and does
    not commit a partial conversation turn. Any approved mutation that completed
    before cancellation remains on disk and is reported. Escape-key cancellation
    is deferred because the current model API is blocking and has no portable,
    transaction-aware cancellation primitive.
11. At the prompt press Ctrl-C to exit with status 130, or use `/exit` for a normal
    exit.

Runtime `/mode chat` and `/mode plan` switching is available only inside an already
read-only project session. Any change that would add authority requires a restart
with an explicit CLI mode. DRY_RUN remains deferred until preview generation can be
isolated from the production mutation executor.
