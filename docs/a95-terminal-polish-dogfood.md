# A95 terminal-polish dogfood guide

Run these checks in a normal interactive terminal. Do not set `PYTHONPATH`; use
the installed `forge` command and the model configuration you ordinarily use.

1. **A — Project detection:** enter a known project and run bare `forge`. Confirm
   that Forge detects the project before model selection.
2. **B — Model selection:** select `qwen-large`. Confirm the selected profile is
   the one shown later in the splash and status line.
3. **C — Loading:** observe the compact animated flame while the model loads.
   Confirm it redraws one line and disappears completely when loading finishes.
4. **D — Splash:** confirm the compact FORGE mark, “Local AI Agent,” and only
   known project/model/mode/display/access/context metadata appear once.
5. **E — Command list:** enter `/`. Confirm all registered commands appear in
   alphabetical order.
6. **F — Filtering/completion:** enter `/tr` and confirm only `/trace` is shown.
   At a fresh prompt, type `/tr` and press Tab; confirm it completes to `/trace`.
7. **G — Display levels:** run `/minimal`, `/normal`, and `/trace` in that order.
   Confirm the prompt-adjacent status line immediately reflects each level.
8. **H — Repository conversation:** ask several repository questions. Confirm
   the activity spinner clears before answers and the status line remains intact.
9. **I — Resize:** resize narrower and wider between prompts. Confirm the status
   text truncates instead of wrapping over the input or conversation.
10. **J — Themes:** run `/theme mono`, `/theme default`, and `/theme ember`.
    Confirm wording and status remain understandable in all three themes.
11. **K — Copy/paste:** copy assistant output from terminal scrollback. Confirm
    prompt/status content is not interleaved with the assistant response.
12. **L — Exit:** run `/exit` and confirm Forge exits without a lingering flame,
    spinner, hidden terminal input, or damaged shell prompt.

Optional compatibility checks:

- Run with `NO_COLOR=1` and confirm no ANSI color is emitted.
- Run with `FORGE_NO_ANIMATION=1` and confirm a static loading message appears.
- Pipe or redirect output and confirm there is no splash, animation, status bar,
  or control-sequence noise.
