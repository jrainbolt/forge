# A95 repository breadth and terminal-state dogfood

Use a disposable or read-only repository. Do not approve mutations during this
check. Start Forge from an installed environment without relying on
`PYTHONPATH`.

## Launch and startup

1. Launch Forge against the read-only `aspect-survivor` workspace with
   `qwen-large`, PLAN mode, and TRACE display.
2. Confirm the startup loading indicator occupies one physical line, changes
   frames, and its elapsed time advances throughout model initialization.
3. Confirm the loading line is removed cleanly when initialization finishes or
   fails, with no orphan renderer or damaged prompt.

## Repository breadth

Submit these prompts in order:

1. `Tell me about this project.`
   Confirm TRACE classifies it as BROAD and reports real tool-budget, area, and
   source counts. The answer must use multiple repository areas or explicitly
   describe the survey as partial.
2. `Tell me about the swordsman class.`
   Confirm it remains targeted, is grounded in relevant source, and does not
   fail with a raw tool-limit message.
3. `I'm thinking about extending this game to multiplayer. Analyze the entire
   project and create a plan.`
   Confirm it is BROAD, samples representative repository-derived areas, cites
   materially relevant sources, and does not claim to have read every file.
4. `Trace how melee combat works from player input through damage being applied.`
   Confirm it follows multiple source boundaries when supported and names any
   link that could not be established.

For any exhausted evidence budget, confirm the user sees a qualified partial
answer or a clear message that Forge could not gather enough repository evidence
within the safe tool limit—not only `tool execution limit exceeded`.

## Active terminal state

1. During each request, confirm the bottom status line remains visible, reports
   `busy`, and coexists with the transient activity line without presenting a
   fake editable prompt.
2. Resize the terminal while idle and again while a request is busy. Check wide,
   medium, and narrow widths. Confirm fields degrade gracefully, no toolbar is
   duplicated, and the prompt returns correctly.
3. Confirm no theme paints a background across the unused remainder of the
   terminal.

## Completion and themes

1. Type `/theme ` and press Tab. Confirm `default`, `ember`, and `mono` are
   offered; `/theme e` should narrow to `ember`.
2. Repeat for `/display ` (`minimal`, `normal`, `trace`) and `/mode ` (`chat`,
   `plan`). Confirm arrow navigation, Enter/Tab selection, and Esc dismissal.
3. Compare `/theme default` and `/theme ember`. Ember should have visibly warmer
   orange/amber brand, prompt, mode, toolbar, and tool accents. Confirm `/theme
   mono` is unstyled.
4. Open a second terminal session and confirm its theme, busy state, completion,
   and resize behavior are independent.

## Exit and non-TTY behavior

1. Exit normally and confirm the terminal cursor and input state are restored.
2. Pipe a non-interactive invocation where supported and confirm no interactive
   spinner or footer escape sequences are emitted.

Record the prompt, reported scope, tool budget, sampled areas, source count,
citations, and any model/backend initialization error for each request.
