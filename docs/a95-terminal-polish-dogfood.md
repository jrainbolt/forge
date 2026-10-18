# A95 terminal UX reliability dogfood guide

Use an installed `forge` command in a normal interactive terminal without
setting `PYTHONPATH`.

1. **A:** Enter the `aspect-survivor` project and launch bare `forge`.
2. **B:** Select `qwen-large`.
3. **C:** Confirm only a single-line spinner with elapsed time animates during
   model loading.
4. **D:** Confirm there is no terminal flicker, vertical cursor jump, duplicated
   line, or prompt corruption.
5. **E:** Confirm the static, colored Forge flame appears after loading.
6. **F:** Confirm the side-by-side startup fits without scrolling in a normal
   terminal and stacks cleanly at narrow widths.
7. **G:** Confirm startup guidance is only `/help commands · Ctrl-C exit`.
8. **H:** Confirm the bottom toolbar uses the terminal background rather than a
   gray filled rectangle.
9. **I:** Ask a repository question and confirm assistant prose uses the normal
   foreground.
10. **J:** Confirm paths, tools, errors, status fields, and metrics remain visually
    distinct through restrained foreground colors.
11. **K:** Open a second Forge terminal simultaneously and confirm both prompts,
    toolbars, and themes remain independent and stable.
12. **L:** Type `/`, navigate with Up/Down, filter with `/tr`, complete with Tab,
    and dismiss with Esc.
13. **M:** Resize through wide, medium, and narrow terminal widths.
14. **N:** Run `/theme mono`, then `/theme ember`; confirm only presentation
    changes and neither theme introduces a toolbar background.
15. **O:** Exit cleanly with Ctrl-C and confirm the shell prompt is intact.

Compatibility checks:

- Set `NO_COLOR=1`; spinner shape remains usable without color.
- Set `FORGE_NO_ANIMATION=1`; a static loading message replaces the spinner.
- Redirect input/output; no splash, spinner, completion UI, toolbar, ANSI, or
  cursor-control output should appear.
