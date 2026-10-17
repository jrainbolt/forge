# A95 interactive-terminal dogfood guide

Use an installed `forge` command in a normal interactive terminal. Do not set
`PYTHONPATH`.

1. **A:** Enter the `aspect-survivor` project and launch bare `forge`.
2. **B:** Select `qwen-large`.
3. **C:** Watch the multi-line Forge flame throughout model loading. Confirm its
   contour visibly flickers without scrolling.
4. **D:** Confirm the final side-by-side flame and metadata fit in a normal
   terminal without scrolling or duplicated project/model output.
5. **E:** Type `/`; confirm the completion popup lists commands alphabetically.
6. **F:** Use Up/Down to change selection and Tab or Enter to select naturally.
7. **G:** Type `/tr`; confirm the popup narrows to `/trace`.
8. **H:** Press Esc; confirm the popup closes without submitting input.
9. **I:** Run `/trace`; confirm TRACE appears in the bottom toolbar.
10. **J:** Ask `How does MeleeAttackSystem work?`
11. **K:** Confirm assistant prose uses the normal foreground while tool verbs,
    paths, metrics, and toolbar fields use restrained semantic colors.
12. **L:** Confirm the toolbar remains attached to the prompt instead of entering
    conversation scrollback.
13. **M:** Resize through wide, medium, and narrow widths. Confirm splash,
    completion menu, prompt, and toolbar adapt without corruption.
14. **N:** Run `/theme mono`, then `/theme ember`; confirm information remains
    readable and only presentation changes.
15. **O:** Copy an assistant answer; confirm completion and toolbar content do not
    contaminate the copied response.
16. **P:** Press Ctrl-C at an idle prompt and confirm Forge exits cleanly.

Compatibility checks:

- Set `NO_COLOR=1`; the flame may still animate, but no color should appear.
- Set `FORGE_NO_ANIMATION=1`; a static loading message should replace animation.
- Redirect output; no splash, animation, completion UI, toolbar, ANSI, or cursor
  controls should be emitted.
