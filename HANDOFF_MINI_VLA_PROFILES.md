# Handoff: fixed task profiles in `mini-vla` (replacing the ⚙ run config)

You are changing the `mini-vla` package so the demo trains **two fixed task profiles**
instead of a user-selectable one. The consuming host (`portfolio-site`, branch
`feat-mobile-view`) has **already been migrated** — its ⚙ menu is gone and it now resolves
the profile from the viewport. This document is the contract it needs you to honor.

Read all of it before writing code. The one thing that must survive unchanged is in §2.

---

## 0. State of the working copy (read first)

As of 2026-07-09 ~16:00, `node_modules/mini-vla` in the host repo was **deleted**, and
shortly before that `src/trainer.core.ts` was edited to add a `[TIMING]` debug log
referencing an out-of-scope `loss` — it does not compile:

```ts
// src/trainer.core.ts, in the warmup loop
if (k % 10 === 0)
  console.log(`[TIMING]   warmup k=${k} loss=${loss.toFixed(4)} …`);
//                                          ^^^^ not in scope
```

If that is your work in progress: it is on the trainer's warmup path, so a live run stalls
rather than fails loudly. One of the host's verification runs timed out in a way consistent
with this, though I could not confirm it. Please make sure it is gone (or fixed) before
tagging a release the host will install.

---

## 1. What the host now does

- **Two profiles, chosen by viewport, not by the viewer.** Below the `max-width: 1099px`
  breakpoint (the same one that stacks the pipeline on phones) the hero trains the smaller
  task, because a phone pays for every gradient step in battery and heat.

  | profile | `numColors` | `maxBlocks` | ETA shown |
  |---------|-------------|-------------|-----------|
  | desktop (≥1100px) | 8 | 4 | ~51s |
  | mobile (<1100px)  | 4 | 3 | ~42s |

- **The profile is latched at Start and released at Reset.** `setRunConfig` installs
  *per-thread* module state; if the profile tracked the media query live, a viewer
  resizing across the breakpoint mid-run would leave the main thread's `randomLayout()`
  sampling scenes the worker is not training on.

- The gear's information did not disappear with the gear. The control bar's idle status
  column now reads `8 colors · ≤4 blocks` / `est. ~51s on a laptop GPU`, from
  `estimateTrainingSeconds(cfg)`.

- Because the mobile profile only ever *learns* 4 of the 8 colors, the host added:
  - **Preset command chips** (mobile only), one per trained color, derived from the
    profile — currently `COLORS.slice(0, cfg.numColors)`.
  - **Two guards** on the free-text "try it" box, surfaced in `.vla-try-note`:
    - a command naming a color this run never trained → *"this run never learned purple —
      only red, black, blue, yellow"*
    - a command naming a trained color absent from the current scene → *"no red block in
      this scene — ⟳ to reshuffle"*

  Both guards exist because the color head stays 8-wide (§5). Without them the model
  quietly answers an untrained color word with the nearest color it *does* know, and picks
  up the wrong block, with no indication anything was out of distribution.

---

## 2. The hard constraint — do not "simplify" this away

**`mini-vla` cannot detect desktop vs. mobile itself.** The trainer runs in a Web Worker
(`trainer.worker.ts`), and a Worker has no `window` and no `matchMedia`. There is no
channel into the worker other than the `{t:"start"}` message.

So the following must keep working exactly as they do today:

```ts
setRunConfig(cfg);                  // installs on THIS thread's samplers
trainer.start(onUpdate, cfg);       // ships cfg to the worker in {t:"start"}
```

The host resolves the profile and passes it in. Do **not** replace this with
package-internal auto-detection, a build-time constant, or an env flag. Keep `cfg` as a
parameter of `start()`, and keep `setRunConfig` exported.

---

## 3. Deliverables

1. **Export the two profiles.** Keep `RunConfig` as a type.

   ```ts
   export const DESKTOP_RUN_CONFIG: RunConfig = { numColors: 8, maxBlocks: 4 };
   export const MOBILE_RUN_CONFIG:  RunConfig = { numColors: 4, maxBlocks: 3 };
   ```

   The host currently defines these locally. The numbers belong with the model. Once you
   export them, `RunConfig`'s unions can narrow (`numColors: 4 | 8`, `maxBlocks: 3 | 4`)
   and `DEFAULT_RUN_CONFIG` can become an alias of the desktop profile or disappear —
   `trainer.start()` currently defaults `cfg` to it.

2. **Export the active palette.** The host builds its preset chips with
   `COLORS.slice(0, cfg.numColors)`, which duplicates *your* sampling rule (the "first N
   entries" logic in `pickColors` / `randomLayout`). If the palette is ever reordered, the
   chips silently misrepresent what was trained. Give the host:

   ```ts
   export function activePalette(cfg: RunConfig): ColorDef[];
   ```

   and use the same function internally in `randomLayout`, so there is one rule, not two.

3. **Keep `estimateTrainingSeconds(cfg)` exported.** The idle bar prints it. Today
   `CONFIG.eta` yields 51s (desktop) and 42s (mobile). If `RunConfig` goes away entirely,
   ship an equivalent — the host has nowhere else to get this number.

4. **Update the stale prose.** `run-config.ts`'s header still describes "the ⚙
   palette/density/task-set a host picks before training." No host has a gear anymore.

---

## 4. Decisions we need from you

**Which four colors does mobile get?** Today "first four" means red, black, blue, yellow
(indices 0-3 of `grammar.json`). Two problems:

- Yellow `#d9a800` against the `#e6e6e6` scene floor is the weakest contrast pair in the
  palette, and at four colors it appears roughly twice as often — on the smallest screens.
- The host **cannot** override the selection. "First N" is your rule. If you want a
  different four, the profile has to carry explicit color indices rather than a count:
  `{ colors: [0, 1, 2, 4], maxBlocks: 3 }`.

We have no strong preference, but the current answer is an accident of palette order rather
than a choice, and it deserves to be one.

---

## 5. Do not break these without telling us

- **The color head stays 8-wide.** `run-config.ts` says so deliberately: `numColors` only
  restricts what the samplers draw, so every profile trains the same architecture. The
  host's "never learned purple" guard is built on this — untrained classes remain in the
  output space and get suppressed, so a stray color word yields a confident *wrong* answer
  rather than an error. **If you narrow the head to `numColors`,** `decodeCommand`'s
  returned index space changes and every `COLORS[d.color]` lookup in `Hero.tsx` must be
  re-based. Say so loudly in the release notes.

- **`DEFAULT_LAYOUT` and `DEFAULT_SENTENCE` must stay valid under *both* profiles.** They
  are the SSR-safe idle scene, rendered before any profile is installed: black + red, and
  `"pick up the red block"`. Both colors sit inside the mobile four *today*. Reorder the
  palette and this breaks silently, on first paint, on the server.

- **Color synonyms must stay single words, and no dropped color may share a synonym with a
  kept one.** The host's untrained-color guard does word-level matching against
  `ColorDef.synonyms`. Verified today: no multi-word synonyms, no clashes — `emerald`,
  `violet`, `magenta` all resolve to their untrained colors.

- **The convergence gate** (`converge.loss: 0.015`, `window: 10`, `streak: 8`,
  `minBatches: 100`, `maxBatches: 450`). The bar's ETA copy and the host's Playwright
  timeouts are calibrated against these.

- **`randomLayout()`'s block count** stays `2..min(maxBlocks, numColors)`, so the mobile
  profile yields 2-3 blocks. The host's rollout canvas and drag hit-testing assume this.

---

## 6. Acceptance

1. `npm run typecheck` clean in `mini-vla`; `npx tsc --noEmit` clean in the host against
   the new version (the host transpiles your `src/` directly via `transpilePackages`).
2. A host run at ≥1100px trains 8 colors / ≤4 blocks; a run at <1100px trains 4 / ≤3.
   Confirm by watching the demonstration scenes: mobile must never show green, orange,
   purple or pink.
3. `activePalette(MOBILE_RUN_CONFIG)` returns exactly the colors mobile scenes contain.
4. Resizing across 1100px **mid-run** does not change the scenes being sampled.
5. The trainer worker still receives the profile in `{t:"start"}` and installs it before
   building the model.
6. No `[TIMING]` / `TEMP` debug logging ships.

## 7. Out of scope

- The host's layout, CSS, preset chips, or note copy — those are `portfolio-site`'s.
- The 1100px breakpoint itself. It is the *layout* breakpoint; the host deliberately reuses
  it for the profile so that a phone and a narrow desktop window, which show the identical
  stacked pipeline, also train the identical task.
