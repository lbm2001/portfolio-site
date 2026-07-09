# Handoff: Mobile version of the VLA hero demo

You are implementing the mobile ("Mini VLA Demo") version of the landing-page hero in this
repo (`portfolio-site`, branch `feat-mobile-view`). Read this whole document before writing
code. Everything you need is in here plus the two files you will touch. Line numbers below
are approximate — grep for the quoted selectors/identifiers rather than trusting offsets.

## 1. What exists today

The desktop hero (`components/Hero.tsx` + the `.vla-*` / `.hero*` sections of
`app/globals.css`) is a live Vision-Language-Action demo: five cards
(Demonstration, Vision Encoder, Language Encoder, Action Head, Rollout) absolutely
positioned in a ring around the centered name, plus a training control bar at the bottom.
"Start Training" runs a genuine TF.js behavioral-cloning loop in a Web Worker; once the
loss converges the Rollout becomes interactive (type a command, run it, drag/reshuffle
blocks).

Key architecture facts:

- **All model/task/rollout logic lives in the external `mini-vla` package**
  (`github:lbm2001/mini-vla#v0.1.0`, consumed via `transpilePackages` in
  `next.config.mjs`). Hero imports `VLATrainer`, `RolloutEngine`, `paintScene`,
  `CONFIG`, etc. from `mini-vla/*`. **You must not modify the mini-vla package**, and you
  must not need to — this task is 100% `components/Hero.tsx` + `app/globals.css`.
- **There are no drawn wires anymore.** The pipeline connection is shown by five small
  "payload" tokens (`.vla-payload`, spans with `data-flow="p1"…"p5"`) that glide along
  **CSS `offset-path` cubic Béziers**. The paths are rebuilt every frame by
  `layoutWires()` in Hero.tsx from the cards' real `getBoundingClientRect()` boxes, in
  **px relative to the stage** (the `<header class="hero">`). The `arc()` helper there
  emits horizontally-biased control points (token leaves/arrives along the horizontal).
  Hops: p1 Demonstration→Vision, p2 Prompt→Language, p3 Vision→Action Head,
  p4 Language→Action Head, p5 Action Head→Rollout.
- The hero `<header>` carries state classes: `is-idle`, or `is-live` plus one of
  `is-loading` / `is-paused` / `is-converged` (see `stateClass` in Hero.tsx).
- Below 1100px today, a media query (`@media (max-width: 1099px)` near the end of the
  vla CSS) simply `display: none`s `.vla-node, .vla-flow, .vla-bar` and the hero becomes
  a plain centered intro (`.hero-content`: name, tag line, two CTA buttons).
- Pause/resume already has correct wall-clock bookkeeping (`pauseStartRef` /
  `pausedAccumRef` + `trainer.pause()/resume()` in `onPrimary`). Reuse it — do not
  invent a second mechanism.
- Touch support already exists where it matters: block dragging uses pointer events with
  `touchAction: "none"`, and the Vision panel has a tap toggle (`modelView` state).

## 2. What to build

On viewports **below 1100px** (same breakpoint the CSS already uses):

1. The hero stays the clean centered intro, but gains a **third CTA button** in
   `.hero-cta`, label: **"Mini VLA Demo"** — visible only below 1100px (hide it with CSS
   on desktop; always render it so SSR/hydration stay consistent).
2. Tapping it swaps the intro out and shows the **full pipeline, stacked vertically, on
   the main page** (no navigation, no route change). Component state `showDemo`
   (boolean, default `false`) → add a `demo-open` class on the hero header.
3. A **close button (✕)** restores the intro. Closing while training must **pause**
   training (reuse the existing pause bookkeeping) so it doesn't burn battery invisibly.
4. Desktop (≥1100px) must be **completely unchanged** — same DOM is fine, but zero
   visual/behavioral difference.

### Stacked layout (demo open, <1100px)

```
[✕]
┌──────────────────────────────┐
│ Demonstration                │   ← .vla-input, full width; the command prompt
│ (canvas)                     │     (.vla-prompt) folded INSIDE/below the card,
│ "pick up the red block"      │     in-flow (see gotcha G2)
└──────────────────────────────┘
┌─────────────┐  ┌─────────────┐
│ Vision      │  │ Language    │   ← .vla-vision / .vla-lang side by side
│ Encoder     │  │ Encoder     │
└─────────────┘  └─────────────┘
┌──────────────────────────────┐
│ Action Head (shoulder/elbow/ │   ← .vla-action, full width (or centered, narrower)
│ gripper readout)             │
└──────────────────────────────┘
┌──────────────────────────────┐
│ Rollout                      │   ← .vla-output, full width; .vla-try row appears
│ (canvas)                     │     in-flow above the canvas when converged
└──────────────────────────────┘
[ status · loss curve · ⚙ · ▶ ]   ← .vla-bar, sticky at the viewport bottom
```

Recommended mechanism (keeps JSX order untouched): under
`@media (max-width: 1099px)` + `.demo-open`, make the hero a two-column CSS grid;
`.vla-input`, `.vla-action`, `.vla-output`, `.vla-bar` span `grid-column: 1 / -1`,
Vision and Language take one column each. All `.vla-node`s switch to
`position: static`, `transform: none`, `width: auto`. The hero drops its
`height: calc(100vh - 61px)` and just flows/scrolls. `.hero-content` is hidden while
`demo-open`. `.vla-flow` stays `position: absolute; inset: 0` overlaying the (now
taller) hero — absolutely positioned grid children are taken out of the grid flow, so
this works without JSX changes.

The payload tokens must keep gliding: in `layoutWires()`, detect the stacked mode (cache
a `matchMedia("(max-width: 1099px)")` — pair it with the `demo-open` state) and emit
**vertically-biased** Béziers between vertically-facing edges instead. Suggested hops
(tune anchors by eye):

- p1: Demonstration bottom edge (at Vision's cx) → Vision top edge
- p2: Prompt bottom edge → Language top edge
- p3: Vision bottom edge → Action Head top/left region
- p4: Language bottom edge → Action Head top/right region
- p5: Action Head bottom edge → Rollout top edge

A vertical `arc()` mirrors the existing one: control points at
`(sx, sy + dy*0.42)` and `(ex, sy + dy*0.58)`. Everything else (glide keyframes,
stagger classes `vla-flow-a/b/c`, paused/converged behavior) works as-is.

### Control bar

Make `.vla-bar` in-flow at the end of the stack but `position: sticky; bottom: 0` (plus
a little padding/background so it reads as a bar) so Start/Pause/Reset and the loss curve
stay reachable while the user scrolls the pipeline. The ⚙ popover (`.vla-cfg-pop`,
232px wide, anchored `bottom: calc(100% + 10px); right: -8px` to the gear) opens upward
— verify it stays fully on-screen at 320px viewport width; adjust its `right` offset on
mobile if it clips.

### Battery / performance guards (also benefit desktop)

1. **Pause on hidden**: on `visibilitychange` → hidden, if `status === "training"`,
   pause (existing bookkeeping) and set an `autoPausedRef` flag; on visible, resume only
   if auto-paused (never override a user's manual pause).
2. **Pause off-screen**: an `IntersectionObserver` on the hero — same auto-pause/resume
   flag when it scrolls fully out of view. On mobile the user *will* scroll past it
   mid-training.
3. **Cap DPR at 2** in `fitCanvas` (`Math.min(window.devicePixelRatio || 1, 2)`).
   iPhones report DPR 3 — 2.25× the pixels of DPR 2 with no visible gain in these
   small panels.

### Touch/iOS specifics

- `.vla-try-input` is `font: … 12px …` — on mobile it must be **≥16px** or iOS Safari
  zooms the page on focus. Override under the mobile media query.
- Closing the demo (and the demo being closed) should not leave the rAF loop painting
  invisible canvases at fallback sizes forever — cheapest fix: skip the draw calls when
  the demo is closed on mobile (the loop can keep running for desktop).
- Optional stretch (only if everything else is done and verified): 3–4 tappable preset
  command chips above the try-input in converged mode, mobile-only — typing is the
  highest-friction step on a phone. Tapping a chip fills the input and runs it.

## 3. Gotchas — read before coding

- **G1 — drift keyframes bake in centering transforms.** `.is-idle` / `.is-converged`
  nodes animate `vla-drift-y` / `vla-drift-x`, whose keyframes hard-code each desktop
  anchor's centering transform (`translateY(-50%) translate(...)`). In the stacked
  layout these MUST be disabled (`animation: none`) and transforms reset, or every card
  is displaced by half its size.
- **G2 — `.vla-prompt` is absolutely positioned** (`top: 100%` below the Demonstration
  card, out of flow). Stacked, it would overlap the encoder row. Make it in-flow
  (`position: static; margin-top: 10px`) inside the card on mobile. `layoutWires`
  measures it wherever it is, so p2 keeps working. Keep its idle/converged
  `opacity: 0` behavior — it must stay in layout (never `display: none`) so it can be
  measured.
- **G3 — `.vla-try` hangs above the Rollout with `left: -44px`** (reaches over the
  Action Head on desktop). On mobile give it a normal in-flow/full-width override
  (`position: static`, `left: auto`, add margin) at the top of the Rollout card.
- **G4 — label hop animation.** Vision/Language/Action labels are absolutely positioned
  (centered when idle, top-left when live, animated via `vla-label-hop`). This is
  self-contained inside each card and should survive the stacked layout untouched —
  just don't change card `padding-top: 32px`.
- **G5 — chip row fit.** `.vla-chip-row` scales itself to its container via a
  JS-measured `--chip-fit` (see `useLayoutEffect` in Hero.tsx). It re-measures on
  window resize and on sentence change — after toggling `showDemo` the container width
  changes without a resize event, so trigger the fit once on toggle (or add `showDemo`
  to that effect's deps).
- **G6 — `layoutWires` early-returns when refs are missing**, and all measured elements
  are always rendered. Don't conditionally unmount any of the five cards or the prompt;
  hide with CSS only.
- **G7 — reduced motion.** Any animation you add (and the mobile drift disabling) must
  respect the existing `@media (prefers-reduced-motion: reduce)` kill-list — extend it
  if you add new animated selectors.
- **G8 — do not touch training/config knobs.** No changes to `mini-vla`, no mobile
  `CONFIG`/`RunConfig` overrides, no batch-size/img-size tweaks — on-device training
  speed is being evaluated separately by the owner.
- **G9 — status semantics.** "Closed" is not a trainer state. Close = UI hidden +
  (if training) paused via the normal path, so the status text/bar stay consistent when
  reopened. Reopening does NOT auto-resume; the user resumes via the bar (exception:
  the visibility/scroll auto-pause DOES auto-resume, because the user never asked for
  that pause).

## 4. Out of scope

- Any change to the `mini-vla` package, training parameters, or convergence gates.
- Auto-starting training when the demo opens (user taps Start Training).
- A third layout tier for portrait iPads — they get the same stacked layout
  (landscape iPads are ≥1100px and already get the desktop ring).
- Rewording the ⚙ ETA copy ("on a laptop GPU") — leave as is.

## 5. Acceptance criteria

1. **Desktop ≥1100px: pixel-identical and behaviorally unchanged** in all states
   (idle → loading → training → paused → converged → reset).
2. Mobile <1100px, demo closed: intro identical to today plus the third CTA button.
3. Tap "Mini VLA Demo" → intro disappears, stacked pipeline appears (order per the
   sketch), page scrolls naturally; ✕ restores the intro and pauses training if it was
   running.
4. Full training run works on mobile layout: payload tokens glide vertically along the
   stacked hops, loss curve draws, convergence unlocks the try-input; typed command runs;
   block drag works by touch; ⟳ reshuffles.
5. Backgrounding the tab or scrolling the hero out of view during training pauses it;
   returning resumes it; a user-initiated pause is never auto-resumed.
6. Focusing the try-input on iOS does not zoom the page.
7. `npx tsc --noEmit` clean; `npm run dev` boots and the trainer worker starts
   (check the browser console for worker errors).

## 6. Verification

- `npx tsc --noEmit`.
- `npm run dev`, then exercise BOTH layouts in a real browser (or Playwright with a
  mobile viewport, e.g. 390×844 iPhone and 1440×900 desktop): run a full training to
  convergence in each, screenshot idle/training/converged.
- Verify the desktop-unchanged criterion by diffing desktop screenshots against `main`
  if possible, or by careful visual comparison of all five cards + bar + payloads.
- Test the close-while-training path and the visibilitychange auto-pause (switch tabs).
- If you have LAN access from a phone, `npm run dev -- -H 0.0.0.0` and test on-device;
  otherwise note in your summary that on-device testing is pending.
