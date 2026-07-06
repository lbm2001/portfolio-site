# Handoff: Portfolio Landing Hero — Live VLA (Vision-Language-Action) Demo

## Overview
This is a redesign of the landing page hero for lukasmueller.dev. The hero (name, tagline, nav, buttons) is unchanged from the live site. The six separate mini ML demos that used to float around it have been replaced with **one unified, interactive Vision-Language-Action (VLA) model diagram**: a robot arm learns to "pick up the red block" from a text prompt + a demonstration image, live in the browser. Clicking "Start Training" runs a fake-but-visually-coherent training loop that animates every stage of the pipeline together.

## About the Design Files
The file in this bundle (`Landing.dc.html`) is a **design reference built in HTML** — a working prototype demonstrating exact layout, motion, and interaction, not production code to copy verbatim. It runs today as a single self-contained HTML file with inline styles and vanilla JS/Canvas (no framework). Your task is to **recreate this design in the target codebase's actual stack** (whatever lukasmueller.dev is built with — appears to be a hand-rolled CSS design system, see tokens below) using its existing components/patterns. If there's no established frontend framework, pick whatever is simplest for a mostly-static personal site with light interactivity.

Note: `Landing.dc.html` includes a `<script src="./support.js">` and custom `<x-dc>` wrapper tags — these are internal to the design tool that produced this file and are **not relevant to your implementation**. Ignore that scaffolding; everything you need is in the inline styles and the `<script>` class body (which is plain, readable JS driving `<canvas>` drawing + React-like state — treat it as pseudocode for the intended behavior, not a library to import).

## Fidelity
**High-fidelity.** Colors, type, spacing, and radii match the site's real design tokens (extracted from the live site's `globals.css`, see below). Recreate pixel-accurately using the codebase's existing CSS/tokens.

## Screens / Views
Single view: the landing page hero section. No routing/navigation changes.

### Layout
- Full-viewport hero (`min-height: 100vh`), white background (`#ffffff`).
- Sticky-feel nav bar at top: `padding: 18px 56px`, bottom hairline border `1px solid rgba(0,0,0,.1)`, background `rgba(225,45,26,.04)` with `backdrop-filter: blur(10px)`. Left: wordmark "Lukas Müller" (700 weight, 16px). Right: flex row of nav links (About / Resume / Projects / Blog), gap 32px, `600 14px` sans.
- Below nav, a relatively-positioned **stage** (`height: calc(100vh - 61px)`, `min-height: 700px`) contains:
  1. The **hero content**, dead-centered via `position:absolute; left:50%; top:50%; transform:translate(-50%,-50%)`, `z-index:10`, `text-align:center`, `pointer-events:none` (except its buttons).
  2. Four **VLA boxes** positioned around the hero at fixed anchor points (see Components), each `position:absolute`, `z-index:3`, background `#fcfcfc`, `1px solid rgba(0,0,0,.1)` border, `border-radius:3px`.
  3. An **SVG wire layer** (`position:absolute;inset:0;z-index:1;pointer-events:none`) drawing orthogonal (90°) connector lines between the boxes, routed around the hero.
  4. A **training control bar** centered at the bottom (`left:50%,bottom:3.5%,transform:translateX(-50%)`), `z-index:8`.

### Components

**Hero**
- H1 "Lukas Müller": `font-weight:700; font-size:46px; line-height:1.02; letter-spacing:-.03em; color:#111`.
- Tagline "MACHINE LEARNING & ROBOT LEARNING": `600 11px` monospace, `letter-spacing:.18em`, uppercase, color `#e12d1a` (brand red), `margin-top:16px`.
- Buttons row, `margin-top:28px`, `gap:12px`:
  - Primary "Contact": bg `#111`, text white, `1px solid #111`, `border-radius:3px`, `padding:14px 24px`, `600 14px` sans.
  - Secondary "Download Resume": transparent bg, text `#111`, `1px solid rgba(0,0,0,.22)`, same radius/padding/font.
- Hero opacity: `1` normally, fades to `0.25` while training is running (`transition: opacity .55s ease`) so focus shifts to the corner cluster of boxes.

**The four VLA boxes** — each is `212px` wide (Vision/Language auto-width to content) with `14px` padding. Anchor positions (idle, i.e. always — boxes do **not** move in the current build; see Interactions for the one thing that does move):
  - **Demonstration** (was "Input"): `left:2.5%; top:50%; transform:translateY(-50%)`.
  - **Vision Encoder**: `left:50%; top:4%; transform:translateX(-50%)`.
  - **Language Encoder**: `left:50%; top:73%; transform:translateX(-50%)`.
  - **Rollout** (was "Output"): `right:2.5%; top:50%; transform:translateY(-50%)`.
- Each box's **label** (e.g. "Demonstration") is a small `600 10px` monospace, `letter-spacing:.16em`, uppercase, color `#9a9a9a` string. **Before training starts**, each label (except Demonstration's) is absolutely centered within its box (`position:absolute;top:50%;left:50%;transform:translate(-50%,-50%)`); **once training starts**, it snaps to the normal top-left flow position (`position:static`) above that box's content. The "Demonstration" label is hidden entirely (`opacity:0`) until training starts (its box shows a live looping animation instead — see below — so the label isn't needed at rest).
- Every box reserves its **full expanded height at all times** (canvas elements keep `display:block` and their layout size always) — only `opacity` toggles content, so nothing reflows when training starts/stops.

**Demonstration box** — a `<canvas>`, 100% width × 186px, showing a small 2D scene: a floor line, a black square block (left), a red square block (right), and a simple 2-link robot arm (upright at rest) centered between them. This canvas **loops a scripted "pick up the red block" animation continuously, always at full opacity**, independent of the training toggle: arm reaches down to the red block (~1.5s), "grasps" it (drawn as an accent-red square following the gripper instead of sitting on the floor), lifts it, then returns upright — repeat every 4.2s, eased with a raised-cosine curve. A full 16×16 grid overlay (the same resolution the vision encoder tokenizes at) is drawn over this canvas, but **only while training is active** — otherwise it's just the scene, no grid.

**Vision Encoder box** — a fixed 176×176 `<canvas>` split into a 4×4 grid of patches (each patch itself a 4×4 pixel block, i.e. 16×16 pixels total = "16 tokens"), with a 3px gap and alternating tint between patches (`#f6f6f6`/`#efefef`) to visually read as a patch-embedding grid. At rest all pixels render flat light-gray (`rgb(244,244,244)`) — dormant. While training, it renders a live, heavily-downsampled (16×16) re-render of the current Rollout scene (literally: draw the scene at ~184×186 to an offscreen canvas, then `drawImage` scaled down to 16×16, then blow that back up pixelated) with noise added proportional to the current loss value — so it starts noisy and "denoises" into a clean crop of the scene as loss converges. Patches containing non-background content get a red (`accent`) border that pulses on/off using a sine wave, simulating "attention" on relevant patches.

**Language Encoder box** — shows the literal prompt text `"pick up the red block"` in small muted monospace, then the same string tokenized into pill chips: `pick → up → the → red → block`, connected by light gray arrow glyphs. Each chip is `500 12px` monospace, `#f4f4f4` background, `1px solid rgba(0,0,0,.1)` border, `6px 10px` padding, `3px` radius (the last token, "block", never gets a special highlight color — keep it visually identical to the others). Under each chip is a tiny 26×4px horizontal bar (`#ececec` track, accent fill) representing a **live attention weight** for that token, with a small caption below the row: `"↳ Live attention weight"` (9px monospace, uppercase, very muted `#c3bab3`). Weights are computed each frame from training progress: "pick" starts highest and decays, "the"/"up" stay low throughout, "red" and "block" ramp up as loss converges (the model's attention shifting onto the target object as it learns). All of this content (prompt text + token row) is invisible (`opacity:0`) until training starts.

**Rollout box** (was "Output · Action Rollout") — a `<canvas>`, 100% width × 186px, visually **identical scene renderer** to the Demonstration canvas (same shared draw routine: floor line, black box left, red box right, 2-link arm, single-dot end effector, sharp/no-rounded box corners) so the two read as the same robot. At rest, arm is upright (same pose as Demonstration's rest pose). Once training starts, 2-link inverse kinematics drives the arm to reach for the red block; the reach has jitter/noise proportional to the current loss (chaotic when loss is high, clean/settled as it converges), and a trailing scribble line (accent color, opacity + amplitude tied to loss) traces recent end-effector positions — "chaotic trails when loss is near 0.82, snapping to a clean single-arc trajectory as it converges near zero," per the original spec.

**Wires** — Orthogonal (90°, not diagonal, not curved) connector lines drawn in an SVG layer, computed at runtime from the actual measured DOM positions of each box (so they stay correct on resize):
  - Demonstration (image) → up → across → into the left edge of **Vision Encoder**, then out its right edge → across → down into the top of **Rollout**.
  - The floating prompt textarea (see below) → down → across → into the left edge of **Language Encoder**, then out its right edge → across → up into the bottom of **Rollout**.
  - Lines must never cross the hero title/buttons — the routing goes around them, entering/exiting each box at its vertical/horizontal center.
  - Base wire: `rgba(0,0,0,.16)`, `1.4px`, always visible (this is the answer to "should wires show before training" — **yes**, always).
  - "Energized" overlay (only opacity'd in while training): a low-opacity accent-colored duplicate of the same path (`opacity:.16`), plus a **continuous flowing signal**: a dashed accent stroke (`dasharray: 8 22`, `stroke-width:1.6`, `opacity:.55`) animating its `stroke-dashoffset` continuously (`@keyframes dashmove`, `3.6s linear infinite`, offset by `.6s` between the two branches) — a slow, gentle marching effect, not a fast pulse.

**Prompt textarea** — floats **above** the Demonstration box (`position:absolute; bottom:100%; margin-bottom:10px`, full width of that box) so it doesn't affect the box's centering/layout. Styled like a real textarea: `1px solid rgba(0,0,0,.2)` border, white bg, `3px` radius, `9px 11px 12px` padding, `500 12.5px` monospace, plus a small fake resize-grip glyph in the bottom-right corner (`repeating-linear-gradient` diagonal stripes). Content: `pick up the red block`. This element is the **source** of the Language Encoder wire (not the Demonstration box).

**Training control bar** — a flat white bar (`1px solid rgba(0,0,0,.1)`, `3px` radius, `12px 14px` padding), flex row, `gap:16px`:
  - Status dot + label: a small 8px circle (`#c9c9c9` idle / accent-red while training, with a `pulse` scale+fade keyframe animation while training) next to `IDLE` / `TRAINING` text (`600 11px` monospace, uppercase, `letter-spacing:.1em`).
  - **Loss curve**: label "browser training · MSE loss" (note: explicitly names the loss function as **MSE**) above a small (`100% × 34px`) canvas sparkline. Plots the last ~90 loss samples as a line + very-light fill under it, baseline (`y=0` loss) at the bottom, so the curve slopes from top-left (high loss, 0.82) down to bottom-right (low loss, ~0.03-0.05) as training "converges" — conventional loss-curve orientation. Draws a small dot at the current value.
  - Current loss value, right-aligned, `600 12px` monospace, e.g. `0.051`.
  - **Start Training / Reset** button: black bg / white text when idle ("Start Training"); white bg / black text with a `rgba(0,0,0,.22)` border when running ("Reset"). Always clickable — this is the only interactive control in the whole design.

## Interactions & Behavior
- **Idle → Training** is the only state transition, toggled by the one button in the training bar.
- **Nothing animates or changes until "Start Training" is clicked** — with two intentional exceptions: the Demonstration canvas's looping pick-up animation always plays (even at rest), and the base (non-energized) wires are always visible.
- On click **Start Training**:
  - `training = true`, loss resets to `0.82`.
  - A 150ms interval nudges `loss` toward `~0.038` with a small easing + jitter term (`loss += (0.038 - loss) * 0.05 + noise`), until Reset is clicked. This is a fake/simulated loss curve for visual purposes only, not a real model.
  - Hero fades to 25% opacity.
  - Vision Encoder, Language Encoder content, and Rollout's reach/trail become visible/active.
  - Every box's label position animates from "centered over the box" to "normal top-left flow position" (only property that visually moves — the boxes themselves stay in their fixed anchor positions the whole time).
  - Wires "energize": the moving dashed overlay and brighter base opacity fade in.
- On click **Reset** (button becomes this once training is true): stops the interval, clears the trajectory trail and loss history, returns the Rollout arm to its upright rest pose, sets `training = false`, loss back to `0.82` for display.
- All per-frame drawing (canvases) runs on a single `requestAnimationFrame` loop, redrawing every box every frame (draws are cheap/idempotent; state like `training`/`loss` gates what's actually rendered).
- Canvases are rendered at `devicePixelRatio` resolution (not just CSS size) for crisp rendering on retina displays — a common thing to get wrong, called out explicitly here.

## State Management
Minimal local component state:
- `training: boolean` — the only user-facing state.
- `loss: number` — decays from 0.82 toward ~0.038 while training, driven by an interval timer.
- `lossHistory: number[]` — rolling buffer (~90 samples) of `loss` for the sparkline.
- `arm: {a1, a2}` — the Rollout arm's two joint angles, smoothed/lerped each frame toward an IK-computed target when training, or a fixed upright pose when idle.
- `trail: {x,y}[]` — rolling buffer of recent end-effector positions for the Rollout's trajectory scribble.
- No routing, no server data — everything is client-side/decorative.

## Design Tokens
(Extracted from the live site's `globals.css` — use the codebase's real token names/variables if they exist, these are just the resolved values.)

**Colors**
- `--red` / accent: `#e12d1a` — links, active/training states, CTA highlights, tokenizer highlight, loss curve.
- `--ink`: `#111111` — primary text/headings.
- `--body`: `#4a4a4a` — body copy.
- `--muted`: `#9a9a9a` — labels, meta text.
- `--bg`: `#ffffff`.
- `--bg-alt` / card bg: `#fcfcfc`.
- `--border`: `rgba(0,0,0,0.1)` hairline, used everywhere instead of shadows.
- Nav wash: `rgba(225,45,26,0.04)` + `backdrop-filter: blur(10px)`.

**Fonts**
- Sans: `"Helvetica Neue", Archivo, Arial, sans-serif` — all UI text/headings.
- Mono: `ui-monospace, Menlo, monospace` — labels, meta, tags, all the ML-diagram chrome.
- No web fonts loaded.

**Type scale used in this design**
- Hero H1: `700 46px/1.02`, `letter-spacing:-.03em`.
- Hero tag / mono labels: `600 10–11px`, `letter-spacing:.16–.18em`, uppercase.
- Body/prompt mono: `500 12–12.5px/1.45`.
- Buttons: `600 14px`.

**Radius / borders / effects**
- Border radius: `3px` everywhere (buttons, cards, tags).
- Hairline borders `1px solid rgba(0,0,0,.1)` instead of shadows — **no box-shadows anywhere**.
- Transitions: simple `ease`, 0.18s–0.55s, on `opacity`/`width`/`left`/`top`/`transform` only.

## Assets
No external images/icons. Everything (robot arm, blocks, patch grid, tokens, loss curve) is drawn with inline SVG (wires) and `<canvas>` 2D drawing (everything else). No asset files to hand off.

## Files
- `Landing.dc.html` — the complete design reference. Open directly in a browser to see/interact with it. All markup is inline-styled HTML; all behavior is in the single `<script>` class body at the bottom of the file (search for `class Component`). Treat the imperative canvas-drawing methods (`paintScene`, `drawVision`, `drawDemo`, `drawArm`, `drawLossCurve`, `layoutWires`) as the executable spec for exactly what each element should look like and how it should animate — they're plain, dependency-free JS/Canvas and read like pseudocode.
