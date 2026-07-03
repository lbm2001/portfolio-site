# Handoff: ML/Robotics Portfolio Website

## Overview
A personal portfolio website for an ML/robotics engineer. The design is minimal and Swiss-inspired — clean grid, strong typography, a single red accent color — with a live canvas animation in the hero that simulates a reinforcement-learning agent training in real time. Three hero animation variants are provided (see Screens below); the developer should pick one.

## About the Design Files
`Portfolio.dc.html` is a **design reference prototype built in HTML**. It is not production code. The task is to **recreate this design in your target stack** (React, Next.js, SvelteKit, etc.) using its established patterns and component libraries. Do not ship the HTML file directly.

The canvas animation logic is intentional production-quality code — it is fine to lift it (or adapt it) directly into whatever framework you choose.

## Fidelity
**High-fidelity.** Colors, typography, spacing, layout and animations are final. Recreate pixel-accurately.

---

## Design Tokens

### Colors
| Token | Hex | Usage |
|---|---|---|
| `--red` | `#E12D1A` | Accent, tags, highlights, logo border |
| `--ink` | `#111111` | Headlines, nav, buttons |
| `--body` | `#4a4a4a` / `#5a5a5a` | Body copy |
| `--muted` | `#9a9a9a` | Labels, dates, captions |
| `--bg` | `#ffffff` | Page background |
| `--bg-alt` | `#FAFAFA` / `#fcfcfc` | Alternate section background |
| `--border` | `rgba(0,0,0,0.10)` | All dividers and borders |

### Typography
- **Font stack:** `'Helvetica Neue', Archivo, Arial, sans-serif`
- **Monospace:** `ui-monospace, Menlo, monospace` (labels, dates, tags)
- **Google Font import:** `Archivo` weights 400, 500, 600, 700

| Role | Size | Weight | Letter-spacing |
|---|---|---|---|
| Hero H1 | 46–66px / 1.02 | 700 | -0.035em |
| Section lead | 26px / 1.5 | 400 | -0.015em |
| Project title | 21–23px / 1.22 | 600 | -0.012em |
| Body | 14–16px / 1.55 | 400 | — |
| Nav links | 14px / 1 | 500 | — |
| Mono label | 10–12px | 600 | 0.14–0.18em (uppercase) |

### Spacing
Section padding: `80px 56px` (desktop). CV strip: `54px 56px`. Footer: `40px 56px`.

### Border Radius
- Buttons: `3px`
- Tags/pills: `999px`
- Logo box: `4px`

---

## Screens / Views

### Navigation (sticky)
- Height: ~62px. Padding: `18px 56px`.
- Background: `rgba(255,255,255,0.86)` + `backdrop-filter: blur(12px)`.
- Border-bottom: `rgba(0,0,0,0)` at top, transitions to `rgba(0,0,0,0.10)` on scroll.
- **Hide on scroll down, show on scroll up** (translateY(-118%) / 0), transition `0.35s ease`.
- Left: `YN` monogram in a 26×26px red-bordered box (`border: 1.5px solid #E12D1A`, `border-radius: 4px`) + "Your Name" in 14px/600.
- Right: nav links — About, Projects, Blog, CV — 14px/500, `color: #333`, `gap: 26px`.

### Hero (3 variants — choose one)

All variants share the same hero structure:
- Full-width header, `min-height: 520–720px`.
- Absolutely-positioned `<canvas>` behind the content (`opacity: 0.9–1`).
- Centered content column (photo → tag line → H1 → body → CTA buttons), `gap: 18–24px`.
- Photo placeholder: `112×112px` circle, striped gray gradient, `border: 1px solid rgba(0,0,0,0.1)`.
- CTA: "Get in touch" (black fill) + "Download CV" (outlined), both `14px/600`, `border-radius: 3px`, `padding: 14px 24px`.

#### Variant 3a — CartPole RL (`data-anim="cartpole"`)
Full-bleed canvas, `opacity: 0.9`. Hero min-height `720px`. Centered hero text column (max-width `740px`). Includes tagline: `Robot Learning · ML Engineering` in `11px/600` monospace uppercase red.

The canvas draws a CartPole physics simulation (cart + pole on a track) that learns to balance over episodes. An episode counter + return value + sparkline chart appear bottom-left. See animation section below.

#### Variant 3b — BipedalWalker hill (`data-anim="walker"`)
Full-bleed canvas, hero `min-height: 520px`, `padding: 80px 40px 72px`.
The canvas draws a rough terrain with a **hill** in the center. A small bipedal walker agent traverses left-to-right, learning to walk episode by episode. Subtle `rgba(225,45,26,0.04)` fill above the terrain line gives the hill a light-red tint. The hero content floats over the hill.

**This is the locked-in direction.** See variant 3c for the valley alternative.

#### Variant 3c — BipedalWalker valley (`data-anim="valley"`)
Same as 3b but terrain is **inverted into a valley**: terrain peaks at the edges and dips to ~90% canvas height at center, creating a U-shaped gorge. Fill still goes above terrain, so the valley interior carries the light-red tint. Hero `padding: 24px 40px 100px` (content sits higher).

---

## Canvas Animation — BipedalWalker (3b / 3c)

The full animation source is in `Portfolio.dc.html` inside the `class Component extends DCLogic { ... }` block — look for `function wkStep`, `function wkDraw`, `function terrainY`. It is self-contained vanilla JS and can be copied directly into a `<canvas>` component.

### Key parameters
| Parameter | Value | Effect |
|---|---|---|
| Leg lengths | L1=11, L2=10 | Agent body size |
| Walk speed | `dt * 0.026 * sp` | Lateral speed per frame |
| Episode end (fall) | `fallT > 380ms` | How long to wait after fall before new episode |
| Skill gain | `+0.16 per episode` | How fast agent improves |
| Wrap-around | `sx > w+40 → sx = -40` | Agent exits right, enters left |

### Terrain — hill (3b)
```js
function terrainY(x) {
  const hills = sin(x*0.011)*22 + sin(x*0.023+1.3)*16 + sin(x*0.0052+0.5)*18
              + sin(x*0.041+2.1)*12 + sin(x*0.067+3.5)*8 + sin(x*0.14+2.4)*6;
  const d = (x - w/2) / (w*0.30);
  const bf = exp(-d*d*1.1);
  return h*0.96 - hills - bf*(h*0.82); // hill bumps up at center
}
```

### Terrain — valley (3c)
```js
return h*0.14 + hills*0.4 + bf*(h*0.76); // valley dips down at center
```

### Canvas fill
```js
// Fill above terrain line with light red tint
ctx.beginPath();
ctx.moveTo(0, 0);
for (let x = 0; x <= w; x += 6) ctx.lineTo(x, terrainY(x));
ctx.lineTo(w, 0);
ctx.closePath();
ctx.fillStyle = 'rgba(225, 45, 26, 0.04)';
ctx.fill();
// Terrain stroke
ctx.strokeStyle = 'rgba(175, 28, 8, 0.13)';
ctx.lineWidth = 1.1;
```

### HUD / chart
A `RETURN / EPISODE` sparkline chart is drawn in a separate `<canvas>` element inserted directly after the hero header (`height: 48px`, `width: 100%`). The main canvas also shows `● TRAINING / BipedalWalker · rough terrain / episode NNN   return NNN   run N` bottom-left in `10px` monospace.

---

## Section: About (01)
- Padding `80px 56px`, `border-bottom`.
- Monospace label `01 — About` in `#9a9a9a` uppercase.
- Lead paragraph: `26px/1.5`, `color: #1c1c1c`, `max-width: 820px`.
- 4-column info grid (`grid-template-columns: repeat(4,1fr)`), each cell `padding: 20px`, divided by `1px solid rgba(0,0,0,0.10)` borders. Fields: Focus, Field, Location, Email.

## Section: Selected Work (02)
- Header row: label left + "All projects →" right link.
- `display: grid; grid-template-columns: 1fr 1fr` with outer border-top + border-left; each card adds border-right + border-bottom.
- Each card: `padding: 30px`, `min-height: 200px`, flex column.
  - Red monospace index + gray venue (space-between row)
  - Title `21px/1.22/600`
  - Blurb `14px/1.55/400 #5a5a5a`
  - Tag pills: `border: 1px solid rgba(0,0,0,0.12)`, `border-radius: 999px`, `padding: 3px 10px`, `11px/500`
  - Links: `12px` monospace, `border-bottom: 1.5px solid #E12D1A`

## Section: From the Blog (03)
- 3-column row layout per post: `grid-template-columns: 100px 96px 1fr; gap: 26px`.
- Date in `#9a9a9a` monospace, category in red uppercase monospace, title+excerpt on the right.
- Each row `padding: 22px 0`, `border-top`.

## Section: CV Download (04)
- Flex row, space-between, `padding: 54px 56px`, `background: #fcfcfc`.
- Left: red `40px/700` section number + label + title.
- Right: "CV.pdf ↓" button, black fill, `border-radius: 3px`.

## Footer
- `padding: 40px 56px`, flex space-between.
- `© 2026 Your Name` in `#9a9a9a 13px/500`.
- Links: GitHub, Scholar, Twitter, Email — `13px/500 #111`.

---

## Interactions & Behavior

- **Nav hide/show:** scroll down → `translateY(-118%)`, scroll up → `translateY(0)`. Transition `0.35s ease`. Border-bottom fades in after `scrollTop > 8px`.
- **Canvas animation:** starts on mount, cleans up on unmount (`cancelAnimationFrame`). Uses `ResizeObserver` to handle canvas resize with devicePixelRatio support.
- **Episode mini-chart:** updates every frame in a sibling canvas below the hero.
- **Agent wrap:** when `sx > canvasWidth + 40`, reset to `sx = -40` so agent walks off the right and enters from the left.

---

## Assets
- No external images. Photo slot is a CSS striped-gradient placeholder circle.
- No icons — uses Unicode characters (`↓`, `→`, `·`, `●`).
- One Google Font: [Archivo](https://fonts.google.com/specimen/Archivo).

## Files
| File | Purpose |
|---|---|
| `Portfolio.dc.html` | Full design prototype — all three hero variants (3a, 3b, 3c) side by side |

---

## Implementation Notes
1. **Start with 3b or 3c** — the hill and valley variants are the locked direction. 3a (CartPole) is an earlier exploration for reference.
2. The canvas animation should run in a `useEffect` (React) or `onMount` (Svelte) that returns a cleanup function.
3. The `terrainY` function must close over `w` and `h` (canvas dimensions), recalculated on resize.
4. The sticky nav hide/show needs scroll event listener scoped to the page scroll container.
5. All section content (projects, posts) is placeholder — replace with real data.
