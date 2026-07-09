// Canvas renderers for the VLA hero. Two flavors of the same scene:
//  - paintScene: the design-styled display renderer (Demonstration + Rollout
//    boxes) — floor line, pedestal, grey links, colored blocks, trail.
//  - paintSilhouette: the flattened high-contrast version (white bg, grey
//    arm, blocks in their own colors) that gets downsampled to 32x32 and fed
//    to the model. Training samples and live rollout inference BOTH go
//    through this exact renderer, so the policy never sees a distribution it
//    wasn't trained on.
// Both map the y-up unit workspace of geometry.ts onto y-down canvas pixels.

import { CONFIG } from "./config";
import { BASE, BLOCK, L1, L2 } from "./geometry";
import { COLORS, type Layout } from "./examples";

export interface SceneMap {
  X: (x: number) => number;
  Y: (y: number) => number;
  S: number;
  floorY: number;
}

// One isotropic scale for both axes (links must not stretch); sized so the
// fully extended arm stays inside the canvas when upright.
export function sceneMap(W: number, H: number): SceneMap {
  const S = CONFIG.render.sceneScale * H;
  const floorY = CONFIG.render.floorY * H;
  return {
    X: (x) => W * 0.5 + (x - 0.5) * S,
    Y: (y) => floorY - y * S,
    S,
    floorY,
  };
}

/** End-effector position in canvas pixels (for the rollout trail). */
export function effectorPx(W: number, H: number, a1: number, a2: number) {
  const m = sceneMap(W, H);
  const ex = BASE.x + Math.cos(a1) * L1 + Math.cos(a1 + a2) * L2;
  const ey = BASE.y + Math.sin(a1) * L1 + Math.sin(a1 + a2) * L2;
  return { x: m.X(ex), y: m.Y(ey) };
}

export interface PaintOpts {
  a1: number;
  a2: number;
  /** The 8-block scene layout to draw. */
  layout: Layout;
  accent: string;
  /** Recent end-effector positions (canvas px); drawn only when provided. */
  trail?: { x: number; y: number }[] | null;
  /** Normalized loss in [0,1] — drives trail jitter/opacity. */
  lossNorm?: number;
  /** COLORS index of the block held at the gripper (its floor spot empties). */
  carry?: number | null;
  /** Gripper state: 1 = closed (pinched jaws), 0 = open (splayed jaws).
      Omitted → the plain solid-dot effector (idle sway, no grasp in play). */
  grip?: 0 | 1;
}

export function paintScene(
  ctx: CanvasRenderingContext2D,
  W: number,
  H: number,
  { a1, a2, layout, accent, trail, lossNorm = 0, carry, grip }: PaintOpts
) {
  const m = sceneMap(W, H);
  const bx = m.X(BASE.x);
  const by = m.Y(BASE.y);

  const j1x = m.X(BASE.x + Math.cos(a1) * L1);
  const j1y = m.Y(BASE.y + Math.sin(a1) * L1);
  const ex = m.X(BASE.x + Math.cos(a1) * L1 + Math.cos(a1 + a2) * L2);
  const ey = m.Y(BASE.y + Math.sin(a1) * L1 + Math.sin(a1 + a2) * L2);

  // floor line — full width
  ctx.strokeStyle = "#e6e6e6";
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.moveTo(0, m.floorY);
  ctx.lineTo(W, m.floorY);
  ctx.stroke();

  // trajectory trail — chaotic (jittered, faint) at high loss, clean near zero
  if (trail && trail.length > 1) {
    ctx.lineWidth = 1.4;
    ctx.strokeStyle = accent;
    ctx.globalAlpha = 0.28 + 0.4 * (1 - lossNorm);
    ctx.beginPath();
    trail.forEach((p, i) => {
      const n = lossNorm * 7;
      const px = p.x + (Math.random() - 0.5) * n;
      const py = p.y + (Math.random() - 0.5) * n;
      if (i) ctx.lineTo(px, py);
      else ctx.moveTo(px, py);
    });
    ctx.stroke();
    ctx.globalAlpha = 1;
  }

  // blocks first — the WHOLE arm draws over them so the reach into a block
  // and the grip on a carried one stay visible. Each block draws at its own
  // randomized side length, bottom at its rest height (y, normally 0).
  for (const b of layout) {
    if (b.color === carry) continue; // carried block leaves its floor spot
    const box = b.size * m.S;
    const rest = (b.y ?? 0) * m.S;
    ctx.fillStyle = COLORS[b.color].hex;
    ctx.fillRect(m.X(b.x) - box / 2, m.floorY - rest - box, box, box);
  }
  if (carry !== null && carry !== undefined) {
    const box = (layout.find((b) => b.color === carry)?.size ?? BLOCK) * m.S;
    ctx.fillStyle = COLORS[carry].hex;
    ctx.fillRect(ex - box / 2, ey - box / 2, box, box);
  }

  // base pedestal, from the shoulder joint down to the floor
  ctx.fillStyle = "#2b2b2b";
  ctx.fillRect(bx - 5, by, 10, m.floorY - by);
  ctx.fillRect(bx - 13, m.floorY - 6, 26, 6);

  // links + revolute joints
  ctx.strokeStyle = "#8a8a8a";
  ctx.lineCap = "round";
  ctx.lineWidth = 7;
  ctx.beginPath();
  ctx.moveTo(bx, by);
  ctx.lineTo(j1x, j1y);
  ctx.stroke();
  ctx.lineWidth = 5.5;
  ctx.beginPath();
  ctx.moveTo(j1x, j1y);
  ctx.lineTo(ex, ey);
  ctx.stroke();
  ctx.lineWidth = 2;
  ctx.strokeStyle = "#8a8a8a";
  ctx.fillStyle = "#fff";
  ctx.beginPath();
  ctx.arc(bx, by, 4.5, 0, 7);
  ctx.fill();
  ctx.stroke();
  ctx.beginPath();
  ctx.arc(j1x, j1y, 3.5, 0, 7);
  ctx.fill();
  ctx.stroke();

  // end effector — a solid circle on top of everything. CLOSED (grasping) is
  // grey; OPEN (approaching) is white with a thin grey outline so it still
  // reads against the scene; idle sway (no grip state) stays the plain grey dot.
  ctx.beginPath();
  ctx.arc(ex, ey, 4, 0, 7);
  if (grip === 0) {
    ctx.fillStyle = "#fff";
    ctx.fill();
    ctx.lineWidth = 1.2;
    ctx.strokeStyle = "#6f6f6f";
    ctx.stroke();
  } else {
    ctx.fillStyle = "#6f6f6f";
    ctx.fill();
  }
}

// Blocks render a touch larger in the model's-eye view: a display-size block
// is only a few px wide after the downsample — this per-block boost keeps each
// color's pixels clearly present without changing the display scene.
const SIL_BLOCK_SCALE = CONFIG.render.silBlockScale;

/**
 * The model's-eye view: white background, the layout's colored blocks at
 * their positions, grey arm. Two grey link tones + a dark effector dot keep
 * the pose readable after the 32x32 downsample — the network has to regress
 * joint angles AND localize the named color from this image alone.
 */
export function paintSilhouette(
  // union: the trainer renders silhouettes on an OffscreenCanvas inside its
  // worker; Hero's model's-eye panel still paints onto a DOM canvas
  ctx: CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D,
  size: number,
  a1: number,
  a2: number,
  layout: Layout,
  carry?: number | null
) {
  const m = sceneMap(size, size);

  const bx = m.X(BASE.x);
  const by = m.Y(BASE.y);
  const j1x = m.X(BASE.x + Math.cos(a1) * L1);
  const j1y = m.Y(BASE.y + Math.sin(a1) * L1);
  const ex = m.X(BASE.x + Math.cos(a1) * L1 + Math.cos(a1 + a2) * L2);
  const ey = m.Y(BASE.y + Math.sin(a1) * L1 + Math.sin(a1 + a2) * L2);

  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, size, size);

  // blocks at their rest spots — the carried one leaves its spot and is
  // redrawn at the gripper, so the model's-eye view matches the lifted demo.
  // Each block draws at its own randomized side length (× the model-view
  // boost — the boost widens the block, the rest height y stays true).
  for (const b of layout) {
    if (b.color === carry) continue;
    const box = b.size * SIL_BLOCK_SCALE * m.S;
    const rest = (b.y ?? 0) * m.S;
    ctx.fillStyle = COLORS[b.color].hex;
    ctx.fillRect(m.X(b.x) - box / 2, m.floorY - rest - box, box, box);
  }
  if (carry !== null && carry !== undefined) {
    const box =
      (layout.find((b) => b.color === carry)?.size ?? BLOCK) *
      SIL_BLOCK_SCALE *
      m.S;
    ctx.fillStyle = COLORS[carry].hex;
    ctx.fillRect(ex - box / 2, ey - box / 2, box, box);
  }

  // The robot body — pedestal, foot, links, effector — is kept in-frame (real
  // VLAs see their own mount) but deliberately rendered LIGHT and THIN so the
  // two color blocks stay the most salient thing in the 32px input. The action
  // label is pose-INDEPENDENT, so the arm and base are non-informative for the
  // target; they're present only to match the rollout's live view, not to be
  // read. After the inverted preprocessing (1 - pixel) a lighter grey is a
  // weaker activation, so pushing these toward #b0/#c0 lets the saturated
  // blocks dominate the conv branch rather than the big grey structures.

  // base pedestal (shoulder joint down to the floor) + foot
  ctx.fillStyle = "#bcbcbc";
  ctx.fillRect(bx - size * 0.02, by, size * 0.04, m.floorY - by);
  ctx.fillRect(bx - size * 0.06, m.floorY - size * 0.03, size * 0.12, size * 0.03);

  ctx.lineCap = "round";
  ctx.strokeStyle = "#a8a8a8";
  ctx.lineWidth = size * 0.028;
  ctx.beginPath();
  ctx.moveTo(bx, by);
  ctx.lineTo(j1x, j1y);
  ctx.stroke();
  ctx.strokeStyle = "#c2c2c2";
  ctx.lineWidth = size * 0.022;
  ctx.beginPath();
  ctx.moveTo(j1x, j1y);
  ctx.lineTo(ex, ey);
  ctx.stroke();

  // effector: a small light locator dot, not the loudest feature. At size*0.05
  // it was ~3.2px at 32 — bigger and darker (#333) than the ~2.7px color
  // blocks it exists to help grasp; shrunk + lightened so the blocks win.
  ctx.fillStyle = "#8f8f8f";
  ctx.beginPath();
  ctx.arc(ex, ey, size * 0.03, 0, 7);
  ctx.fill();
}
