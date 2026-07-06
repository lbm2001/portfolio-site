// Canvas renderers for the VLA hero. Two flavors of the same scene:
//  - paintScene: the design-styled display renderer (Demonstration + Rollout
//    boxes) — floor line, pedestal, grey links, colored blocks, trail.
//  - paintSilhouette: the flattened high-contrast version (white bg, grey
//    arm, blocks in their own colors) that gets downsampled to 32x32 and fed
//    to the model. Training samples and live rollout inference BOTH go
//    through this exact renderer, so the policy never sees a distribution it
//    wasn't trained on.
// Both map the y-up unit workspace of geometry.ts onto y-down canvas pixels.

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
  const S = 0.8 * H;
  const floorY = 0.86 * H;
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
}

export function paintScene(
  ctx: CanvasRenderingContext2D,
  W: number,
  H: number,
  { a1, a2, layout, accent, trail, lossNorm = 0, carry }: PaintOpts
) {
  const m = sceneMap(W, H);
  const bx = m.X(BASE.x);
  const by = m.Y(BASE.y);
  const box = BLOCK * m.S;

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
  // and the grip on a carried one stay visible
  for (const b of layout) {
    if (b.color === carry) continue; // carried block leaves its floor spot
    ctx.fillStyle = COLORS[b.color].hex;
    ctx.fillRect(m.X(b.x) - box / 2, m.floorY - box, box, box);
  }
  if (carry !== null && carry !== undefined) {
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

  // end effector — a solid grey dot on top of everything; stays the same
  // grey whether reaching or carrying a block
  ctx.fillStyle = "#6f6f6f";
  ctx.beginPath();
  ctx.arc(ex, ey, 4, 0, 7);
  ctx.fill();
}

// Blocks render a touch larger in the model's-eye view: at 32x32 a
// display-size block is ~2px wide — the boost keeps each color's pixels
// clearly present after the downsample without changing the display scene.
const SIL_BLOCK = BLOCK * 1.3;

/**
 * The model's-eye view: white background, the 8 colored blocks at their
 * layout positions, grey arm. Two grey link tones + a dark effector dot keep
 * the pose readable after the 32x32 downsample — the network has to regress
 * joint angles AND localize the named color from this image alone.
 */
export function paintSilhouette(
  ctx: CanvasRenderingContext2D,
  size: number,
  a1: number,
  a2: number,
  layout: Layout,
  carry?: number | null
) {
  const m = sceneMap(size, size);
  const box = SIL_BLOCK * m.S;

  const bx = m.X(BASE.x);
  const by = m.Y(BASE.y);
  const j1x = m.X(BASE.x + Math.cos(a1) * L1);
  const j1y = m.Y(BASE.y + Math.sin(a1) * L1);
  const ex = m.X(BASE.x + Math.cos(a1) * L1 + Math.cos(a1 + a2) * L2);
  const ey = m.Y(BASE.y + Math.sin(a1) * L1 + Math.sin(a1 + a2) * L2);

  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, size, size);

  // blocks at their floor spots — the carried one leaves its spot and is
  // redrawn at the gripper, so the model's-eye view matches the lifted demo
  for (const b of layout) {
    if (b.color === carry) continue;
    ctx.fillStyle = COLORS[b.color].hex;
    ctx.fillRect(m.X(b.x) - box / 2, m.floorY - box, box, box);
  }
  if (carry !== null && carry !== undefined) {
    ctx.fillStyle = COLORS[carry].hex;
    ctx.fillRect(ex - box / 2, ey - box / 2, box, box);
  }

  ctx.lineCap = "round";
  ctx.strokeStyle = "#666666";
  ctx.lineWidth = size * 0.04;
  ctx.beginPath();
  ctx.moveTo(bx, by);
  ctx.lineTo(j1x, j1y);
  ctx.stroke();
  ctx.strokeStyle = "#aaaaaa";
  ctx.lineWidth = size * 0.03;
  ctx.beginPath();
  ctx.moveTo(j1x, j1y);
  ctx.lineTo(ex, ey);
  ctx.stroke();

  ctx.fillStyle = "#333333";
  ctx.beginPath();
  ctx.arc(ex, ey, size * 0.05, 0, 7);
  ctx.fill();
}
