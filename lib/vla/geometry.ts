// Shared 2-link arm geometry for the VLA hero, in a normalized 1x1 workspace
// with y UP (unlike canvas coords — the renderers in scene.ts do the flip).
// The arm base is anchored at (0.5, 0.2); blocks rest on the floor (y = 0) at
// per-scene randomized positions (see examples.ts Layout). Everything
// downstream — the analytical IK expert, the training-sample synthesizer,
// the rollout integrator and both canvas renderers — works in these units so
// the vision input and the expert labels can never disagree about where
// things are.

// All values below are knobs — tune them in lib/vla/config.ts. See the CONFIG
// comments there for the reach-circle / pose-range / block-size rationale.
import { CONFIG } from "./config";

export const L1 = CONFIG.arm.l1;
export const L2 = CONFIG.arm.l2;
export const BASE = CONFIG.arm.base;

/** Reference block side length, in workspace units — the SSR-default size and
    the fallback when a block carries no explicit size. Per-scene blocks
    randomize their side length in [BLOCK_MIN, BLOCK_MAX] (see examples.ts). */
export const BLOCK = CONFIG.block.ref;
export const BLOCK_MIN = CONFIG.block.min;
export const BLOCK_MAX = CONFIG.block.max;

/** Upright rest pose (straight up from the base). */
export const REST: [number, number] = CONFIG.arm.rest;

export const THETA1_RANGE: [number, number] = CONFIG.arm.theta1Range;
export const THETA2_RANGE: [number, number] = CONFIG.arm.theta2Range;

/**
 * Analytical 2-link inverse kinematics. Target coords are relative to the
 * arm base (subtract BASE before calling). Returns the safe fallback [0, 0]
 * when the target is geometrically out of reach.
 */
export function solveIK(
  targetX: number,
  targetY: number,
  l1 = L1,
  l2 = L2
): [number, number] {
  const dSq = targetX * targetX + targetY * targetY;
  const cosAngle2 = (dSq - l1 * l1 - l2 * l2) / (2 * l1 * l2);

  if (Math.abs(cosAngle2) > 1) return [0, 0];

  // elbow-UP branch: which sine sign is "up" depends on the target's side of
  // the base (negative for targets to the right, positive to the left) — a
  // fixed sign choice puts the elbow underground on one side and yields
  // joint targets outside the sampled pose ranges
  const sinAngle2 = (targetX >= 0 ? -1 : 1) * Math.sqrt(1 - cosAngle2 * cosAngle2);
  const theta2 = Math.atan2(sinAngle2, cosAngle2);

  const k1 = l1 + l2 * cosAngle2;
  const k2 = l2 * sinAngle2;
  let theta1 = Math.atan2(targetY, targetX) - Math.atan2(k2, k1);

  // The atan2 difference can wrap (e.g. -3.93 instead of the identical
  // +2.35 for a left-side block). As a regression LABEL the raw value
  // matters: un-normalized it tells the policy to push theta1 the wrong way
  // around, through the joint limit. Wrap into [-pi/2, 3pi/2), the band
  // around the sampled theta1 range.
  while (theta1 < -Math.PI / 2) theta1 += 2 * Math.PI;
  while (theta1 >= (3 * Math.PI) / 2) theta1 -= 2 * Math.PI;

  return [theta1, theta2];
}

/** Grasp target for a block of side `size` centered at floor position x: the
    block CENTER (y = size/2) — the effector moves into the block before the
    grasp/lift. A bigger block is grasped higher, so its size feeds the IK. */
export function graspTarget(x: number, size = BLOCK) {
  return { x, y: size / 2 };
}

/** IK joint angles that put the end effector at a block's grasp point. */
export function ikToX(x: number, size = BLOCK): [number, number] {
  const t = graspTarget(x, size);
  return solveIK(t.x - BASE.x, t.y - BASE.y);
}

/** Forward kinematics: elbow + end-effector positions in workspace units. */
export function fk(a1: number, a2: number) {
  const j1x = BASE.x + Math.cos(a1) * L1;
  const j1y = BASE.y + Math.sin(a1) * L1;
  const ex = j1x + Math.cos(a1 + a2) * L2;
  const ey = j1y + Math.sin(a1 + a2) * L2;
  return { j1x, j1y, ex, ey };
}

export function clamp(v: number, lo: number, hi: number) {
  return Math.max(lo, Math.min(hi, v));
}
