// Shared 2-link arm geometry for the VLA hero, in a normalized 1x1 workspace
// with y UP (unlike canvas coords — the renderers in scene.ts do the flip).
// The arm base is anchored at (0.5, 0.2); blocks rest on the floor (y = 0) at
// per-scene randomized positions (see examples.ts Layout). Everything
// downstream — the analytical IK expert, the training-sample synthesizer,
// the rollout integrator and both canvas renderers — works in these units so
// the vision input and the expert labels can never disagree about where
// things are.

// Link lengths sized so the full reach circle (base ± 0.58) stays inside the
// rendered canvas — longer links let wild early-training poses swing the
// forearm out of the box.
export const L1 = 0.32;
export const L2 = 0.26;
export const BASE = { x: 0.5, y: 0.2 };

/** Block side length, in workspace units. */
export const BLOCK = 0.08;

/** Upright rest pose (straight up from the base). */
export const REST: [number, number] = [Math.PI / 2, 0];

// Pose-sampling ranges for synthesizing training states. theta2 spans BOTH
// elbow configurations: the IK solutions for floor blocks sit near
// |theta2| ≈ 2 rad, so a narrower range would mean the expert's own target
// configurations are never seen during training and the converged rollout
// runs out-of-distribution.
export const THETA1_RANGE: [number, number] = [-0.3, Math.PI + 0.3];
export const THETA2_RANGE: [number, number] = [-2.4, 2.4];

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

/** Grasp target for a block centered at floor position x: the block CENTER —
    the effector moves into the block before the grasp/lift. */
export function graspTarget(x: number) {
  return { x, y: BLOCK / 2 };
}

/** IK joint angles that put the end effector at a block's grasp point. */
export function ikToX(x: number): [number, number] {
  const t = graspTarget(x);
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
