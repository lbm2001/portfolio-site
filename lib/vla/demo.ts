// Scripted demonstration trajectories for the Demonstration box. A fresh
// plan is generated for every cycle: a new random 8-block layout, a random
// commanded color, and noisy waypoints so no two demonstrations are
// identical. Same shape as a successful rollout episode: descend to the
// commanded block's CENTER, grasp, lift STRAIGHT UP to the rest pose, hold
// it aloft, then release — the block resets to the floor for the next cycle
// (never carried back down).
//
// The rollout box runs in LOCKSTEP with this cycle (same layout + command,
// reset at every cycle boundary; see Hero.tsx), and it runs a FROZEN snapshot
// of the policy taken at that boundary — a fixed policy for the whole attempt
// (how a real rollout works) rather than the live model drifting mid-reach as
// background training updates it. So each of the ~5 cycles before convergence
// is a clean side-by-side readout of ONE policy generation vs. the expert.
// The demo MOTION is defined in ABSOLUTE time (the *_MS phase constants
// below), NOT as fractions of the period, so the crisp scripted reach keeps
// the same speed regardless of the period; the period only sets how much
// resting tail follows the release before the next cycle begins.

import { BASE, REST, graspTarget, solveIK } from "./geometry";
import { blockOfColor, type Layout } from "./examples";

// Synced cycle length. Trimmed from 8000 to cut most of the post-release
// resting tail: the scripted demo's motion finishes at ~4.26s (release; see
// the phase sums below), so 5000 leaves only a short beat at REST before the
// next cycle. At ~5s/cycle the viewer sees ~5 policy generations in the ~25s
// before convergence unlocks "try it" mode. Hero.tsx's REACH_TIMEOUT must
// stay >= this in frames or a rollout would give up before the cycle reset.
export const DEMO_PERIOD_MS = 5000;

// Absolute-time trajectory phases (ms), independent of DEMO_PERIOD_MS: the
// scripted arm always reaches at this fixed, crisp speed. Sum of the motion
// phases (~2.85s) + HOLD is the "active" part; the remainder of the period is
// the arm resting at home before the next cycle.
const VIA_MS = 672; // rest -> mid-trajectory waypoint
const REACH_MS = 672; // waypoint -> block center
const SETTLE_MS = 420; // settle on the block center
const LIFT_MS = 1092; // straight up back to rest
const GRASP_AT_MS = 1430; // block grasped mid-settle (carry begins)
const HOLD_MS = 1400; // held aloft after the lift completes (was ~500)

export interface DemoPlan {
  color: number; // index into COLORS
  via: [number, number];
  reach: [number, number];
}

export interface DemoPose {
  a1: number;
  a2: number;
  /** COLORS index of the carried block, or null. */
  carry: number | null;
}

const jitter = (amp: number) => (Math.random() - 0.5) * 2 * amp;
const lerp = (a: number, b: number, u: number) => a + (b - a) * u;
const ease = (x: number) =>
  x <= 0 ? 0 : x >= 1 ? 1 : (1 - Math.cos(x * Math.PI)) / 2;

export function makeDemoPlan(layout: Layout, color: number): DemoPlan {
  const g = graspTarget(blockOfColor(layout, color).x);
  const reach = solveIK(g.x + jitter(0.012) - BASE.x, g.y + jitter(0.008) - BASE.y);
  // a noisy mid-trajectory waypoint so every approach path differs
  const via: [number, number] = [
    lerp(REST[0], reach[0], 0.5) + jitter(0.3),
    lerp(REST[1], reach[1], 0.5) + jitter(0.45),
  ];
  return { color, via, reach };
}

/** Evaluate the demonstration at cycle phase t in [0,1). */
export function demoPose(plan: DemoPlan, t: number): DemoPose {
  const seg = (
    from: [number, number],
    to: [number, number],
    u: number
  ): [number, number] => [
    lerp(from[0], to[0], ease(u)),
    lerp(from[1], to[1], ease(u)),
  ];

  // Work in absolute ms so the motion speed is independent of the (long)
  // period — the tail past LIFT+HOLD is just the arm resting at home.
  const ms = t * DEMO_PERIOD_MS;
  const reachStart = VIA_MS;
  const settleStart = reachStart + REACH_MS;
  const liftStart = settleStart + SETTLE_MS;
  const liftEnd = liftStart + LIFT_MS;
  const releaseAt = liftEnd + HOLD_MS;

  let pose: [number, number];
  if (ms < reachStart) pose = seg(REST, plan.via, ms / VIA_MS);
  else if (ms < settleStart) pose = seg(plan.via, plan.reach, (ms - reachStart) / REACH_MS);
  else if (ms < liftStart) pose = plan.reach; // settle on the block centre
  else if (ms < liftEnd) pose = seg(plan.reach, REST, (ms - liftStart) / LIFT_MS); // lift up
  else pose = REST; // held at the top, then empty-handed until the cycle ends

  // grasped once the effector settles on the centre, released after the
  // hold at the top
  const carry = ms >= GRASP_AT_MS && ms < releaseAt ? plan.color : null;
  return { a1: pose[0], a2: pose[1], carry };
}
