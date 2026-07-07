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

import { CONFIG } from "./config";
import { BASE, REST, graspTarget, solveIK } from "./geometry";
import { blockOfColor, type Layout } from "./examples";

// Cycle length + trajectory phases are knobs — tune in lib/vla/config.ts.
// The scripted motion is defined in ABSOLUTE ms (the phases below), independent
// of the period, so the crisp reach keeps its speed regardless of how much
// resting tail the period leaves. rollout.reachTimeout (config) in frames must
// stay >= DEMO_PERIOD_MS or a rollout would give up before the cycle reset.
export const DEMO_PERIOD_MS = CONFIG.demo.periodMs;

const VIA_MS = CONFIG.demo.phases.viaMs; // rest -> mid-trajectory waypoint
const REACH_MS = CONFIG.demo.phases.reachMs; // waypoint -> block center
const SETTLE_MS = CONFIG.demo.phases.settleMs; // settle on the block center
const LIFT_MS = CONFIG.demo.phases.liftMs; // straight up back to rest
const GRASP_AT_MS = CONFIG.demo.phases.graspAtMs; // block grasped mid-settle
const HOLD_MS = CONFIG.demo.phases.holdMs; // held aloft after the lift completes

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
  const b = blockOfColor(layout, color);
  const j = CONFIG.demo.jitter;
  const g = graspTarget(b.x, b.size); // grasp height follows the block's size
  const reach = solveIK(g.x + jitter(j.graspX) - BASE.x, g.y + jitter(j.graspY) - BASE.y);
  // a noisy mid-trajectory waypoint so every approach path differs
  const via: [number, number] = [
    lerp(REST[0], reach[0], 0.5) + jitter(j.viaTheta1),
    lerp(REST[1], reach[1], 0.5) + jitter(j.viaTheta2),
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
