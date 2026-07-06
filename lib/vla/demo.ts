// Scripted demonstration trajectories for the Demonstration box. A fresh
// plan is generated for every ~4.2s cycle: the commanded block alternates
// with the displayed training example, and every waypoint carries a little
// noise so no two demonstrations are identical. The grasp happens exactly
// when the end effector reaches the block CENTER; the lift follows.

import {
  BASE,
  BLACK_BLOCK,
  RED_BLOCK,
  REST,
  graspTarget,
  solveIK,
} from "./geometry";
import type { BlockColor } from "./examples";

export const DEMO_PERIOD_MS = 4200;

export interface DemoPlan {
  color: BlockColor;
  via: [number, number];
  reach: [number, number];
}

export interface DemoPose {
  a1: number;
  a2: number;
  carry: BlockColor | null;
}

const jitter = (amp: number) => (Math.random() - 0.5) * 2 * amp;
const lerp = (a: number, b: number, u: number) => a + (b - a) * u;
const ease = (x: number) =>
  x <= 0 ? 0 : x >= 1 ? 1 : (1 - Math.cos(x * Math.PI)) / 2;

export function makeDemoPlan(color: BlockColor): DemoPlan {
  const block = color === "red" ? RED_BLOCK : BLACK_BLOCK;
  const g = graspTarget(block);
  const reach = solveIK(g.x + jitter(0.015) - BASE.x, g.y + jitter(0.01) - BASE.y);
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

  // Same shape as a successful rollout episode: descend to the block centre,
  // grasp, lift STRAIGHT UP to the rest pose, hold ~0.5s at the top, then
  // release — the block resets to the floor for the next cycle. The arm never
  // carries the block back down, so there's no "dropping block" on the way
  // home.
  let pose: [number, number];
  if (t < 0.16) pose = seg(REST, plan.via, t / 0.16);
  else if (t < 0.32) pose = seg(plan.via, plan.reach, (t - 0.16) / 0.16);
  else if (t < 0.42) pose = plan.reach; // settle on the block centre
  else if (t < 0.68) pose = seg(plan.reach, REST, (t - 0.42) / 0.26); // lift up
  else pose = REST; // held at the top, then empty-handed until the cycle ends

  // grasped once the effector settles on the centre, released after the
  // 0.5s hold at the top (0.68 lift-done → 0.80 ≈ 0.5s of the 4.2s cycle)
  const carry = t >= 0.34 && t < 0.8 ? plan.color : null;
  return { a1: pose[0], a2: pose[1], carry };
}
