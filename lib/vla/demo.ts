// Scripted demonstration trajectories for the Demonstration box. A fresh
// plan is generated for every cycle: a new random layout, a random command
// from the run-config's task set, and noisy waypoints so no two
// demonstrations are identical. Each plan has the same shape as a successful
// rollout episode of its task:
//   lift : descend to the commanded block's CENTER, grasp, lift STRAIGHT UP
//          to the rest pose, hold aloft, release (the block resets to the
//          floor for the next cycle — never carried back down)
//   stack: descend, grasp, swing through a raised carry waypoint to just
//          above the REFERENCE block, settle, release — the block stays
//          seated there for the rest of the cycle (see DemoPose.placed)
//
// The rollout box runs in LOCKSTEP with this cycle (same layout + command,
// reset at every cycle boundary; see Hero.tsx), and it runs a FROZEN snapshot
// of the policy taken at that boundary — a fixed policy for the whole attempt
// (how a real rollout works) rather than the live model drifting mid-reach as
// background training updates it. So each of the cycles before convergence
// is a clean side-by-side readout of ONE policy generation vs. the expert.
// The demo MOTION is defined in ABSOLUTE time (the *_MS phase constants
// below), NOT as fractions of the period, so the crisp scripted reach keeps
// the same speed regardless of the period; the period only sets how much
// resting tail follows the release before the next cycle begins.

import { CONFIG } from "./config";
import { BASE, REST, graspTarget, ikToPlace, solveIK } from "./geometry";
import { blockOfColor, type Layout, type Sentence } from "./examples";
import type { TaskKind } from "./run-config";

// Cycle length + trajectory phases are knobs — tune in lib/vla/config.ts.
// The scripted motion is defined in ABSOLUTE ms (the phases below), independent
// of the period, so the crisp reach keeps its speed regardless of how much
// resting tail the period leaves. rollout.reachTimeout (config) in frames must
// stay >= DEMO_PERIOD_MS or a rollout would give up before the cycle reset.
export const DEMO_PERIOD_MS = CONFIG.demo.periodMs;

const VIA_MS = CONFIG.demo.phases.viaMs; // rest -> mid-trajectory waypoint
const REACH_MS = CONFIG.demo.phases.reachMs; // waypoint -> block center
const SETTLE_MS = CONFIG.demo.phases.settleMs; // settle on the block center
const LIFT_MS = CONFIG.demo.phases.liftMs; // lift: straight up back to rest
const GRASP_AT_MS = CONFIG.demo.phases.graspAtMs; // grasped mid-settle (lift/stack)
const HOLD_MS = CONFIG.demo.phases.holdMs; // lift: held aloft at the top
const CARRY_MS = CONFIG.demo.phases.carryMs; // stack: grasp -> above the ref block
const PLACE_SETTLE_MS = CONFIG.demo.phases.placeSettleMs; // stack: settle, then release
const RETURN_MS = CONFIG.demo.phases.returnMs; // stack: return to rest
const CARRY_HEIGHT = CONFIG.demo.carryHeight;

export interface DemoPlan {
  task: TaskKind;
  color: number; // index into COLORS — the block acted on
  refColor: number | null; // stack: the block it goes on top of
  via: [number, number];
  reach: [number, number];
  /** stack only: raised waypoint the carried block swings through. */
  carryVia: [number, number] | null;
  /** stack only: pose holding the carried block seated on the ref block. */
  place: [number, number] | null;
}

export interface DemoPose {
  a1: number;
  a2: number;
  /** COLORS index of the carried block, or null. */
  carry: number | null;
  /** stack: the carried block has been RELEASED onto the reference block —
      Hero seats it there in the demo layout so the scene shows the result. */
  placed: boolean;
}

const jitter = (amp: number) => (Math.random() - 0.5) * 2 * amp;
const lerp = (a: number, b: number, u: number) => a + (b - a) * u;
const ease = (x: number) =>
  x <= 0 ? 0 : x >= 1 ? 1 : (1 - Math.cos(x * Math.PI)) / 2;

export function makeDemoPlan(layout: Layout, command: Sentence): DemoPlan {
  const b = blockOfColor(layout, command.color);
  const j = CONFIG.demo.jitter;
  const g = graspTarget(b.x, b.size, b.y ?? 0); // grasp height follows size + rest
  const reach = solveIK(g.x + jitter(j.graspX) - BASE.x, g.y + jitter(j.graspY) - BASE.y);
  // a noisy mid-trajectory waypoint so every approach path differs
  const via: [number, number] = [
    lerp(REST[0], reach[0], 0.5) + jitter(j.viaTheta1),
    lerp(REST[1], reach[1], 0.5) + jitter(j.viaTheta2),
  ];
  let carryVia: [number, number] | null = null;
  let place: [number, number] | null = null;
  if (command.task === "stack" && command.refColor !== null) {
    const ref = blockOfColor(layout, command.refColor);
    // swing the carried block through a raised waypoint between the two
    // blocks so it clears the scene, then seat it on the ref block's top
    carryVia = solveIK((b.x + ref.x) / 2 - BASE.x, CARRY_HEIGHT - BASE.y);
    place = ikToPlace(
      ref.x + jitter(j.graspX),
      (ref.y ?? 0) + ref.size,
      b.size
    );
  }
  return { task: command.task, color: command.color, refColor: command.refColor, via, reach, carryVia, place };
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
  // period — the tail past each task's last phase is the arm resting at home.
  const ms = t * DEMO_PERIOD_MS;
  const reachStart = VIA_MS;
  const settleStart = reachStart + REACH_MS;

  let pose: [number, number];
  let carry: number | null = null;
  let placed = false;

  // shared approach: rest → via → the commanded block's centre
  const approach = (): [number, number] | null => {
    if (ms < reachStart) return seg(REST, plan.via, ms / VIA_MS);
    if (ms < settleStart)
      return seg(plan.via, plan.reach, (ms - reachStart) / REACH_MS);
    return null; // past the approach — task-specific tail below
  };

  if (plan.task === "stack") {
    // grasp mid-settle, swing through the carry waypoint to above the ref
    // block, settle, release (the block stays seated), return empty
    const liftStart = settleStart + SETTLE_MS;
    const carryMid = liftStart + CARRY_MS * 0.45; // reach → carryVia
    const carryEnd = liftStart + CARRY_MS; // carryVia → place
    const releaseAt = carryEnd + PLACE_SETTLE_MS;
    const returnEnd = releaseAt + RETURN_MS;
    pose =
      approach() ??
      (ms < liftStart
        ? plan.reach
        : ms < carryMid
          ? seg(plan.reach, plan.carryVia!, (ms - liftStart) / (CARRY_MS * 0.45))
          : ms < carryEnd
            ? seg(plan.carryVia!, plan.place!, (ms - carryMid) / (CARRY_MS * 0.55))
            : ms < releaseAt
              ? plan.place!
              : ms < returnEnd
                ? seg(plan.place!, REST, (ms - releaseAt) / RETURN_MS)
                : REST);
    carry = ms >= GRASP_AT_MS && ms < releaseAt ? plan.color : null;
    placed = ms >= releaseAt;
  } else {
    // lift: grasp mid-settle, straight up to rest, hold aloft, release (the
    // block resets to the floor at the next cycle)
    const liftStart = settleStart + SETTLE_MS;
    const liftEnd = liftStart + LIFT_MS;
    const releaseAt = liftEnd + HOLD_MS;
    pose =
      approach() ??
      (ms < liftStart
        ? plan.reach
        : ms < liftEnd
          ? seg(plan.reach, REST, (ms - liftStart) / LIFT_MS)
          : REST);
    carry = ms >= GRASP_AT_MS && ms < releaseAt ? plan.color : null;
  }

  return { a1: pose[0], a2: pose[1], carry, placed };
}
