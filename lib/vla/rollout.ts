// The policy-rollout state machine — the closed-loop episode that drives the
// Rollout box. Extracted from Hero.tsx so the exact same engine can run the
// mini-vla demo/eval renderers. Hero keeps ONLY the rAF loop, canvas painting,
// overlays, panels and readouts; everything about HOW an episode unfolds
// (phases, the learned-grasp gate, carry attachment, stepping toward the last
// async-predicted target while replies are in flight) lives here.
//
// The engine owns the episode's mutable state (phase, arm pose, last predicted
// target + gaze + gripper, trail, and the prediction throttle/in-flight guard)
// and exposes a plain serializable RolloutFrame per step. It is renderer-
// agnostic: the trail is kept in workspace effector coordinates (fk output),
// and the host maps it to canvas pixels (sceneMap) at paint time.

import { CONFIG } from "./config";
import {
  REST,
  THETA1_RANGE,
  THETA2_RANGE,
  clamp,
  effectorOverBlock,
  fk,
} from "./geometry";
import type { Layout } from "./examples";
import type { PredictResult } from "./trainer.core";

// Rollout control + episode timing are knobs — tune in lib/vla/config.ts
// (CONFIG.rollout). Episode counts are frames at ~60fps; reachTimeout must stay
// above the synced demo cycle (DEMO_PERIOD_MS in frames) so a training rollout
// is bounded by the cycle reset, not by giving up early.
const STEP_GAIN = CONFIG.rollout.stepGain;
const PREDICT_MS = CONFIG.rollout.predictMs;
const TRAIL_LEN = CONFIG.rollout.trailLen;
const GRIP_RADIUS = CONFIG.gripper.radius; // effector disk radius for the grasp predicate
const GRIP_THRESHOLD = CONFIG.gripper.threshold; // sigmoid ≥ this = "close"
const NEAR_FRAMES = CONFIG.rollout.nearFrames;
const REACH_TIMEOUT = CONFIG.rollout.reachTimeout;
const SETTLE_EPS = CONFIG.rollout.settleEps; // rad/joint: carry-phase settle test
const TOP_HOLD = CONFIG.rollout.topHold;
const RETURN_FRAMES = CONFIG.rollout.returnFrames;

const ease = (x: number) =>
  x <= 0 ? 0 : x >= 1 ? 1 : (1 - Math.cos(x * Math.PI)) / 2;
const lerp = (a: number, b: number, u: number) => a + (b - a) * u;

export type RolloutPhase = "reach" | "carry" | "hold" | "return";

// The episode machine:
//   reach → approach with the gripper OPEN; the grasp fires on the LEARNED
//           gripper action: the effector fully over a block (effectorOverBlock)
//           AND the policy's predicted gripper crossing open→closed there, held
//           NEAR_FRAMES. A policy that keeps the gripper closed the whole way
//           never arms (no rising edge) → it can't enter a block with a closed
//           gripper, which is what forces the open approach. Then carry begins.
//   carry → policy-driven with the block in hand; once the arm settles at
//           the predicted target, the block is held aloft
//   hold  → holding the carried block wherever the policy parked it
//   return→ scripted empty-handed return to rest
interface Episode {
  phase: RolloutPhase;
  f: number; // frames in the current phase
  near: number;
  nearColor: number; // COLORS index of the block being hovered, or -1
  settle: number; // consecutive frames within SETTLE_EPS of the carry target
  color: number; // COLORS index of the commanded block
  tokens: number[];
  from: { a1: number; a2: number };
  carry: number | null;
  /** Latest predicted gripper sigmoid (0=open → 1=closed); starts open. */
  predGrip: number;
  /** True once the gripper has been seen OPEN during this attempt — the
      rising-edge guard: without a prior open frame the grasp can't arm, so an
      always-closed policy never grabs. */
  sawOpen: boolean;
}

const newEpisode = (color: number, tokens: number[]): Episode => ({
  phase: "reach",
  f: 0,
  near: 0,
  nearColor: -1,
  settle: 0,
  color,
  tokens,
  from: { a1: REST[0], a2: REST[1] },
  carry: null,
  predGrip: 0,
  sawOpen: false,
});

/** A serializable per-frame snapshot the host renders from. */
export interface RolloutFrame {
  a1: number;
  a2: number;
  /** Raw predicted gripper sigmoid (0=open → 1=closed). */
  predGrip: number;
  /** The DRAWN gripper state (1 = closed jaws, 0 = open jaws) — computed here
      via the gripOf logic; the host only renders it. */
  grip: 0 | 1;
  /** COLORS index of the block held at the gripper, or null. */
  carry: number | null;
  phase: RolloutPhase;
  /** The policy's last predicted ABSOLUTE target joint angles (the Action Head
      readout), or null before the first reply / after an episode ends. */
  target: [number, number] | null;
  /** The last prediction's spatial-attention soft-argmax, in [0,1] silhouette
      image coords — the rollout gaze marker. Null when no prediction is live. */
  gaze: [number, number] | null;
  /** Recent end-effector positions in WORKSPACE coords (fk output); the host
      maps these to canvas pixels. Drawn only during reach/carry. */
  trail: { x: number; y: number }[];
  /** False once the episode has ended (arm parked at REST). */
  active: boolean;
}

/** The async policy call the engine steps against (the frozen per-cycle
    snapshot during training, the final weights once converged). */
export type PredictFn = (
  a1: number,
  a2: number,
  tokens: number[],
  layout: Layout,
  carry: number | null
) => Promise<PredictResult | null>;

export class RolloutEngine {
  private ep: Episode | null = null;
  private arm = { a1: REST[0], a2: REST[1] };
  private target: [number, number] | null = null;
  // the spatial-attention readout riding along with the last prediction: where
  // the (frozen) policy is looking (xy in [0,1] silhouette image coords)
  private gaze: [number, number] | null = null;
  private trail: { x: number; y: number }[] = [];
  // in-flight guard + throttle for the async predict round-trip: never queue a
  // second predict while one is outstanding. These persist ACROSS episodes (a
  // reply for a since-ended attempt is dropped by the identity/phase check, not
  // by clearing the guard), matching the old mount-level refs.
  private predInFlight = false;
  private lastPred = 0;

  get hasEpisode(): boolean {
    return this.ep !== null;
  }

  get phase(): RolloutPhase | null {
    return this.ep?.phase ?? null;
  }

  /** Start a fresh attempt (new scene command). Leaves the prediction throttle
      alone — the first predict times off the previous attempt's cadence, as
      before. */
  begin(color: number, tokens: number[]): void {
    this.arm = { a1: REST[0], a2: REST[1] };
    this.trail = [];
    this.target = null;
    this.gaze = null;
    this.ep = newEpisode(color, tokens);
  }

  /** Abort/clear: arm parked at REST, no episode. */
  reset(): void {
    this.arm = { a1: REST[0], a2: REST[1] };
    this.trail = [];
    this.target = null;
    this.gaze = null;
    this.ep = null;
  }

  /** The drawn gripper state: closed once a block is in hand (carry/hold);
      during the reach it mirrors the policy's predicted gripper command so the
      viewer sees it close as it settles over the block; failed return, idle, or
      no episode shows it OPEN (a resting gripper is open — including before
      training starts). */
  private gripOf(): 0 | 1 {
    const ep = this.ep;
    if (!ep) return 0;
    if (ep.carry !== null) return 1;
    if (ep.phase === "reach") return ep.predGrip >= GRIP_THRESHOLD ? 1 : 0;
    return 0;
  }

  /** Current snapshot without advancing (paused redraw, between-episode hold). */
  frame(): RolloutFrame {
    const ep = this.ep;
    return {
      a1: this.arm.a1,
      a2: this.arm.a2,
      predGrip: ep?.predGrip ?? 0,
      grip: this.gripOf(),
      carry: ep?.carry ?? null,
      phase: ep?.phase ?? "return",
      target: this.target,
      gaze: this.gaze,
      trail: this.trail,
      active: ep !== null,
    };
  }

  private endEpisode(): void {
    this.arm = { a1: REST[0], a2: REST[1] };
    this.trail = [];
    this.target = null;
    this.gaze = null;
    this.ep = null;
  }

  /**
   * Advance one episode by a frame and return the frame snapshot. Ends by
   * parking the arm at REST (active=false) — the host re-syncs a fresh attempt
   * on the next demo cycle, or waits for the next command once converged.
   */
  step(now: number, layout: Layout, predict: PredictFn): RolloutFrame {
    const ep = this.ep;
    if (!ep) return this.frame();
    const arm = this.arm;

    if (ep.phase === "reach" || ep.phase === "carry") {
      // the model predicts an ABSOLUTE target (refreshed every PREDICT_MS); the
      // step direction is recomputed every FRAME from that target against the
      // arm's actual current pose, so the step naturally shrinks as it closes in
      // (proportional control).
      if (now - this.lastPred > PREDICT_MS && !this.predInFlight) {
        // frozen per-cycle snapshot (training) / final weights (converged) — the
        // arm still re-predicts every PREDICT_MS as it moves (closed loop), just
        // against fixed weights for the whole attempt. The prediction is ASYNC
        // (render + forward pass in the trainer worker); the arm keeps stepping
        // toward its previous target until the reply lands, and a reply for an
        // episode that has since ended (or switched phase — its render carried
        // the wrong carry state) is dropped instead of re-arming a cleared
        // target.
        this.lastPred = now;
        this.predInFlight = true;
        const phaseAtRequest = ep.phase;
        void predict(
          arm.a1,
          arm.a2,
          ep.tokens,
          layout,
          ep.phase === "carry" ? ep.carry : null
        ).then((r) => {
          this.predInFlight = false;
          if (r && this.ep === ep && ep.phase === phaseAtRequest) {
            this.target = r.target;
            // the same forward pass carries the policy's gaze — drawn as a
            // marker on the rollout scene while the phase is live
            this.gaze = r.xy;
            // and the gripper command — the reach gate closes on its rising
            // edge over a block
            ep.predGrip = r.grip;
          }
        });
      }
      const t = this.target;
      if (t) {
        arm.a1 = clamp(
          arm.a1 + clamp(t[0] - arm.a1, -Math.PI, Math.PI) * STEP_GAIN,
          THETA1_RANGE[0],
          THETA1_RANGE[1]
        );
        arm.a2 = clamp(
          arm.a2 + clamp(t[1] - arm.a2, -Math.PI, Math.PI) * STEP_GAIN,
          THETA2_RANGE[0],
          THETA2_RANGE[1]
        );
        const e = fk(arm.a1, arm.a2);
        this.trail.push({ x: e.ex, y: e.ey });
        if (this.trail.length > TRAIL_LEN) this.trail.shift();
      }

      if (ep.phase === "reach") {
        // LEARNED grasp gate. Contact = the effector fully over SOME block — not
        // just the commanded one — so a wrong-side reach visibly acts on the
        // wrong block instead of hovering until the timeout. The grasp only
        // fires on the RISING EDGE of the predicted gripper close while over the
        // block (and only after the gripper was seen OPEN during the approach),
        // held NEAR_FRAMES — the same effectorOverBlock predicate the training
        // label uses.
        const closing = ep.predGrip >= GRIP_THRESHOLD;
        let over = -1;
        for (const b of layout) {
          if (effectorOverBlock(arm.a1, arm.a2, b, GRIP_RADIUS)) {
            over = b.color;
            break;
          }
        }
        ep.near =
          over >= 0 && closing && ep.sawOpen && over === ep.nearColor
            ? ep.near + 1
            : over >= 0 && closing && ep.sawOpen
              ? 1
              : 0;
        ep.nearColor = over;
        // record an open frame AFTER the gate read, so the first close frame
        // still counts as a rising edge against a prior open one
        if (!closing) ep.sawOpen = true;
        if (ep.near >= NEAR_FRAMES) {
          // GRASP: the block the gripper closed over attaches to the effector
          // and the carry phase begins
          ep.phase = "carry";
          ep.carry = over;
          ep.settle = 0;
          ep.f = 0;
          // force a fresh prediction — the model must now SEE the block in the
          // gripper (in-flight reach replies are phase-dropped)
          this.target = null;
          this.lastPred = 0;
        } else if (ep.f > REACH_TIMEOUT) {
          ep.phase = "return";
          ep.from = { ...arm };
          ep.carry = null;
          ep.f = 0;
        }
      } else {
        // carry: policy-driven with the block in hand. "Settled" = the arm has
        // effectively arrived at the predicted target (there may be no block to
        // be near — the target is wherever the policy wants to go).
        const settled =
          t !== null &&
          Math.abs(t[0] - arm.a1) < SETTLE_EPS &&
          Math.abs(t[1] - arm.a2) < SETTLE_EPS;
        ep.settle = settled ? ep.settle + 1 : 0;
        if (ep.settle >= NEAR_FRAMES) {
          // hold the block aloft wherever the policy parked it
          ep.phase = "hold";
          ep.f = 0;
        } else if (ep.f > REACH_TIMEOUT) {
          // carry never settled — drop the block and give up (it resets to its
          // floor spot once the episode ends)
          ep.carry = null;
          ep.phase = "return";
          ep.from = { ...arm };
          ep.f = 0;
        }
      }
    } else if (ep.phase === "hold") {
      // motionless beat holding the block aloft, then the lift is done
      if (ep.f > TOP_HOLD) {
        this.endEpisode();
        return this.frame();
      }
    } else {
      const u = ep.f / RETURN_FRAMES;
      arm.a1 = lerp(ep.from.a1, REST[0], ease(u));
      arm.a2 = lerp(ep.from.a2, REST[1], ease(u));
      if (u >= 1) {
        this.endEpisode();
        return this.frame();
      }
    }
    if (this.ep) ep.f++;
    return this.frame();
  }
}
