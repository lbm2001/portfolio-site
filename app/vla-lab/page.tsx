"use client";

// ─────────────────────────────────────────────────────────────────────────
// /vla-lab — headless sweep harness for the VLA hero's trainer. NOT linked
// from anywhere; it exists so a playwright script can run controlled
// training experiments against the real pipeline:
//
//   /vla-lab?colors=4&blocks=3&probe=25&max=600
//
// It runs VLATrainerCore INLINE on the main thread (the core is
// environment-agnostic; the worker indirection buys nothing headless),
// installs the run config from the query string, turns on the per-bucket
// probe telemetry, and mirrors the full trainer state onto
// window.__vlaLab after every batch for the driving script to poll.
//
// `max` soft-caps the run: the lab stops (pause) at that many batches even
// if the core's own convergence hasn't fired, so experiment wall-time is
// bounded independent of CONFIG.trainer.converge.maxBatches.
// ─────────────────────────────────────────────────────────────────────────

import { useEffect, useState } from "react";
import { CONFIG } from "@/lib/vla/config";
import { setRunConfig, type RunConfig } from "@/lib/vla/run-config";

// NOTE: trainer.core (and model.ts underneath) snapshot CONFIG values into
// module constants when they first evaluate — so this page must NOT import
// them statically. The `set` query param mutates CONFIG first, and the
// trainer is dynamic-imported afterwards, so knob-override sweep runs
// (e.g. ?set=model.mapLossWeight:2.5,trainer.carryFrac:0.6) see the
// overridden values.

/** Closed-loop rollout metrics over simulated episodes with the converged
    policy — the dial the open-loop probes lack: probes score held-out Huber
    on random states, but "does the arm actually reach, grasp and seat, and
    how much do its predictions wobble en route" is a property of the
    CLOSED loop (2026-07-08 lesson: a live "sloppy reaching" regression was
    invisible in every probe bucket). */
interface RolloutEval {
  episodes: number;
  graspRate: number;
  /** mean frames from episode start to grasp (successful grasps only). */
  meanGraspFrames: number | null;
  /** mean |Δ target| (rad) between consecutive reach-phase predictions —
      the "sloppy reaching" number; a converged policy should re-predict
      nearly the same target every frame. */
  reachJitter: number | null;
}

declare global {
  interface Window {
    __vlaLab?: {
      status: string;
      batches: number;
      loss: number;
      smoothLoss: number;
      lossHistory: number[];
      probes: unknown[];
      rollout?: RolloutEval | null;
      done: boolean;
    };
  }
}

/** Simulate closed-loop episodes against the trained core — same integrator
    shape as Hero's rollout (proportional step toward the latest prediction,
    re-predict every few frames, proximity snap-grasp, settle-then-release). */
async function closedLoopEval(
  core: import("@/lib/vla/trainer.core").VLATrainerCore,
  episodes: number
): Promise<RolloutEval> {
  const { randomLayout, sampleCommand, blockOfColor } = await import(
    "@/lib/vla/examples"
  );
  const { REST, fk } = await import("@/lib/vla/geometry");
  const R = CONFIG.rollout;
  const PRED_EVERY = 6; // frames between re-predictions ≈ Hero's predictMs
  const MAX_FRAMES = 300; // SwiftShader predicts are ~50-250ms each — keep
  // the worst-case predict count bounded (episodes × MAX_FRAMES/PRED_EVERY)

  let grasps = 0;
  let graspFramesSum = 0;
  let jitterSum = 0;
  let jitterN = 0;

  for (let e = 0; e < episodes; e++) {
    // yield between episodes so the driving script's polling evaluate()
    // calls aren't starved for the whole eval (the frame loop is sync)
    await new Promise((r) => setTimeout(r, 0));
    console.log(`[vla-lab] eval episode ${e + 1}/${episodes}`);
    const layout = randomLayout();
    const cmd = sampleCommand(layout);
    const target = blockOfColor(layout, cmd.color);

    let a1 = REST[0];
    let a2 = REST[1];
    let carry: number | null = null;
    let pred: [number, number] = [a1, a2];
    let prevPred: [number, number] | null = null;
    let near = 0;
    let settle = 0;
    let grasped = false;

    for (let f = 0; f < MAX_FRAMES; f++) {
      if (f % PRED_EVERY === 0) {
        const p = core.predictTarget(a1, a2, cmd.tokens, layout, carry);
        if (!p) return { episodes: 0, graspRate: 0, meanGraspFrames: null, reachJitter: null };
        if (prevPred && carry === null && f > PRED_EVERY)
          // reach-phase prediction wobble, skipping the first transient
          (jitterSum += Math.hypot(p.target[0] - prevPred[0], p.target[1] - prevPred[1])), jitterN++;
        prevPred = pred = p.target;
      }
      a1 += (pred[0] - a1) * R.stepGain;
      a2 += (pred[1] - a2) * R.stepGain;
      const { ex, ey } = fk(a1, a2);

      if (carry === null) {
        const gy = (target.y ?? 0) + target.size / 2;
        if (Math.hypot(ex - target.x, ey - gy) < R.graspEps) {
          if (++near >= R.nearFrames) {
            carry = cmd.color;
            grasped = true;
            grasps++;
            graspFramesSum += f;
            prevPred = null; // carry phase — jitter metric stays reach-only
          }
        } else near = 0;
      } else if (
        Math.abs(pred[0] - a1) < R.settleEps &&
        Math.abs(pred[1] - a2) < R.settleEps
      ) {
        // carry settled — the arm holds the block aloft; episode done
        if (++settle >= 4) break;
      } else settle = 0;
      void grasped;
    }
  }
  return {
    episodes,
    graspRate: grasps / episodes,
    meanGraspFrames: grasps ? graspFramesSum / grasps : null,
    reachJitter: jitterN ? jitterSum / jitterN : null,
  };
}

export default function VLALab() {
  const [line, setLine] = useState("vla-lab: idle");

  useEffect(() => {
    const q = new URLSearchParams(window.location.search);
    const rc: RunConfig = {
      numColors: (Number(q.get("colors")) || 8) as RunConfig["numColors"],
      maxBlocks: (Number(q.get("blocks")) || 4) as RunConfig["maxBlocks"],
    };
    setRunConfig(rc);

    // CONFIG knob overrides, applied BEFORE trainer.core/model evaluate
    // (see the import note above). Numeric values only.
    const set = q.get("set");
    if (set)
      for (const kv of set.split(",")) {
        const [path, val] = kv.split(":");
        const keys = path.split(".");
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        let o: any = CONFIG;
        for (const k of keys.slice(0, -1)) o = o[k];
        o[keys[keys.length - 1]] = Number(val);
      }

    let core: import("@/lib/vla/trainer.core").VLATrainerCore | undefined;
    let cancelled = false;
    (async () => {
      const { VLATrainerCore } = await import("@/lib/vla/trainer.core");
      if (cancelled) return;
      core = new VLATrainerCore();
      core.probeEveryN = Number(q.get("probe")) || 25;
      const maxBatches = Number(q.get("max")) || 0;
      // `max` both soft-caps short runs AND extends past the demo budget —
      // the core's own fallback would otherwise end the run at CONFIG's 450
      if (maxBatches) core.maxBatchesOverride = maxBatches;

      const evalEpisodes = Number(q.get("eval")) || 24;
      let evalStarted = false;
      const publish = () => {
        if (!core) return;
        const done = core.status === "converged" || core.status === "paused";
        window.__vlaLab = {
          status: core.status,
          batches: core.batches,
          loss: core.loss,
          smoothLoss: core.smoothLoss,
          lossHistory: core.lossHistory,
          probes: core.probes,
          rollout: window.__vlaLab?.rollout ?? null,
          done,
        };
        setLine(
          `vla-lab: ${core.status} b=${core.batches} smooth=${core.smoothLoss.toFixed(4)}`
        );
        if (maxBatches && core.batches >= maxBatches && core.status === "training")
          core.pause(); // soft cap — keeps the trained weights inspectable
        // once training ends, score the policy CLOSED-LOOP (grasp/seat/jitter)
        if (done && !evalStarted && core.ready) {
          evalStarted = true;
          closedLoopEval(core, evalEpisodes)
            .then((r) => {
              if (window.__vlaLab) window.__vlaLab.rollout = r;
              setLine((l) => `${l} | rollout ${JSON.stringify(r)}`);
            })
            .catch((err) => {
              console.error("[vla-lab] closed-loop eval failed", err);
              if (window.__vlaLab)
                window.__vlaLab.rollout = {
                  episodes: -1,
                  graspRate: -1,
                  meanGraspFrames: null,
                  reachJitter: null,
                };
            });
        }
      };

      core.start(publish);
    })();
    return () => {
      cancelled = true;
      core?.reset();
    };
  }, []);

  return (
    <main style={{ padding: 32, fontFamily: "monospace" }}>
      <h1>VLA lab</h1>
      <p data-testid="vla-lab-status">{line}</p>
    </main>
  );
}
