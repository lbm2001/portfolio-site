"use client";

import { Fragment, useEffect, useLayoutEffect, useRef, useState } from "react";
import { profile, resumeDownloadName } from "@/lib/content";
import { REST, clamp } from "mini-vla/geometry";
import {
  paintScene,
  paintSilhouette,
  sceneMap,
  type ScenePalette,
} from "mini-vla/scene";
import {
  COLORS,
  DEFAULT_LAYOUT,
  DEFAULT_SENTENCE,
  MAX_SEQ_LEN,
  randomLayout,
  sampleCommand,
  tokenize,
  DEMO_PERIOD_MS,
  demoPose,
  makeDemoPlan,
  type BlockPos,
  type Layout,
  type Sentence,
  type DemoPlan,
} from "mini-vla/task";
import {
  CONFIG,
  DEFAULT_RUN_CONFIG,
  estimateTrainingSeconds,
  setRunConfig,
  type RunConfig,
} from "mini-vla/config";
import { IMG_SIZE } from "mini-vla/model";
import { VLATrainer, type TrainerStatus } from "mini-vla/trainer";
import {
  RolloutEngine,
  type RolloutFrame,
  type RolloutPhase,
} from "mini-vla/rollout";

// Live Vision-Language-Action hero: four pipeline boxes ringed around the
// name (Demonstration left w/ floating prompt above, Vision Encoder top,
// Language Encoder bottom, Rollout right). "Start Training" runs a GENUINE
// TensorFlow.js behavioral-cloning loop — hosted in a Web Worker off the main
// thread (mini-vla's trainer.worker.ts, via the mini-vla/trainer proxy) so
// gradient steps never fight this component's 60fps rAF loop — against an
// analytical-IK expert on pick-up commands (2-4 blocks from a 2/4/8-color
// palette; mini-vla/config): each scene's slot-grammar command names
// one block to pick up. The displayed demonstration swaps every cycle while
// thousands of examples train invisibly; the Rollout runs policy-driven
// episodes — the arm approaches with an open gripper and the LEARNED gripper
// action closes it over the block (attaching it), then the carry phase (lift
// it home) follows the policy's predictions. Once the action
// loss converges, training stops and the Rollout box becomes interactive:
// type your own command, run it, reshuffle the blocks.

const ACCENT = "#e12d1a"; // = --red; canvases can't read CSS vars cheaply

// The demo/rollout scenes' cosmetic look is owned HERE (host-side) and passed
// into paintScene as a palette. These match the renderer's defaults, so the
// visible output is unchanged — the point is that the look now lives with the
// host, not the model. (paintSilhouette takes no palette: the model's-eye view
// versions with the model.)
const SCENE_PALETTE: ScenePalette = {
  floor: "#e6e6e6",
  pedestal: "#2b2b2b",
  link: "#8a8a8a",
  joint: "#fff",
  effectorOpen: "#fff",
  effectorOpenEdge: "#6f6f6f",
  effectorClosed: "#6f6f6f",
};

// The rollout state machine — phases (reach → carry → hold → return), the
// learned-grasp gate, carry attachment, and stepping toward the last async-
// predicted target — lives in mini-vla/rollout (RolloutEngine). Hero only
// drives it and paints the RolloutFrames it returns. LANG_MS is the shared
// throttle for the language + vision-gaze readouts (CONFIG.rollout).
const LANG_MS = CONFIG.rollout.langMs;

const fmtAngle = (v: number) => `${v >= 0 ? "+" : "−"}${Math.abs(v).toFixed(2)}`;

/** Resize a canvas to its CSS box at devicePixelRatio; returns a cleared ctx. */
function fitCanvas(c: HTMLCanvasElement, fallbackW = 190, fallbackH = 186) {
  const dpr = window.devicePixelRatio || 1;
  const W = c.offsetWidth || fallbackW;
  const H = c.offsetHeight || fallbackH;
  if (c.width !== Math.round(W * dpr)) c.width = Math.round(W * dpr);
  if (c.height !== Math.round(H * dpr)) c.height = Math.round(H * dpr);
  const ctx = c.getContext("2d")!;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, W, H);
  return { ctx, W, H };
}

const SIL_RENDER = CONFIG.rollout.silRender; // silhouette render size before the imgSize downsample

export default function Hero() {
  const stageRef = useRef<HTMLElement>(null);
  const flowRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLDivElement>(null);
  const promptRef = useRef<HTMLDivElement>(null);
  const visionCardRef = useRef<HTMLDivElement>(null);
  const langCardRef = useRef<HTMLDivElement>(null);
  const actionCardRef = useRef<HTMLDivElement>(null);
  const outputCardRef = useRef<HTMLDivElement>(null);
  const demoRef = useRef<HTMLCanvasElement>(null);
  const visionRef = useRef<HTMLCanvasElement>(null);
  const armRef = useRef<HTMLCanvasElement>(null);
  const lossRef = useRef<HTMLCanvasElement>(null);
  const actionValsRef = useRef<HTMLDivElement>(null);

  const trainerRef = useRef<VLATrainer | null>(null);
  const statusRef = useRef<TrainerStatus>("idle");
  // the DISPLAY pose for the non-episode states (idle / converged-waiting sway,
  // not-ready rest). The rollout's own arm pose during an episode lives inside
  // the RolloutEngine and comes back on each RolloutFrame.
  const armState = useRef({ a1: REST[0], a2: REST[1] });
  // the whole policy-rollout state machine (episode, arm, target, gaze, trail,
  // predict throttle/in-flight guard) — Hero drives it and paints its frames.
  const engineRef = useRef<RolloutEngine | null>(null);
  if (engineRef.current === null) engineRef.current = new RolloutEngine();
  // live-model attention over the DEMONSTRATION state — the Vision Encoder
  // panel's heatmap (ATTN_GRID² weights, peak-normalized by the trainer)
  const visGazeRef = useRef<number[] | null>(null);
  const lastHudRef = useRef(0);
  const lastLangRef = useRef(0);
  const lastVisGazeRef = useRef(0);
  // in-flight guards for the async trainer-worker round-trips: never queue a
  // second language-readout / gaze request while one is outstanding. On a slow
  // GPU a request can take longer than its throttle interval — an unbounded
  // FIFO backlog in the worker would starve training entirely. (The rollout's
  // own predict guard lives inside the RolloutEngine.)
  const langInFlightRef = useRef(false);
  const visGazeInFlightRef = useRef(false);
  // paused-duration accounting so the demo cycle resumes exactly where it
  // left off instead of jumping ahead by the real wall-clock pause length
  const pausedAccumRef = useRef(0);
  const pauseStartRef = useRef<number | null>(null);
  // wall-clock origin of the demonstration trajectory clock. Anchored to the
  // click (not the huge performance.now uptime) so the first cycle begins at
  // t=0 — the resting default demonstration already on screen — instead of
  // snapping into a random mid-reach. Offset +500ms so the arm holds that
  // initial pose for half a second before it starts moving (no harsh jump on
  // click). Pause duration is folded in via pausedAccumRef, not this.
  const trainStartRef = useRef(0);

  // demonstration state — a fresh layout/sentence/trajectory every cycle
  const demoLayoutRef = useRef(DEFAULT_LAYOUT);
  const demoSentenceRef = useRef<Sentence>(DEFAULT_SENTENCE);
  const demoPlanRef = useRef<DemoPlan | null>(null);
  const lastCycleRef = useRef(-1);
  const demoPoseRef = useRef({
    a1: REST[0],
    a2: REST[1],
    carry: null as number | null,
    grip: 0 as 0 | 1,
  });

  // rollout state
  const rolloutLayoutRef = useRef(DEFAULT_LAYOUT);
  const activeTokensRef = useRef<number[]>(DEFAULT_SENTENCE.tokens);
  const userSentenceRef = useRef<Sentence | null>(null);
  // which demo cycle the current training rollout episode belongs to — the
  // rollout re-syncs to the demonstration (same scene + command, fresh
  // attempt) whenever this falls behind the demo's cycle counter
  const rolloutCycleRef = useRef(-1);

  // offscreen canvases: the literal silhouette pipeline the model sees
  const silRef = useRef<HTMLCanvasElement | null>(null);
  const silThumbRef = useRef<HTMLCanvasElement | null>(null);

  const [status, setStatus] = useState<TrainerStatus>("idle");
  const [hud, setHud] = useState({ lossText: "—", samples: 0, batches: 0 });
  // the ⚙ run config (palette / density / task set) picked before training.
  // Mirrored into a ref because the rAF-loop closures (demo-cycle command
  // sampling) mount once and would otherwise capture the initial state.
  const [runCfg, setRunCfgState] = useState<RunConfig>(DEFAULT_RUN_CONFIG);
  const runCfgRef = useRef<RunConfig>(DEFAULT_RUN_CONFIG);
  const setRunCfg = (next: RunConfig) => {
    runCfgRef.current = next;
    setRunCfgState(next);
  };
  const [cfgOpen, setCfgOpen] = useState(false);
  // demonstrations the CURRENTLY rolled-out policy has seen: the sample count
  // frozen into the per-cycle snapshot during training, or the final count
  // once converged (see drawArm's snapshot boundary + onUpdate).
  const [rolloutSamples, setRolloutSamples] = useState(0);
  const [demoSentence, setDemoSentence] = useState<Sentence>(DEFAULT_SENTENCE);
  const [userSentence, setUserSentence] = useState<Sentence | null>(null);
  const [tryText, setTryText] = useState("");
  const chipRowRef = useRef<HTMLDivElement | null>(null);
  const [tokenBars, setTokenBars] = useState<number[]>([]);
  const [decoded, setDecoded] = useState<{
    name: string;
    hex: string;
    prob: number;
  } | null>(null);
  const [tryNote, setTryNote] = useState<string | null>(null);
  // Vision Encoder panel: flip to the exact inverted tensor the CNN receives.
  // Hover handles it on desktop (CSS); this drives the tap toggle on touch.
  const [modelView, setModelView] = useState(false);

  const setStatusBoth = (s: TrainerStatus) => {
    statusRef.current = s;
    setStatus(s);
  };

  // ---- the single rAF loop: wires + all four canvases, every frame ----
  useEffect(() => {
    const reduced = window.matchMedia("(prefers-reduced-motion: reduce)");

    const layoutWires = () => {
      const stage = stageRef.current;
      const flow = flowRef.current;
      if (
        !stage ||
        !flow ||
        !inputRef.current ||
        !promptRef.current ||
        !visionCardRef.current ||
        !langCardRef.current ||
        !actionCardRef.current ||
        !outputCardRef.current
      )
        return;
      const R = stage.getBoundingClientRect();
      if (R.width === 0 || R.height === 0) return;
      // Card box in PX relative to the stage (offset-path lives in the flow
      // overlay's own px coordinate space, no viewBox stretching).
      const m = (el: Element) => {
        const c = el.getBoundingClientRect();
        return {
          cx: c.left - R.left + c.width / 2,
          cy: c.top - R.top + c.height / 2,
          left: c.left - R.left,
          right: c.right - R.left,
          top: c.top - R.top,
          bottom: c.bottom - R.top,
        };
      };
      const I = m(inputRef.current);
      const P = m(promptRef.current);
      const V = m(visionCardRef.current);
      const L = m(langCardRef.current);
      const A = m(actionCardRef.current);
      const O = m(outputCardRef.current);
      const r = (n: number) => n.toFixed(1);
      // No drawn connectors any more — a payload token glides across each gap,
      // so the "connection" is carried by motion. Each hop is a single smooth
      // cubic Bézier between two facing card edges (horizontal-biased controls
      // → the token leaves and arrives along the horizontal, arcing gently in
      // between; nearly straight for the short Action→Rollout hop). Five hops:
      //   p1 Demonstration → Vision      p3 Vision   → Action Head (top)
      //   p2 Prompt        → Language    p4 Language → Action Head (bottom)
      //                                  p5 Action Head → Rollout
      const arc = (
        sx: number,
        sy: number,
        ex: number,
        ey: number
      ) => {
        const dx = ex - sx;
        return `path("M ${r(sx)} ${r(sy)} C ${r(sx + dx * 0.42)} ${r(sy)}, ${r(
          sx + dx * 0.58
        )} ${r(ey)}, ${r(ex)} ${r(ey)}")`;
      };
      const paths: Record<string, string> = {
        p1: arc(I.right, I.cy, V.cx, V.bottom),
        p2: arc(P.right, P.cy, L.cx, L.top),
        p3: arc(V.right, V.cy, A.cx, A.top),
        p4: arc(L.right, L.cy, A.cx, A.bottom),
        p5: arc(A.right, A.cy, O.left, O.cy),
      };
      flow.querySelectorAll<HTMLElement>("[data-flow]").forEach((el) => {
        const d = paths[el.dataset.flow ?? ""];
        if (d) el.style.offsetPath = d;
      });
    };

    // the Action Head's live output — the policy's current predicted target
    // joint angles plus the gripper state (open/closed). null angles → em
    // dashes; grip undefined → em dash, 0 → open, 1 → closed.
    const setActionVals = (
      t: [number, number] | null,
      grip?: 0 | 1 | undefined
    ) => {
      const el = actionValsRef.current;
      if (!el || el.children.length < 3) return;
      const v0 = el.children[0].lastElementChild;
      const v1 = el.children[1].lastElementChild;
      const v2 = el.children[2].lastElementChild;
      if (v0) v0.textContent = t ? fmtAngle(t[0]) : "—";
      if (v1) v1.textContent = t ? fmtAngle(t[1]) : "—";
      if (v2)
        v2.textContent = grip === 1 ? "closed" : grip === 0 ? "open" : "—";
    };

    // small idle sway around the rest pose — "there is something here"
    const wiggle = (now: number, p1: number, p2: number): [number, number] =>
      reduced.matches
        ? REST
        : [
            REST[0] + 0.05 * Math.sin(now * 0.0011 + p1),
            0.07 * Math.sin(now * 0.0008 + p2),
          ];

    const drawDemo = (now: number) => {
      const c = demoRef.current;
      if (!c) return;
      const { ctx, W, H } = fitCanvas(c);
      const st = statusRef.current;

      if (st === "paused") {
        // frozen mid-cycle: redraw the last computed pose, advance nothing
        const p = demoPoseRef.current;
        paintScene(ctx, W, H, {
          a1: p.a1,
          a2: p.a2,
          layout: demoLayoutRef.current,
          accent: ACCENT,
          carry: p.carry,
          grip: p.grip,
          palette: SCENE_PALETTE,
        });
        return;
      }

      let a1: number;
      let a2: number;
      let carry: number | null = null;
      let grip: 0 | 1 | undefined;

      if (st === "training" || st === "loading") {
        // clock measured from the click (trainStartRef), minus accumulated
        // paused time so the cycle resumes exactly where it left off
        const effectiveNow = now - trainStartRef.current - pausedAccumRef.current;
        if (effectiveNow < 0) {
          // the half-second hold right after the click: sit on the initial
          // demonstration (t=0 → resting pose over the default layout) so the
          // pipeline eases in instead of the arm snapping into motion
          if (!demoPlanRef.current)
            demoPlanRef.current = makeDemoPlan(
              demoLayoutRef.current,
              demoSentenceRef.current
            );
          const pose = demoPose(demoPlanRef.current, 0);
          demoPoseRef.current = {
            a1: pose.a1,
            a2: pose.a2,
            carry: pose.carry,
            grip: pose.grip,
          };
          paintScene(ctx, W, H, {
            a1: pose.a1,
            a2: pose.a2,
            layout: demoLayoutRef.current,
            accent: ACCENT,
            carry: pose.carry,
            grip: pose.grip,
            palette: SCENE_PALETTE,
          });
          return;
        }
        const cycle = Math.floor(effectiveNow / DEMO_PERIOD_MS);
        if (cycle !== lastCycleRef.current || !demoPlanRef.current) {
          const first = lastCycleRef.current === -1;
          lastCycleRef.current = cycle;
          if (!first) {
            demoLayoutRef.current = randomLayout();
            // a pick-up command naming one of the scene's own blocks
            const s = sampleCommand(demoLayoutRef.current);
            demoSentenceRef.current = s;
            setDemoSentence(s);
            // the language panel follows the demo until a user command locks it
            if (!userSentenceRef.current) activeTokensRef.current = s.tokens;
          }
          demoPlanRef.current = makeDemoPlan(
            demoLayoutRef.current,
            demoSentenceRef.current
          );
        }

        const t = reduced.matches
          ? 0.2
          : (effectiveNow % DEMO_PERIOD_MS) / DEMO_PERIOD_MS;
        const plan = demoPlanRef.current;
        const pose = demoPose(plan, t);
        a1 = pose.a1;
        a2 = pose.a2;
        carry = pose.carry;
        grip = pose.grip;
      } else {
        // idle OR converged: the demonstrations stop once the policy is
        // trained — the arm returns to the resting sway (the CSS also fades
        // the box back to its dormant, pre-training transparency). A resting
        // gripper is OPEN (including before training starts).
        [a1, a2] = wiggle(now, 0, 1.7);
        grip = 0;
      }

      demoPoseRef.current = { a1, a2, carry, grip: grip ?? 0 };
      paintScene(ctx, W, H, {
        a1,
        a2,
        layout: demoLayoutRef.current,
        accent: ACCENT,
        carry,
        grip: grip ?? 0,
        palette: SCENE_PALETTE,
      });
    };

    // The policy's gaze on the Rollout scene: the soft-argmax of the spatial
    // attention map that rode along with the last prediction. xy is in the
    // SILHOUETTE's [0,1] image coords — invert that renderer's sceneMap back
    // to workspace units, then forward-map into this canvas. Drawn only while
    // a policy-driven phase is live (that's when the prediction is current).
    const drawGaze = (
      ctx: CanvasRenderingContext2D,
      W: number,
      H: number,
      gaze: [number, number] | null,
      phase: RolloutPhase | null
    ) => {
      if (!gaze || (phase !== "reach" && phase !== "carry")) return;
      const sc = CONFIG.render.sceneScale;
      const wx = 0.5 + (gaze[0] - 0.5) / sc;
      const wy = (CONFIG.render.floorY - gaze[1]) / sc;
      const m = sceneMap(W, H);
      const px = m.X(wx);
      const py = m.Y(wy);
      const rad = 14;
      const grad = ctx.createRadialGradient(px, py, 0, px, py, rad);
      grad.addColorStop(0, "rgba(225,45,26,0.30)");
      grad.addColorStop(1, "rgba(225,45,26,0)");
      ctx.fillStyle = grad;
      ctx.beginPath();
      ctx.arc(px, py, rad, 0, 7);
      ctx.fill();
      ctx.strokeStyle = "rgba(225,45,26,0.75)";
      ctx.lineWidth = 1.2;
      ctx.beginPath();
      ctx.arc(px, py, 4.5, 0, 7);
      ctx.stroke();
    };

    // The async policy call the engine steps against: the FROZEN per-cycle
    // snapshot (training) / final weights (converged). The engine keeps stepping
    // toward its last target until each reply lands (see RolloutEngine.step).
    const predictFrozen = (
      a1: number,
      a2: number,
      tokens: number[],
      layout: Layout,
      carry: number | null
    ) =>
      trainerRef.current
        ? trainerRef.current.predictFrozenTarget(a1, a2, tokens, layout, carry)
        : Promise.resolve(null);

    // The engine keeps its trail in workspace effector coords (renderer-neutral);
    // map to canvas px here — identical to the old effectorPx pushes, since
    // effectorPx === sceneMap ∘ fk.
    const trailToPx = (
      trail: { x: number; y: number }[],
      W: number,
      H: number
    ) => {
      const m = sceneMap(W, H);
      return trail.map((p) => ({ x: m.X(p.x), y: m.Y(p.y) }));
    };

    // rollout: during training the episode runs in LOCKSTEP with the
    // demonstration — same scene + command, restarted each demo cycle — so
    // the two can be compared side by side; converged runs user commands;
    // paused freezes it.
    const drawArm = (now: number) => {
      const c = armRef.current;
      if (!c) return;
      const { ctx, W, H } = fitCanvas(c);
      const trainer = trainerRef.current;
      const arm = armState.current;
      const st = statusRef.current;
      const engine = engineRef.current!;

      if (st === "paused") {
        // frozen: redraw the engine's current pose, advance nothing (matches
        // the demo). The rollout arm lives in the engine now, so the frozen pose
        // is engine.frame() — NOT the wiggling armState.
        const f = engine.frame();
        paintScene(ctx, W, H, {
          a1: f.a1,
          a2: f.a2,
          layout: rolloutLayoutRef.current,
          accent: ACCENT,
          carry: f.carry,
          grip: f.grip,
          palette: SCENE_PALETTE,
        });
        drawGaze(ctx, W, H, f.gaze, f.phase);
        setActionVals(f.target, f.grip);
        return;
      }

      // f = the engine frame when an episode is (or was) live this frame; null
      // for the non-episode display states (idle / converged-waiting sway,
      // not-ready rest) that paint the wiggling armState with an empty rollout.
      let f: RolloutFrame | null = null;

      if (st === "idle") {
        [arm.a1, arm.a2] = wiggle(now, 2.6, 4.1);
      } else if (trainer?.ready) {
        if (st === "converged") {
          if (engine.hasEpisode)
            f = engine.step(now, rolloutLayoutRef.current, predictFrozen);
          else [arm.a1, arm.a2] = wiggle(now, 2.6, 4.1); // waiting for command
        } else {
          // training: a fresh synced attempt at every new demonstration cycle,
          // run against a policy snapshot frozen at this boundary so the whole
          // attempt reflects one policy generation (not a live-drifting target)
          if (rolloutCycleRef.current !== lastCycleRef.current) {
            rolloutCycleRef.current = lastCycleRef.current;
            // per-block CLONE of the demo scene, not a shared reference: the
            // viewer can drag the rollout's blocks once converged, so sharing
            // objects would leak one box's positions into the other
            rolloutLayoutRef.current = demoLayoutRef.current.map((b) => ({ ...b }));
            trainer.snapshotPolicy();
            // the snapshot freezes the policy at THIS many seen demonstrations —
            // surface it on the Rollout so the attempt is read as "this is what
            // N examples of training buys you" (updates once per demo cycle)
            setRolloutSamples(trainer.samples);
            engine.begin(
              demoSentenceRef.current.color,
              demoSentenceRef.current.tokens
            );
          }
          // step the live attempt, or hold at REST (engine.frame()) between the
          // end of one attempt and the next cycle's re-sync
          f = engine.hasEpisode
            ? engine.step(now, rolloutLayoutRef.current, predictFrozen)
            : engine.frame();
        }
      } else {
        arm.a1 = REST[0];
        arm.a2 = REST[1];
      }

      // single paint: from the engine frame when an episode is live, else the
      // wiggling/rest display pose with an empty rollout state.
      paintScene(ctx, W, H, {
        a1: f ? f.a1 : arm.a1,
        a2: f ? f.a2 : arm.a2,
        layout: rolloutLayoutRef.current,
        accent: ACCENT,
        trail:
          f && (f.phase === "reach" || f.phase === "carry")
            ? trailToPx(f.trail, W, H)
            : null,
        lossNorm: trainer?.lossNorm() ?? 1,
        carry: f ? f.carry : null,
        grip: f ? f.grip : 0,
        palette: SCENE_PALETTE,
      });
      drawGaze(ctx, W, H, f ? f.gaze : null, f ? f.phase : null);
      setActionVals(st === "idle" ? null : f ? f.target : null, f ? f.grip : 0);
    };

    const drawVision = (now: number) => {
      const c = visionRef.current;
      if (!c) return;
      // once trained the encoder goes dormant: stop feeding it the demo
      // silhouette (the CSS fades the canvas back to transparent). The last
      // frame stays underneath the fade-out, which is fine — it's hidden.
      if (statusRef.current === "converged") return;
      const { ctx, W } = fitCanvas(c, 176, 176);

      // offscreen silhouette pipeline (the literal model input view)
      if (!silRef.current) {
        silRef.current = document.createElement("canvas");
        silRef.current.width = SIL_RENDER;
        silRef.current.height = SIL_RENDER;
        silThumbRef.current = document.createElement("canvas");
        silThumbRef.current.width = IMG_SIZE;
        silThumbRef.current.height = IMG_SIZE;
      }
      const sil = silRef.current;
      const thumb = silThumbRef.current!;
      const p = demoPoseRef.current;
      paintSilhouette(
        sil.getContext("2d")!,
        SIL_RENDER,
        p.a1,
        p.a2,
        demoLayoutRef.current,
        p.carry
      );
      const tctx = thumb.getContext("2d", { willReadFrequently: true })!;
      tctx.imageSmoothingEnabled = true;
      tctx.clearRect(0, 0, IMG_SIZE, IMG_SIZE);
      tctx.drawImage(sil, 0, 0, SIL_RENDER, SIL_RENDER, 0, 0, IMG_SIZE, IMG_SIZE);

      // 32x32 input, blown up pixelated
      ctx.imageSmoothingEnabled = false;
      ctx.drawImage(thumb, 0, 0, IMG_SIZE, IMG_SIZE, 0, 0, W, W);

      // "where the model looks": the live model's spatial attention over THIS
      // demonstration state, as a per-cell alpha overlay (peak-normalized by
      // the trainer). Early in training it's a diffuse smear; it visibly
      // sharpens onto the commanded block as the policy learns to bind the
      // command to the scene.
      const gaze = visGazeRef.current;
      if (gaze && statusRef.current !== "idle") {
        const G = Math.round(Math.sqrt(gaze.length));
        const cell = W / G;
        for (let i = 0; i < G; i++)
          for (let j = 0; j < G; j++) {
            const a = gaze[i * G + j];
            if (a < 0.04) continue;
            ctx.fillStyle = `rgba(225,45,26,${(0.45 * a).toFixed(3)})`;
            ctx.fillRect(j * cell, i * cell, cell + 0.5, cell + 0.5);
          }
      }

      ctx.strokeStyle = "rgba(0,0,0,.12)";
      ctx.lineWidth = 1;
      ctx.strokeRect(0.5, 0.5, W - 1, W - 1);

      // refresh the gaze on a throttle (same cadence as the language panel),
      // against the demonstration's current pose + command + carry state, on
      // the LIVE model — this is training-progress chrome, like the decoded-
      // command readout. One request in flight, ever (see the in-flight
      // guards note above).
      const trainer = trainerRef.current;
      if (
        trainer?.ready &&
        statusRef.current === "training" &&
        now - lastVisGazeRef.current > LANG_MS &&
        !visGazeInFlightRef.current
      ) {
        lastVisGazeRef.current = now;
        visGazeInFlightRef.current = true;
        const p = demoPoseRef.current;
        void trainer
          .predictLive(
            p.a1,
            p.a2,
            demoSentenceRef.current.tokens,
            demoLayoutRef.current,
            p.carry
          )
          .then((r) => {
            visGazeInFlightRef.current = false;
            if (r) visGazeRef.current = r.attn;
          });
      }
    };

    const drawLossCurve = () => {
      const c = lossRef.current;
      if (!c) return;
      const { ctx, W, H } = fitCanvas(c, 300, 34);
      const raw = trainerRef.current?.lossHistory ?? [];
      const pad = 3;

      ctx.strokeStyle = "#efefef";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(0, H - pad);
      ctx.lineTo(W, H - pad);
      ctx.stroke();
      if (raw.length < 2) return;

      // Smooth out the batch-to-batch noise so the curve shows the trend, not
      // the jitter: a trailing rolling mean over the last SMOOTH_WINDOW batches
      // (prefix sum → O(n)). The window shrinks near the start so the curve
      // still begins at the first real loss.
      const SMOOTH_WINDOW = 30;
      const prefix = [0];
      for (let i = 0; i < raw.length; i++) prefix.push(prefix[i] + raw[i]);
      const hist = raw.map((_, i) => {
        const lo = Math.max(0, i - SMOOTH_WINDOW + 1);
        return (prefix[i + 1] - prefix[lo]) / (i + 1 - lo);
      });

      const maxLoss = Math.max(trainerRef.current?.initialLoss ?? 0, ...hist);
      const n = hist.length;
      const x = (i: number) => (i / (n - 1)) * W;
      const y = (v: number) =>
        pad + (1 - Math.min(1, v / maxLoss)) * (H - pad * 2);

      ctx.beginPath();
      ctx.moveTo(0, y(hist[0]));
      hist.forEach((v, i) => ctx.lineTo(x(i), y(v)));
      ctx.lineTo(W, H - pad);
      ctx.lineTo(0, H - pad);
      ctx.closePath();
      ctx.fillStyle = ACCENT;
      ctx.globalAlpha = 0.08;
      ctx.fill();
      ctx.globalAlpha = 1;

      ctx.beginPath();
      ctx.moveTo(0, y(hist[0]));
      hist.forEach((v, i) => ctx.lineTo(x(i), y(v)));
      ctx.strokeStyle = ACCENT;
      ctx.lineWidth = 1.6;
      ctx.lineJoin = "round";
      ctx.lineCap = "round";
      ctx.stroke();

      ctx.fillStyle = ACCENT;
      ctx.beginPath();
      ctx.arc(x(n - 1), y(hist[n - 1]), 2.6, 0, 7);
      ctx.fill();
    };

    // real language-encoder readouts: decoded command (color) + each token's
    // live ATTENTION weight (see attentionWeights in trainer.core.ts — the
    // exact masked-softmax weights the attention-pooling layer uses,
    // recomputed from the linear scorer's weights, not an approximation).
    // Both are async round-trips to the trainer worker; LANG_MS throttling
    // means at most one pair is ever in flight, so replies apply in order.
    const langViz = (now: number) => {
      const trainer = trainerRef.current;
      if (!trainer?.ready || now - lastLangRef.current < LANG_MS) return;
      if (langInFlightRef.current) return;
      lastLangRef.current = now;
      langInFlightRef.current = true;
      const tokens = activeTokensRef.current;
      const decodeP = trainer.decodeCommand(tokens).then((d) => {
        if (!d) return;
        setDecoded({
          name: COLORS[d.color].name,
          hex: COLORS[d.color].hex,
          prob: d.colorProb,
        });
      });
      const attnP = trainer.attentionWeights(tokens).then((bars) => {
        if (bars) setTokenBars(bars);
      });
      void Promise.all([decodeP, attnP]).then(() => {
        langInFlightRef.current = false;
      });
    };

    let raf = 0;
    const loop = (now: number) => {
      layoutWires();
      drawDemo(now);
      drawArm(now);
      drawVision(now);
      drawLossCurve();
      langViz(now);
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);
    window.addEventListener("resize", layoutWires);
    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener("resize", layoutWires);
      trainerRef.current?.destroy(); // reset + terminate the trainer worker
    };
  }, []);

  // ---- controls ----
  const onUpdate = () => {
    const trainer = trainerRef.current!;
    if (trainer.status !== statusRef.current) {
      setStatusBoth(trainer.status);
      if (trainer.status === "converged") {
        // auto-episodes end; the rollout waits for the user's command
        engineRef.current!.reset();
        visGazeRef.current = null;
        armState.current = { a1: REST[0], a2: REST[1] };
        // the interactive policy is the final model — show the full training
        // budget it saw, not the last per-cycle snapshot count
        setRolloutSamples(trainer.samples);
      }
    }
    const now = performance.now();
    if (now - lastHudRef.current > 150 && !Number.isNaN(trainer.loss)) {
      lastHudRef.current = now;
      setHud({
        lossText: trainer.loss.toFixed(3),
        samples: trainer.samples,
        batches: trainer.batches,
      });
    }
  };

  const onPrimary = () => {
    const trainer = (trainerRef.current ??= new VLATrainer());
    if (trainer.status === "training") {
      trainer.pause();
      pauseStartRef.current = performance.now();
      setStatusBoth("paused");
    } else if (trainer.status === "paused") {
      if (pauseStartRef.current !== null) {
        pausedAccumRef.current += performance.now() - pauseStartRef.current;
        pauseStartRef.current = null;
      }
      trainer.resume();
      setStatusBoth("training");
    } else if (trainer.status === "idle") {
      engineRef.current!.reset();
      rolloutCycleRef.current = -1;
      visGazeRef.current = null;
      armState.current = { a1: REST[0], a2: REST[1] };
      userSentenceRef.current = null;
      pausedAccumRef.current = 0;
      pauseStartRef.current = null;
      // half-second hold on the initial demonstration before the arm moves
      trainStartRef.current = performance.now() + 500;
      setUserSentence(null);
      setDecoded(null);
      setTokenBars([]);
      setRolloutSamples(0);
      setCfgOpen(false);
      // install the ⚙ run config on this thread's samplers before anything
      // draws a command (trainer.start also ships it to the worker)
      const cfg = runCfgRef.current;
      setRunConfig(cfg);
      // per-block clone of the default scene — the rollout's blocks are
      // draggable once converged, so the module-level DEFAULT_LAYOUT must
      // stay pristine
      demoLayoutRef.current = DEFAULT_LAYOUT.map((b) => ({ ...b }));
      rolloutLayoutRef.current = DEFAULT_LAYOUT.map((b) => ({ ...b }));
      void trainer.start(onUpdate, cfg);
    }
  };

  const onReset = () => {
    trainerRef.current?.reset();
    setStatusBoth("idle");
    engineRef.current!.reset();
    rolloutCycleRef.current = -1;
    visGazeRef.current = null;
    armState.current = { a1: REST[0], a2: REST[1] };
    // clones — the rollout's blocks are draggable; DEFAULT_LAYOUT stays pristine
    demoLayoutRef.current = DEFAULT_LAYOUT.map((b) => ({ ...b }));
    demoSentenceRef.current = DEFAULT_SENTENCE;
    demoPlanRef.current = null;
    lastCycleRef.current = -1;
    rolloutLayoutRef.current = DEFAULT_LAYOUT.map((b) => ({ ...b }));
    activeTokensRef.current = DEFAULT_SENTENCE.tokens;
    userSentenceRef.current = null;
    pausedAccumRef.current = 0;
    pauseStartRef.current = null;
    trainStartRef.current = 0;
    setDemoSentence(DEFAULT_SENTENCE);
    setUserSentence(null);
    setTryText("");
    setTryNote(null);
    setTokenBars([]);
    setDecoded(null);
    setRolloutSamples(0);
    setHud({ lossText: "—", samples: 0, batches: 0 });
  };

  /** "Try it" mode: run the user's own sentence through the trained policy.
      The color head decodes which block to pick up; the motion itself is
      entirely the policy's. */
  const runCommand = async () => {
    const trainer = trainerRef.current;
    if (!trainer?.ready || statusRef.current !== "converged") return;
    const text = tryText.trim();
    if (!text) return;
    const tokens = tokenize(text);
    const d = await trainer.decodeCommand(tokens); // worker round-trip (~ms)
    if (!d || statusRef.current !== "converged") return;
    const words = text
      .toLowerCase()
      .replace(/[^a-z ]/g, "")
      .split(" ")
      .filter(Boolean)
      .slice(0, MAX_SEQ_LEN);
    const sentence: Sentence = {
      color: d.color,
      text,
      words,
      tokens,
    };
    userSentenceRef.current = sentence;
    setUserSentence(sentence);
    activeTokensRef.current = tokens;
    setDecoded({
      name: COLORS[d.color].name,
      hex: COLORS[d.color].hex,
      prob: d.colorProb,
    });
    setTryNote(null);
    armState.current = { a1: REST[0], a2: REST[1] };
    engineRef.current!.begin(d.color, tokens);
  };

  const randomizeBlocks = () => {
    rolloutLayoutRef.current = randomLayout();
    // abort any in-flight episode so the arm re-plans against the new scene
    engineRef.current!.reset();
    armState.current = { a1: REST[0], a2: REST[1] };
    setTryNote(null);
  };

  // ---- drag the Rollout blocks (converged / "try it" mode only) ----
  // Once the policy is trained the viewer can reposition either block by
  // dragging it, constrained to that block's cleanly-reachable side band
  // (CONFIG.task.placeLeft/Right — the centre is a near-singular dead zone).
  // The block object is mutated in place, so the rAF loop redraws it moving
  // live; any in-flight attempt is cancelled so the arm re-plans on the next
  // Run against wherever the blocks now sit.
  const dragRef = useRef<{ block: BlockPos; band: [number, number] } | null>(null);

  // Invert sceneMap's X: canvas CSS-pixel x → workspace x. (X = W/2 + (x-0.5)*S.)
  const canvasX = (c: HTMLCanvasElement, clientX: number) => {
    const rect = c.getBoundingClientRect();
    const S = CONFIG.render.sceneScale * rect.height;
    return 0.5 + (clientX - rect.left - rect.width * 0.5) / S;
  };

  // The block under a pointer, if any (a few px of touch padding around the box).
  const blockAt = (c: HTMLCanvasElement, clientX: number, clientY: number) => {
    const rect = c.getBoundingClientRect();
    const S = CONFIG.render.sceneScale * rect.height;
    const floorY = CONFIG.render.floorY * rect.height;
    const px = clientX - rect.left;
    const py = clientY - rect.top;
    const pad = 6;
    for (const b of rolloutLayoutRef.current) {
      const cx = rect.width * 0.5 + (b.x - 0.5) * S;
      const half = (b.size * S) / 2;
      const bottom = floorY - (b.y ?? 0) * S; // rest height (normally floor)
      if (
        px >= cx - half - pad &&
        px <= cx + half + pad &&
        py >= bottom - b.size * S - pad &&
        py <= bottom + pad
      )
        return b;
    }
    return null;
  };

  const onBlockPointerDown = (e: React.PointerEvent<HTMLCanvasElement>) => {
    if (statusRef.current !== "converged") return;
    const c = armRef.current;
    if (!c) return;
    const b = blockAt(c, e.clientX, e.clientY);
    if (!b) return;
    e.preventDefault();
    c.setPointerCapture(e.pointerId);
    // lock to the side band the block starts on (bands never cross centre)
    dragRef.current = {
      block: b,
      band: b.x < 0.5 ? CONFIG.task.placeLeft : CONFIG.task.placeRight,
    };
    // cancel the current attempt; the viewer re-runs once blocks are placed
    engineRef.current!.reset();
    armState.current = { a1: REST[0], a2: REST[1] };
    setTryNote(null);
    // a dragged block rests on the floor
    b.y = 0;
    c.style.cursor = "grabbing";
  };

  const onBlockPointerMove = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const c = armRef.current;
    if (!c) return;
    const drag = dragRef.current;
    if (!drag) {
      // hover affordance: show a grab cursor over a draggable block
      if (statusRef.current === "converged")
        c.style.cursor = blockAt(c, e.clientX, e.clientY) ? "grab" : "default";
      return;
    }
    drag.block.x = clamp(canvasX(c, e.clientX), drag.band[0], drag.band[1]);
  };

  const onBlockPointerUp = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const c = armRef.current;
    if (dragRef.current && c) {
      c.releasePointerCapture?.(e.pointerId);
      c.style.cursor = "grab";
    }
    dragRef.current = null;
  };

  // ---- ⚙ run-config menu (editable while idle; locked once training runs;
  // Reset returns to idle and re-opens the choice) ----
  const setNumColors = (n: RunConfig["numColors"]) =>
    setRunCfg({
      ...runCfg,
      numColors: n,
      // colors are unique per scene, so the palette caps the block count
      maxBlocks: Math.min(runCfg.maxBlocks, n) as RunConfig["maxBlocks"],
    });
  const setMaxBlocks = (n: RunConfig["maxBlocks"]) =>
    setRunCfg({ ...runCfg, maxBlocks: n });

  const active = userSentence ?? demoSentence;

  // Keep the language-encoder chips on a single line for any sentence length:
  // measure the row's natural (unwrapped) width against its width budget and
  // scale the whole row down to fit. Short sentences render at 1:1.
  useLayoutEffect(() => {
    const row = chipRowRef.current;
    if (!row) return;
    const fit = () => {
      row.style.setProperty("--chip-fit", "1");
      const budget = row.clientWidth; // the max-width budget from CSS
      const natural = row.scrollWidth; // full single-line content width
      const scale = natural > budget ? budget / natural : 1;
      row.style.setProperty("--chip-fit", `${scale}`);
    };
    fit();
    window.addEventListener("resize", fit);
    return () => window.removeEventListener("resize", fit);
  }, [active.text]);

  const live = status !== "idle";
  const statusText =
    status === "idle"
      ? "Idle"
      : status === "loading"
        ? "Loading"
        : status === "paused"
          ? "Paused"
          : status === "converged"
            ? "Ready"
            : "Training";
  const stateClass =
    status === "idle"
      ? "is-idle"
      : status === "loading"
        ? "is-live is-loading"
        : status === "paused"
          ? "is-live is-paused"
          : status === "converged"
            ? "is-live is-converged"
            : "is-live";

  return (
    <header className={`hero ${stateClass}`} ref={stageRef}>
      {/* No drawn connectors — the pipeline is stitched together purely by
          motion: a payload token detaches from each source card and glides
          across the gap into the next (offset-path trajectories set in
          layoutWires). Staggered a→b→c so the flow reads source → encoder →
          action head → rollout. Hidden until training starts. */}
      <div className="vla-flow" ref={flowRef} aria-hidden="true">
        <span className="vla-payload vla-flow-a" data-flow="p1" />
        <span className="vla-payload vla-flow-a" data-flow="p2" />
        <span className="vla-payload vla-flow-b" data-flow="p3" />
        <span className="vla-payload vla-flow-b" data-flow="p4" />
        <span className="vla-payload vla-flow-c" data-flow="p5" />
      </div>

      {/* Demonstration — far-left anchor; prompt floats above it */}
      <div className="vla-node vla-input" ref={inputRef}>
        <div className="vla-prompt" ref={promptRef}>
          {demoSentence.text}
          <span className="vla-grip" aria-hidden="true" />
        </div>
        <div className="vla-label">Demonstration</div>
        <canvas className="vla-canvas" ref={demoRef} />
      </div>

      {/* Vision Encoder — 32x32 CNN input, blown up pixelated. Hover (or tap on
          touch) flips the panel to the exact tensor the CNN receives: the input
          is 1 - pixel (see visionTensor in trainer.core.ts), a per-channel invert, so
          a CSS invert(1) reproduces the model's-eye view precisely. */}
      <div
        className={`vla-node vla-vision${modelView ? " model-view" : ""}`}
        ref={visionCardRef}
        onClick={() => setModelView((v) => !v)}
      >
        <div className="vla-label">Vision Encoder</div>
        <canvas className="vla-vision-canvas" ref={visionRef} />
        <div className="vla-vision-hint" aria-hidden="true" />
      </div>

      {/* Language Encoder — frozen pretrained GloVe embeddings, attention-
          pooled (a learned scorer weights each token so filler + padding stop
          diluting the color/verb word); near-synonyms the grammar never
          trained on ("gold", "violet") resolve via the pretrained geometry */}
      <div className="vla-node vla-lang" ref={langCardRef}>
        <div className="vla-label">Language Encoder</div>
        <div className="vla-prompt-echo">&quot;{active.text}&quot;</div>
        <div className="vla-chip-row" ref={chipRowRef}>
          {active.words.map((w, i) => (
            <Fragment key={`${w}-${i}`}>
              {i > 0 && (
                <span className="vla-arrow" aria-hidden="true">
                  →
                </span>
              )}
              <div className="vla-tok">
                <span className="vla-chip">{w}</span>
                <div className="vla-attn">
                  <div
                    className="vla-attn-fill"
                    style={{
                      width: `${Math.round(
                        Math.max(0.06, tokenBars[i] ?? 0.06) * 100
                      )}%`,
                    }}
                  />
                </div>
              </div>
            </Fragment>
          ))}
        </div>
        <div className="vla-decoded">
          decoded target:{" "}
          {decoded ? (
            <>
              <span
                className="vla-swatch"
                style={{ background: decoded.hex }}
              />
              {decoded.name} {(decoded.prob * 100).toFixed(0)}%
            </>
          ) : (
            "—"
          )}
        </div>
      </div>

      {/* Action Head — where the vision + language wires merge; shows the
          policy's current output (the predicted target joint angles) */}
      <div className="vla-node vla-action" ref={actionCardRef}>
        <div className="vla-label">Action Head</div>
        <div className="vla-action-vals" ref={actionValsRef}>
          <span>
            <span className="vla-av-k">shoulder</span>
            <span className="vla-av-v">—</span>
          </span>
          <span>
            <span className="vla-av-k">elbow</span>
            <span className="vla-av-v">—</span>
          </span>
          <span>
            <span className="vla-av-k">gripper</span>
            <span className="vla-av-v">open</span>
          </span>
        </div>
      </div>

      {/* Rollout — policy episodes; becomes an interactive prompt when done */}
      <div className="vla-node vla-output" ref={outputCardRef}>
        {status === "converged" && (
          <div className="vla-try">
            <input
              className="vla-try-input"
              placeholder="e.g. grab the blue cube"
              value={tryText}
              onChange={(e) => setTryText(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") void runCommand();
              }}
            />
            <button className="vla-try-btn" onClick={runCommand} type="button">
              Run
            </button>
            <button
              className="vla-try-btn vla-try-shuffle"
              onClick={randomizeBlocks}
              type="button"
              title="Randomize blocks"
            >
              ⟳
            </button>
            {tryNote && <div className="vla-try-note">{tryNote}</div>}
          </div>
        )}
        <div className="vla-out-head">
          <div className="vla-label">
            {status === "converged" ? "Policy — your command" : "Rollout"}
          </div>
          {rolloutSamples > 0 &&
            (status === "training" ||
              status === "paused" ||
              status === "converged") && (
              <div className="vla-seen">
                trained on {rolloutSamples.toLocaleString()} demonstrations
              </div>
            )}
        </div>
        <canvas
          className="vla-canvas"
          ref={armRef}
          style={status === "converged" ? { touchAction: "none" } : undefined}
          onPointerDown={onBlockPointerDown}
          onPointerMove={onBlockPointerMove}
          onPointerUp={onBlockPointerUp}
          onPointerCancel={onBlockPointerUp}
        />
      </div>

      {/* training control bar */}
      <div className="vla-bar">
        <div className="vla-status">
          <span
            className={`vla-dot${status === "training" ? " is-on" : ""}${
              status === "converged" ? " is-done" : ""
            }`}
          />
          <div className="vla-status-col">
            <span className="vla-status-text">{statusText}</span>
            {live && hud.samples > 0 && (
              <>
                <span className="vla-status-sub">
                  {hud.samples.toLocaleString()} examples
                </span>
                <span className="vla-status-sub">
                  {hud.batches.toLocaleString()} batches
                </span>
              </>
            )}
          </div>
        </div>
        <div className="vla-loss">
          <div className="vla-loss-label">Huber (smooth L1) Loss</div>
          <canvas className="vla-loss-canvas" ref={lossRef} />
        </div>
        <div className="vla-loss-val">{hud.lossText}</div>
        {status === "idle" || status === "loading" ? (
          <>
            {/* ⚙ run config — what to train (palette / density / task set),
                pickable only before training starts */}
            <div className="vla-cfg">
              <button
                className={`vla-btn vla-btn-ghost vla-cfg-btn${cfgOpen ? " is-open" : ""}`}
                onClick={() => setCfgOpen((o) => !o)}
                type="button"
                disabled={status === "loading"}
                aria-expanded={cfgOpen}
                title="Configure what to train"
              >
                ⚙
              </button>
              {cfgOpen && status === "idle" && (
                <div className="vla-cfg-pop">
                  <div className="vla-cfg-row">
                    <span className="vla-cfg-k">colors</span>
                    <div className="vla-cfg-opts">
                      {([2, 4, 8] as const).map((n) => (
                        <button
                          key={n}
                          type="button"
                          className={`vla-cfg-opt${runCfg.numColors === n ? " is-sel" : ""}`}
                          onClick={() => setNumColors(n)}
                        >
                          {n}
                        </button>
                      ))}
                    </div>
                  </div>
                  <div className="vla-cfg-row">
                    <span className="vla-cfg-k">blocks</span>
                    <div className="vla-cfg-opts">
                      {([2, 3, 4] as const).map((n) => (
                        <button
                          key={n}
                          type="button"
                          className={`vla-cfg-opt${runCfg.maxBlocks === n ? " is-sel" : ""}`}
                          disabled={n > runCfg.numColors}
                          title={
                            n > runCfg.numColors
                              ? "every block is a unique color — capped by the palette"
                              : undefined
                          }
                          onClick={() => setMaxBlocks(n)}
                        >
                          ≤{n}
                        </button>
                      ))}
                    </div>
                  </div>
                  <div className="vla-cfg-eta">
                    est. training ~{estimateTrainingSeconds(runCfg)}s
                    <span className="vla-cfg-eta-note"> · on a laptop GPU</span>
                  </div>
                </div>
              )}
            </div>
            <button
              className="vla-btn"
              onClick={onPrimary}
              type="button"
              disabled={status === "loading"}
            >
              Start Training
            </button>
          </>
        ) : status === "converged" ? (
          <button className="vla-btn vla-btn-ghost" onClick={onReset} type="button">
            Reset
          </button>
        ) : (
          <>
            <button className="vla-btn" onClick={onPrimary} type="button">
              {status === "training" ? "Pause" : "Resume"}
            </button>
            <button
              className="vla-btn vla-btn-ghost"
              onClick={onReset}
              type="button"
            >
              Reset
            </button>
          </>
        )}
      </div>

      <div className="hero-content">
        <h1>{profile.name}</h1>
        <div className="hero-tag">{profile.field}</div>
        <div className="hero-cta">
          <a className="btn-primary" href={profile.links.email}>
            Contact
          </a>
          <a className="btn-outline" href="/resume.pdf" download={resumeDownloadName()}>
            Download Resume
          </a>
        </div>
      </div>
    </header>
  );
}
