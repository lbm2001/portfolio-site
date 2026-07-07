"use client";

import { Fragment, useEffect, useLayoutEffect, useRef, useState } from "react";
import { profile, resumeDownloadName } from "@/lib/content";
import {
  REST,
  THETA1_RANGE,
  THETA2_RANGE,
  clamp,
  fk,
  graspTarget,
} from "@/lib/vla/geometry";
import { effectorPx, paintScene, paintSilhouette } from "@/lib/vla/scene";
import {
  COLORS,
  DEFAULT_LAYOUT,
  DEFAULT_SENTENCE,
  MAX_SEQ_LEN,
  hasColor,
  presentColor,
  randomLayout,
  sampleSentence,
  tokenize,
  type BlockPos,
  type Sentence,
} from "@/lib/vla/examples";
import { DEMO_PERIOD_MS, demoPose, makeDemoPlan, type DemoPlan } from "@/lib/vla/demo";
import { IMG_SIZE } from "@/lib/vla/model";
import { VLATrainer, type TrainerStatus } from "@/lib/vla/trainer";
import { CONFIG } from "@/lib/vla/config";

// Live Vision-Language-Action hero: four pipeline boxes ringed around the
// name (Demonstration left w/ floating prompt above, Vision Encoder top,
// Language Encoder bottom, Rollout right). "Start Training" runs a GENUINE
// TensorFlow.js behavioral-cloning loop (lib/vla/trainer.ts) against an
// analytical-IK expert: every scene places two blocks, one per side, colors
// drawn from a 4-color palette, and the command (slot-grammar sentence)
// names one of them. The displayed demonstration swaps every cycle while
// thousands of examples train invisibly; the Rollout runs policy-driven
// episodes. Once the real MSE converges, training stops and the Rollout box
// becomes interactive: type your own command, run it, reshuffle the blocks.

const ACCENT = "#e12d1a"; // = --red; canvases can't read CSS vars cheaply

// Rollout control + episode timing are knobs — tune in lib/vla/config.ts
// (CONFIG.rollout). Episode counts are frames at ~60fps; reachTimeout must stay
// above the synced demo cycle (DEMO_PERIOD_MS in frames) so a training rollout
// is bounded by the cycle reset, not by giving up early.
const STEP_GAIN = CONFIG.rollout.stepGain;
const PREDICT_MS = CONFIG.rollout.predictMs;
const TRAIL_LEN = CONFIG.rollout.trailLen;
const LANG_MS = CONFIG.rollout.langMs;
const GRASP_EPS = CONFIG.rollout.graspEps; // workspace units from block center
const NEAR_FRAMES = CONFIG.rollout.nearFrames;
const REACH_TIMEOUT = CONFIG.rollout.reachTimeout;
const LIFT_FRAMES = CONFIG.rollout.liftFrames;
const TOP_HOLD = CONFIG.rollout.topHold;
const RETURN_FRAMES = CONFIG.rollout.returnFrames;

interface Episode {
  phase: "reach" | "lift" | "return";
  f: number; // frames in the current phase
  near: number;
  nearColor: number; // COLORS index of the block being hovered, or -1
  color: number; // COLORS index of the commanded block
  tokens: number[];
  from: { a1: number; a2: number };
  carry: number | null;
}

const newEpisode = (color: number, tokens: number[]): Episode => ({
  phase: "reach",
  f: 0,
  near: 0,
  nearColor: -1,
  color,
  tokens,
  from: { a1: REST[0], a2: REST[1] },
  carry: null,
});

const ease = (x: number) =>
  x <= 0 ? 0 : x >= 1 ? 1 : (1 - Math.cos(x * Math.PI)) / 2;
const lerp = (a: number, b: number, u: number) => a + (b - a) * u;
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
  const svgRef = useRef<SVGSVGElement>(null);
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
  const armState = useRef({ a1: REST[0], a2: REST[1] });
  const trailRef = useRef<{ x: number; y: number }[]>([]);
  const targetRef = useRef<[number, number] | null>(null);
  const lastPredRef = useRef(0);
  const lastHudRef = useRef(0);
  const lastLangRef = useRef(0);
  // paused-duration accounting so the demo cycle resumes exactly where it
  // left off instead of jumping ahead by the real wall-clock pause length
  const pausedAccumRef = useRef(0);
  const pauseStartRef = useRef<number | null>(null);

  // demonstration state — a fresh layout/sentence/trajectory every cycle
  const demoLayoutRef = useRef(DEFAULT_LAYOUT);
  const demoSentenceRef = useRef<Sentence>(DEFAULT_SENTENCE);
  const demoPlanRef = useRef<DemoPlan | null>(null);
  const lastCycleRef = useRef(-1);
  const demoPoseRef = useRef({ a1: REST[0], a2: REST[1], carry: null as number | null });

  // rollout state
  const rolloutLayoutRef = useRef(DEFAULT_LAYOUT);
  const episodeRef = useRef<Episode | null>(null);
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
  const [hud, setHud] = useState({ lossText: "—", samples: 0 });
  const [demoSentence, setDemoSentence] = useState<Sentence>(DEFAULT_SENTENCE);
  const [userSentence, setUserSentence] = useState<Sentence | null>(null);
  const [tryText, setTryText] = useState("");
  const chipRowRef = useRef<HTMLDivElement | null>(null);
  const [tokenBars, setTokenBars] = useState<number[]>([]);
  const [decoded, setDecoded] = useState<{ name: string; hex: string; prob: number } | null>(null);
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
      const svg = svgRef.current;
      if (
        !stage ||
        !svg ||
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
      const m = (el: Element) => {
        const c = el.getBoundingClientRect();
        return {
          cx: ((c.left - R.left + c.width / 2) / R.width) * 1000,
          cy: ((c.top - R.top + c.height / 2) / R.height) * 1000,
          left: ((c.left - R.left) / R.width) * 1000,
          right: ((c.right - R.left) / R.width) * 1000,
          top: ((c.top - R.top) / R.height) * 1000,
          bottom: ((c.bottom - R.top) / R.height) * 1000,
        };
      };
      const I = m(inputRef.current);
      const P = m(promptRef.current);
      const V = m(visionCardRef.current);
      const L = m(langCardRef.current);
      const A = m(actionCardRef.current);
      const O = m(outputCardRef.current);
      const r = (n: number) => n.toFixed(1);
      // Pipeline, five edge-to-edge segments (base + traveling pulse share the
      // SAME geometry so the current rides exactly on the drawn wire). Two
      // rails, NEITHER of which crosses the centered name:
      //   TOP rail    (y = V.cy): Demonstration → Vision.left, then
      //               Vision.right → across → DOWN into the Action Head TOP.
      //   BOTTOM rail (y = L.cy): Prompt → Language.left, then
      //               Language.right → across → UP into the Action Head BOTTOM.
      // Each encoder sits INLINE on its rail (wire enters its left edge, exits
      // its right edge); both rails then drop VERTICALLY into the Action Head
      // from above/below. The Action Head drives the Rollout (p5). Every
      // endpoint lands on a real card EDGE — no floating mid-air stops.
      const p1 = `M${r(I.cx)},${r(I.top)} V${r(V.cy)} H${r(V.left)}`;
      const p2 = `M${r(P.cx)},${r(P.bottom)} V${r(L.cy)} H${r(L.left)}`;
      const p3 = `M${r(V.right)},${r(V.cy)} H${r(A.cx)} V${r(A.top)}`;
      const p4 = `M${r(L.right)},${r(L.cy)} H${r(A.cx)} V${r(A.bottom)}`;
      const p5 = `M${r(A.right)},${r(A.cy)} H${r(O.left)}`;
      const set = (wire: string, d: string) =>
        svg.querySelectorAll(`[data-wire="${wire}"]`).forEach((p) => p.setAttribute("d", d));
      set("p1", p1);
      set("p2", p2);
      set("p3", p3);
      set("p4", p4);
      set("p5", p5);
    };

    // the Action Head's live output — the policy's current predicted target
    // joint angles (null → em dashes when nothing is being predicted)
    const setActionVals = (t: [number, number] | null) => {
      const el = actionValsRef.current;
      if (!el || el.children.length < 2) return;
      const v0 = el.children[0].lastElementChild;
      const v1 = el.children[1].lastElementChild;
      if (v0) v0.textContent = t ? fmtAngle(t[0]) : "—";
      if (v1) v1.textContent = t ? fmtAngle(t[1]) : "—";
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
        });
        return;
      }

      let a1: number;
      let a2: number;
      let carry: number | null = null;

      if (st === "training" || st === "loading") {
        // subtract accumulated paused time so the cycle resumes exactly
        // where it left off instead of jumping ahead by the pause length
        const effectiveNow = now - pausedAccumRef.current;
        const cycle = Math.floor(effectiveNow / DEMO_PERIOD_MS);
        if (cycle !== lastCycleRef.current || !demoPlanRef.current) {
          const first = lastCycleRef.current === -1;
          lastCycleRef.current = cycle;
          if (!first) {
            demoLayoutRef.current = randomLayout();
            // the demonstrated command names one of the scene's two blocks
            const s = sampleSentence(presentColor(demoLayoutRef.current));
            demoSentenceRef.current = s;
            setDemoSentence(s);
            // the language panel follows the demo until a user command locks it
            if (!userSentenceRef.current) activeTokensRef.current = s.tokens;
          }
          demoPlanRef.current = makeDemoPlan(
            demoLayoutRef.current,
            demoSentenceRef.current.color
          );
        }

        const t = reduced.matches
          ? 0.2
          : (effectiveNow % DEMO_PERIOD_MS) / DEMO_PERIOD_MS;
        const pose = demoPose(demoPlanRef.current, t);
        a1 = pose.a1;
        a2 = pose.a2;
        carry = pose.carry;
      } else {
        // idle OR converged: the demonstrations stop once the policy is
        // trained — the arm returns to the resting sway (the CSS also fades
        // the box back to its dormant, pre-training transparency)
        [a1, a2] = wiggle(now, 0, 1.7);
      }

      demoPoseRef.current = { a1, a2, carry };
      paintScene(ctx, W, H, {
        a1,
        a2,
        layout: demoLayoutRef.current,
        accent: ACCENT,
        carry,
      });
    };

    // advance one episode by a frame (reach → lift → return); ends by
    // nulling the episode (arm left at REST) — training re-syncs a fresh
    // attempt on the next demo cycle, converged waits for the next command
    const runEpisode = (ep: Episode, now: number, W: number, H: number) => {
      const arm = armState.current;
      const trainer = trainerRef.current!;
      if (ep.phase === "reach") {
        // the model predicts an ABSOLUTE target (refreshed every PREDICT_MS);
        // the step direction is recomputed every FRAME from that target
        // against the arm's actual current pose, so the step naturally
        // shrinks as it closes in (proportional control)
        if (now - lastPredRef.current > PREDICT_MS) {
          // frozen per-cycle snapshot (training) / final weights (converged) —
          // the arm still re-predicts every PREDICT_MS as it moves (closed
          // loop), just against fixed weights for the whole attempt
          const t = trainer.predictFrozenTarget(arm.a1, arm.a2, ep.tokens, rolloutLayoutRef.current);
          if (t) targetRef.current = t;
          lastPredRef.current = now;
        }
        const t = targetRef.current;
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
          trailRef.current.push(effectorPx(W, H, arm.a1, arm.a2));
          if (trailRef.current.length > TRAIL_LEN) trailRef.current.shift();
        }
        // the gripper closes on whatever block the effector settles over —
        // not just the commanded one — so a wrong-side reach visibly lifts
        // the wrong block instead of hovering next to it until the timeout
        const e = fk(arm.a1, arm.a2);
        let touched = -1;
        for (const b of rolloutLayoutRef.current) {
          const g = graspTarget(b.x, b.size); // grasp height follows block size
          if (Math.hypot(e.ex - g.x, e.ey - g.y) < GRASP_EPS) {
            touched = b.color;
            break;
          }
        }
        ep.near = touched >= 0 && touched === ep.nearColor ? ep.near + 1 : touched >= 0 ? 1 : 0;
        ep.nearColor = touched;
        if (touched >= 0 && ep.near >= NEAR_FRAMES) {
          ep.phase = "lift"; // grasp whatever it settled on, lift straight up
          ep.from = { ...arm };
          ep.carry = touched;
          ep.f = 0;
        } else if (ep.f > REACH_TIMEOUT) {
          ep.phase = "return";
          ep.from = { ...arm };
          ep.carry = null;
          ep.f = 0;
        }
      } else if (ep.phase === "lift") {
        const u = ease(Math.min(1, ep.f / LIFT_FRAMES));
        arm.a1 = lerp(ep.from.a1, REST[0], u);
        arm.a2 = lerp(ep.from.a2, REST[1], u);
        if (ep.f > LIFT_FRAMES + TOP_HOLD) endEpisode();
      } else {
        const u = ep.f / RETURN_FRAMES;
        arm.a1 = lerp(ep.from.a1, REST[0], ease(u));
        arm.a2 = lerp(ep.from.a2, REST[1], ease(u));
        if (u >= 1) endEpisode();
      }
      if (episodeRef.current) ep.f++;
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

      if (st === "paused") {
        // frozen: redraw current pose, advance nothing (matches the demo)
        const ep = episodeRef.current;
        paintScene(ctx, W, H, {
          a1: arm.a1,
          a2: arm.a2,
          layout: rolloutLayoutRef.current,
          accent: ACCENT,
          carry: ep?.carry ?? null,
        });
        setActionVals(targetRef.current);
        return;
      }

      if (st === "idle") {
        [arm.a1, arm.a2] = wiggle(now, 2.6, 4.1);
      } else if (trainer?.ready) {
        if (st === "converged") {
          const ep = episodeRef.current;
          if (ep) runEpisode(ep, now, W, H);
          else [arm.a1, arm.a2] = wiggle(now, 2.6, 4.1); // waiting for command
        } else {
          // training: a fresh synced attempt at every new demonstration cycle,
          // run against a policy snapshot frozen at this boundary so the whole
          // attempt reflects one policy generation (not a live-drifting target)
          if (rolloutCycleRef.current !== lastCycleRef.current) {
            rolloutCycleRef.current = lastCycleRef.current;
            rolloutLayoutRef.current = demoLayoutRef.current;
            arm.a1 = REST[0];
            arm.a2 = REST[1];
            trailRef.current = [];
            targetRef.current = null;
            trainer.snapshotPolicy();
            episodeRef.current = newEpisode(
              demoSentenceRef.current.color,
              demoSentenceRef.current.tokens
            );
          }
          const ep = episodeRef.current;
          if (ep) runEpisode(ep, now, W, H);
          // else: this cycle's attempt is done — hold at REST until the next
        }
      } else {
        arm.a1 = REST[0];
        arm.a2 = REST[1];
      }

      const ep = episodeRef.current;
      paintScene(ctx, W, H, {
        a1: arm.a1,
        a2: arm.a2,
        layout: rolloutLayoutRef.current,
        accent: ACCENT,
        trail: ep?.phase === "reach" ? trailRef.current : null,
        lossNorm: trainer?.lossNorm() ?? 1,
        carry: ep?.carry ?? null,
      });
      setActionVals(st === "idle" ? null : targetRef.current);
    };

    const endEpisode = () => {
      const arm = armState.current;
      arm.a1 = REST[0];
      arm.a2 = REST[1];
      trailRef.current = [];
      targetRef.current = null;
      episodeRef.current = null;
    };

    const drawVision = () => {
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
      ctx.strokeStyle = "rgba(0,0,0,.12)";
      ctx.lineWidth = 1;
      ctx.strokeRect(0.5, 0.5, W - 1, W - 1);
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

    // real language-encoder readouts: decoded color + each token's live
    // ATTENTION weight (see attentionWeights in trainer.ts — the exact
    // masked-softmax weights the attention-pooling layer uses, recomputed
    // from the linear scorer's weights, not an approximation)
    const langViz = (now: number) => {
      const trainer = trainerRef.current;
      if (!trainer?.ready || now - lastLangRef.current < LANG_MS) return;
      lastLangRef.current = now;
      const tokens = activeTokensRef.current;
      const d = trainer.decodeColor(tokens);
      if (!d) return;
      setDecoded({ name: COLORS[d.color].name, hex: COLORS[d.color].hex, prob: d.prob });
      const bars = trainer.attentionWeights(tokens);
      if (bars) setTokenBars(bars);
    };

    let raf = 0;
    const loop = (now: number) => {
      layoutWires();
      drawDemo(now);
      drawArm(now);
      drawVision();
      drawLossCurve();
      langViz(now);
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);
    window.addEventListener("resize", layoutWires);
    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener("resize", layoutWires);
      trainerRef.current?.reset();
    };
  }, []);

  // ---- controls ----
  const onUpdate = () => {
    const trainer = trainerRef.current!;
    if (trainer.status !== statusRef.current) {
      setStatusBoth(trainer.status);
      if (trainer.status === "converged") {
        // auto-episodes end; the rollout waits for the user's command
        episodeRef.current = null;
        trailRef.current = [];
        targetRef.current = null;
        armState.current = { a1: REST[0], a2: REST[1] };
        // detach the rollout scene from the demo's: during training
        // rolloutLayoutRef holds the SAME objects as demoLayoutRef (assigned,
        // not copied), so dragging a rollout block would also move it in the
        // Demonstration box. Clone so "try it" edits are rollout-only.
        rolloutLayoutRef.current = rolloutLayoutRef.current.map((b) => ({ ...b }));
      }
    }
    const now = performance.now();
    if (now - lastHudRef.current > 150 && !Number.isNaN(trainer.loss)) {
      lastHudRef.current = now;
      setHud({ lossText: trainer.loss.toFixed(3), samples: trainer.samples });
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
      episodeRef.current = null;
      rolloutCycleRef.current = -1;
      trailRef.current = [];
      targetRef.current = null;
      armState.current = { a1: REST[0], a2: REST[1] };
      userSentenceRef.current = null;
      pausedAccumRef.current = 0;
      pauseStartRef.current = null;
      setUserSentence(null);
      setDecoded(null);
      setTokenBars([]);
      void trainer.start(onUpdate);
    }
  };

  const onReset = () => {
    trainerRef.current?.reset();
    setStatusBoth("idle");
    episodeRef.current = null;
    rolloutCycleRef.current = -1;
    trailRef.current = [];
    targetRef.current = null;
    armState.current = { a1: REST[0], a2: REST[1] };
    demoLayoutRef.current = DEFAULT_LAYOUT;
    demoSentenceRef.current = DEFAULT_SENTENCE;
    demoPlanRef.current = null;
    lastCycleRef.current = -1;
    rolloutLayoutRef.current = DEFAULT_LAYOUT;
    activeTokensRef.current = DEFAULT_SENTENCE.tokens;
    userSentenceRef.current = null;
    pausedAccumRef.current = 0;
    pauseStartRef.current = null;
    setDemoSentence(DEFAULT_SENTENCE);
    setUserSentence(null);
    setTryText("");
    setTryNote(null);
    setTokenBars([]);
    setDecoded(null);
    setHud({ lossText: "—", samples: 0 });
  };

  /** "Try it" mode: run the user's own sentence through the trained policy. */
  const runCommand = () => {
    const trainer = trainerRef.current;
    if (!trainer?.ready || statusRef.current !== "converged") return;
    const text = tryText.trim();
    if (!text) return;
    const tokens = tokenize(text);
    const d = trainer.decodeColor(tokens);
    if (!d) return;
    const words = text
      .toLowerCase()
      .replace(/[^a-z ]/g, "")
      .split(" ")
      .filter(Boolean)
      .slice(0, MAX_SEQ_LEN);
    const sentence: Sentence = { color: d.color, text, words, tokens };
    userSentenceRef.current = sentence;
    setUserSentence(sentence);
    activeTokensRef.current = tokens;
    setDecoded({ name: COLORS[d.color].name, hex: COLORS[d.color].hex, prob: d.prob });
    // only two of the eight colors are in the scene — a command naming an
    // absent one can't be executed; point at the shuffle button instead
    if (!hasColor(rolloutLayoutRef.current, d.color)) {
      setTryNote(`no ${COLORS[d.color].name} block in this scene — ⟳ to reshuffle`);
      return;
    }
    setTryNote(null);
    armState.current = { a1: REST[0], a2: REST[1] };
    trailRef.current = [];
    targetRef.current = null;
    episodeRef.current = newEpisode(d.color, tokens);
  };

  const randomizeBlocks = () => {
    rolloutLayoutRef.current = randomLayout();
    // abort any in-flight episode so the arm re-plans against the new scene
    episodeRef.current = null;
    trailRef.current = [];
    targetRef.current = null;
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
      if (
        px >= cx - half - pad &&
        px <= cx + half + pad &&
        py >= floorY - b.size * S - pad &&
        py <= floorY + pad
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
    episodeRef.current = null;
    trailRef.current = [];
    targetRef.current = null;
    armState.current = { a1: REST[0], a2: REST[1] };
    setTryNote(null);
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
      {/* orthogonal data-pipeline rails; hidden until training starts, then
          the pipeline "draws on" (Demonstration→Encoders, then
          Encoders→Rollout) once, and a single small "current" travels each
          wire left→right for as long as training is actively running */}
      <svg
        className="vla-wires"
        ref={svgRef}
        viewBox="0 0 1000 1000"
        preserveAspectRatio="none"
        aria-hidden="true"
      >
        <g className="vla-wire-base">
          <path
            data-wire="p1"
            vectorEffect="non-scaling-stroke"
            pathLength={100}
            className="vla-seg-a"
          />
          <path
            data-wire="p2"
            vectorEffect="non-scaling-stroke"
            pathLength={100}
            className="vla-seg-a"
          />
          <path
            data-wire="p3"
            vectorEffect="non-scaling-stroke"
            pathLength={100}
            className="vla-seg-b"
          />
          <path
            data-wire="p4"
            vectorEffect="non-scaling-stroke"
            pathLength={100}
            className="vla-seg-b"
          />
          <path
            data-wire="p5"
            vectorEffect="non-scaling-stroke"
            pathLength={100}
            className="vla-seg-c"
          />
        </g>
        <g className="vla-wire-pulse">
          <path
            data-wire="p1"
            vectorEffect="non-scaling-stroke"
            pathLength={100}
            className="vla-pulse-a"
          />
          <path
            data-wire="p2"
            vectorEffect="non-scaling-stroke"
            pathLength={100}
            className="vla-pulse-a"
          />
          <path
            data-wire="p3"
            vectorEffect="non-scaling-stroke"
            pathLength={100}
            className="vla-pulse-b"
          />
          <path
            data-wire="p4"
            vectorEffect="non-scaling-stroke"
            pathLength={100}
            className="vla-pulse-b"
          />
          <path
            data-wire="p5"
            vectorEffect="non-scaling-stroke"
            pathLength={100}
            className="vla-pulse-c"
          />
        </g>
      </svg>

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
          is 1 - pixel (see visionTensor in trainer.ts), a per-channel invert, so
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
                if (e.key === "Enter") runCommand();
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
        <div className="vla-label">
          {status === "converged" ? "Policy — your command" : "Rollout"}
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
              <span className="vla-status-sub">
                {hud.samples.toLocaleString()} ex
              </span>
            )}
          </div>
        </div>
        <div className="vla-loss">
          <div className="vla-loss-label">browser training · MSE loss</div>
          <canvas className="vla-loss-canvas" ref={lossRef} />
        </div>
        <div className="vla-loss-val">{hud.lossText}</div>
        {status === "idle" || status === "loading" ? (
          <button
            className="vla-btn"
            onClick={onPrimary}
            type="button"
            disabled={status === "loading"}
          >
            Start Training
          </button>
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
