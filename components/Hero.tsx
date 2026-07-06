"use client";

import { Fragment, useEffect, useRef, useState } from "react";
import { profile, resumeDownloadName } from "@/lib/content";
import {
  BLACK_BLOCK,
  RED_BLOCK,
  REST,
  THETA1_RANGE,
  THETA2_RANGE,
  clamp,
  fk,
  graspTarget,
} from "@/lib/vla/geometry";
import { effectorPx, paintScene } from "@/lib/vla/scene";
import {
  EXAMPLES,
  attentionWeight,
  randomExample,
  type BlockColor,
  type Example,
} from "@/lib/vla/examples";
import {
  DEMO_PERIOD_MS,
  demoPose,
  makeDemoPlan,
  type DemoPlan,
  type DemoPose,
} from "@/lib/vla/demo";
import { VLATrainer, type TrainerStatus } from "@/lib/vla/trainer";

// Live Vision-Language-Action hero: four pipeline boxes ringed around the
// name (Demonstration left w/ floating prompt above, Vision Encoder top,
// Language Encoder bottom, Rollout right). "Start Training" runs a GENUINE
// TensorFlow.js behavioral-cloning loop (lib/vla/trainer.ts) against an
// analytical-IK expert. The displayed example (prompt + demonstration
// trajectory) swaps every demo cycle while thousands of varied examples
// train in the background; the Rollout runs repeated policy-driven episodes
// that start failing and begin to succeed (reach center → grasp → lift)
// after ~10-15s as the real MSE converges.

const ACCENT = "#e12d1a"; // = --red; canvases can't read CSS vars cheaply

/** Per-frame joint update: fraction of the predicted delta applied at 60fps. */
const STEP_GAIN = 0.08;
/** Min ms between policy inference calls for the rollout. */
const PREDICT_MS = 80;
/** Max rendered trail points. */
const TRAIL_LEN = 64;

// rollout episode tuning (frames at ~60fps)
const GRASP_EPS = 0.07; // workspace units from the block center
const NEAR_FRAMES = 6; // consecutive close frames that count as a grasp
const REACH_TIMEOUT = 280; // ~4.7s per attempt before giving up
const LIFT_FRAMES = 55; // grasp point -> straight-up
const TOP_HOLD = 30; // 0.5s holding the block at the top, arm straight
const RETURN_FRAMES = 40; // failed attempts lerp back to rest

interface Episode {
  phase: "reach" | "lift" | "return";
  f: number; // frames in the current phase
  near: number;
  color: BlockColor;
  tokens: number[];
  from: { a1: number; a2: number };
  lift: [number, number];
  carry: BlockColor | null;
}

const newEpisode = (ex: Example): Episode => ({
  phase: "reach",
  f: 0,
  near: 0,
  color: ex.color,
  tokens: ex.tokens,
  from: { a1: REST[0], a2: REST[1] },
  lift: REST,
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

const blockOf = (color: BlockColor) => (color === "red" ? RED_BLOCK : BLACK_BLOCK);

export default function Hero() {
  const stageRef = useRef<HTMLElement>(null);
  const svgRef = useRef<SVGSVGElement>(null);
  const inputRef = useRef<HTMLDivElement>(null);
  const promptRef = useRef<HTMLDivElement>(null);
  const visionCardRef = useRef<HTMLDivElement>(null);
  const langCardRef = useRef<HTMLDivElement>(null);
  const outputCardRef = useRef<HTMLDivElement>(null);
  const demoRef = useRef<HTMLCanvasElement>(null);
  const visionRef = useRef<HTMLCanvasElement>(null);
  const armRef = useRef<HTMLCanvasElement>(null);
  const lossRef = useRef<HTMLCanvasElement>(null);
  const demoJointsRef = useRef<HTMLDivElement>(null);
  const armJointsRef = useRef<HTMLDivElement>(null);

  const trainerRef = useRef<VLATrainer | null>(null);
  const trainingRef = useRef(false);
  const armState = useRef({ a1: REST[0], a2: REST[1] });
  const trailRef = useRef<{ x: number; y: number }[]>([]);
  const deltaRef = useRef<[number, number] | null>(null);
  const lastPredRef = useRef(0);
  const lastHudRef = useRef(0);
  // displayed training example — swapped once per demo cycle
  const exampleRef = useRef<Example>(EXAMPLES[0]);
  const planRef = useRef<DemoPlan | null>(null);
  const lastCycleRef = useRef(-1);
  const demoPoseRef = useRef<DemoPose>({ a1: REST[0], a2: REST[1], carry: null });
  const episodeRef = useRef<Episode>(newEpisode(EXAMPLES[0]));
  // offscreen render of the demonstration scene for the vision encoder
  const demoSceneRef = useRef<HTMLCanvasElement | null>(null);
  const demoThumbRef = useRef<HTMLCanvasElement | null>(null);

  const [training, setTraining] = useState(false);
  const [status, setStatus] = useState<TrainerStatus>("idle");
  const [hud, setHud] = useState({ lossText: "—", progress: 0, samples: 0 });
  const [example, setExample] = useState<Example>(EXAMPLES[0]);

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
          top: ((c.top - R.top) / R.height) * 1000,
          bottom: ((c.bottom - R.top) / R.height) * 1000,
        };
      };
      const I = m(inputRef.current);
      const P = m(promptRef.current);
      const V = m(visionCardRef.current);
      const L = m(langCardRef.current);
      const O = m(outputCardRef.current);
      const r = (n: number) => n.toFixed(1);
      // vision reads the demonstration image; language reads the raw prompt
      const dTop = `M${r(I.cx)},${r(I.top)} V${r(V.cy)} H${r(O.cx)} V${r(O.top)}`;
      const dBot = `M${r(P.cx)},${r(P.bottom)} V${r(L.cy)} H${r(O.cx)} V${r(O.bottom)}`;
      svg
        .querySelectorAll('[data-wire="top"]')
        .forEach((p) => p.setAttribute("d", dTop));
      svg
        .querySelectorAll('[data-wire="bottom"]')
        .forEach((p) => p.setAttribute("d", dBot));
    };

    const setJoints = (el: HTMLDivElement | null, a1: number, a2: number) => {
      if (!el || el.children.length < 2) return;
      el.children[0].textContent = `θ1 ${fmtAngle(a1)}`;
      el.children[1].textContent = `θ2 ${fmtAngle(a2)}`;
    };

    const drawDemo = (now: number) => {
      const c = demoRef.current;
      if (!c) return;
      const { ctx, W, H } = fitCanvas(c);
      const isTraining = trainingRef.current;

      // swap in a fresh training example (new phrasing/color + noisy
      // trajectory) at every cycle boundary — the thousands of examples
      // trained in between are not displayed
      const cycle = Math.floor(now / DEMO_PERIOD_MS);
      if (cycle !== lastCycleRef.current || !planRef.current) {
        const first = lastCycleRef.current === -1;
        lastCycleRef.current = cycle;
        const ex = first ? exampleRef.current : randomExample();
        exampleRef.current = ex;
        planRef.current = makeDemoPlan(ex.color);
        setExample(ex);
      }

      // 16x16 tokenization grid — the vision encoder's resolution — only
      // overlaid once training starts
      if (isTraining) {
        ctx.strokeStyle = "rgba(0,0,0,.16)";
        ctx.lineWidth = 1;
        for (let i = 0; i <= 16; i++) {
          const x = Math.min(W - 0.5, Math.max(0.5, Math.round((i * W) / 16) + 0.5));
          ctx.beginPath();
          ctx.moveTo(x, 0);
          ctx.lineTo(x, H);
          ctx.stroke();
          const y = Math.min(H - 0.5, Math.max(0.5, Math.round((i * H) / 16) + 0.5));
          ctx.beginPath();
          ctx.moveTo(0, y);
          ctx.lineTo(W, y);
          ctx.stroke();
        }
      }

      const t = reduced.matches ? 0.18 : (now % DEMO_PERIOD_MS) / DEMO_PERIOD_MS;
      const pose = demoPose(planRef.current, t);
      demoPoseRef.current = pose;

      paintScene(ctx, W, H, {
        a1: pose.a1,
        a2: pose.a2,
        accent: ACCENT,
        carry: pose.carry,
      });
      setJoints(demoJointsRef.current, pose.a1, pose.a2);
    };

    // one policy-driven rollout episode after another: reach (closed loop on
    // the model's own predictions) → grasp when the effector holds the block
    // center → scripted lift → return → next attempt
    const drawArm = (now: number) => {
      const c = armRef.current;
      if (!c) return;
      const { ctx, W, H } = fitCanvas(c);
      const trainer = trainerRef.current;
      const arm = armState.current;
      const ep = episodeRef.current;
      const isTraining = trainingRef.current;

      if (isTraining && trainer?.ready) {
        if (ep.phase === "reach") {
          if (now - lastPredRef.current > PREDICT_MS) {
            const d = trainer.predictDelta(arm.a1, arm.a2, ep.tokens);
            if (d) deltaRef.current = d;
            lastPredRef.current = now;
          }
          const d = deltaRef.current;
          if (d) {
            arm.a1 = clamp(
              arm.a1 + clamp(d[0], -Math.PI, Math.PI) * STEP_GAIN,
              THETA1_RANGE[0],
              THETA1_RANGE[1]
            );
            arm.a2 = clamp(
              arm.a2 + clamp(d[1], -Math.PI, Math.PI) * STEP_GAIN,
              THETA2_RANGE[0],
              THETA2_RANGE[1]
            );
            trailRef.current.push(effectorPx(W, H, arm.a1, arm.a2));
            if (trailRef.current.length > TRAIL_LEN) trailRef.current.shift();
          }
          const g = graspTarget(blockOf(ep.color));
          const e = fk(arm.a1, arm.a2);
          const dist = Math.hypot(e.ex - g.x, e.ey - g.y);
          ep.near = dist < GRASP_EPS ? ep.near + 1 : 0;
          if (ep.near >= NEAR_FRAMES) {
            // success: the policy held the block center — grasp, then lift
            // it all the way up until the arm is straight
            ep.phase = "lift";
            ep.from = { ...arm };
            ep.lift = REST;
            ep.carry = ep.color;
            ep.f = 0;
          } else if (ep.f > REACH_TIMEOUT) {
            // failed attempt: back to rest, try again with a fresher policy
            ep.phase = "return";
            ep.from = { ...arm };
            ep.carry = null;
            ep.f = 0;
          }
        } else if (ep.phase === "lift") {
          const u = ease(Math.min(1, ep.f / LIFT_FRAMES));
          arm.a1 = lerp(ep.from.a1, ep.lift[0], u);
          arm.a2 = lerp(ep.from.a2, ep.lift[1], u);
          // the episode only ends after the block has been HELD at the top
          // with the arm straight for TOP_HOLD frames (~0.5s)
          if (ep.f > LIFT_FRAMES + TOP_HOLD) {
            episodeRef.current = newEpisode(exampleRef.current);
            trailRef.current = [];
            deltaRef.current = null;
            arm.a1 = REST[0];
            arm.a2 = REST[1];
          }
        } else {
          // failed attempt: lerp back to rest empty-handed, then retry
          const u = ep.f / RETURN_FRAMES;
          arm.a1 = lerp(ep.from.a1, REST[0], ease(u));
          arm.a2 = lerp(ep.from.a2, REST[1], ease(u));
          if (u >= 1) {
            episodeRef.current = newEpisode(exampleRef.current);
            trailRef.current = [];
            deltaRef.current = null;
            arm.a1 = REST[0];
            arm.a2 = REST[1];
          }
        }
        ep.f++;
      } else if (!isTraining) {
        arm.a1 = REST[0];
        arm.a2 = REST[1];
      }

      paintScene(ctx, W, H, {
        a1: arm.a1,
        a2: arm.a2,
        accent: ACCENT,
        trail: isTraining && ep.phase === "reach" ? trailRef.current : null,
        lossNorm: trainer?.lossNorm() ?? 1,
        carry: ep.carry,
      });
      setJoints(armJointsRef.current, arm.a1, arm.a2);
    };

    const drawVision = () => {
      const c = visionRef.current;
      if (!c) return;
      const dpr = window.devicePixelRatio || 1;
      const S = 176;
      if (c.width !== Math.round(S * dpr)) c.width = Math.round(S * dpr);
      if (c.height !== Math.round(S * dpr)) c.height = Math.round(S * dpr);
      const ctx = c.getContext("2d")!;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, S, S);

      const isTraining = trainingRef.current;

      // the encoder's input is the DEMONSTRATION image (the box it's wired
      // from): re-render the current demo pose offscreen, downsample to the
      // model's 16x16 resolution, display it as the patch grid
      let thumb: Uint8ClampedArray | null = null;
      if (isTraining) {
        if (!demoSceneRef.current) {
          demoSceneRef.current = document.createElement("canvas");
          demoSceneRef.current.width = 184;
          demoSceneRef.current.height = 186;
          demoThumbRef.current = document.createElement("canvas");
          demoThumbRef.current.width = 16;
          demoThumbRef.current.height = 16;
        }
        const sc = demoSceneRef.current;
        const tc = demoThumbRef.current!;
        const sctx = sc.getContext("2d")!;
        const p = demoPoseRef.current;
        sctx.fillStyle = "#ffffff";
        sctx.fillRect(0, 0, sc.width, sc.height);
        paintScene(sctx, sc.width, sc.height, {
          a1: p.a1,
          a2: p.a2,
          accent: ACCENT,
          carry: p.carry,
        });
        const tctx = tc.getContext("2d", { willReadFrequently: true })!;
        tctx.imageSmoothingEnabled = true;
        tctx.clearRect(0, 0, 16, 16);
        tctx.drawImage(sc, 0, 0, sc.width, sc.height, 0, 0, 16, 16);
        thumb = tctx.getImageData(0, 0, 16, 16).data;
      }

      const gap = 3;
      const np = 4;
      const po = (S - gap * (np + 1)) / np;
      const ps = po / 4;
      for (let py = 0; py < np; py++) {
        for (let px = 0; px < np; px++) {
          const ox = gap + px * (po + gap);
          const oy = gap + py * (po + gap);
          ctx.fillStyle = (px + py) % 2 ? "#f6f6f6" : "#efefef";
          ctx.fillRect(ox, oy, po, po);
          for (let iy = 0; iy < 4; iy++) {
            for (let ix = 0; ix < 4; ix++) {
              let r = 244;
              let g = 244;
              let b = 244;
              if (thumb) {
                const idx = ((py * 4 + iy) * 16 + (px * 4 + ix)) * 4;
                r = thumb[idx];
                g = thumb[idx + 1];
                b = thumb[idx + 2];
              }
              ctx.fillStyle = `rgb(${r | 0},${g | 0},${b | 0})`;
              ctx.fillRect(ox + ix * ps + 0.5, oy + iy * ps + 0.5, ps - 1, ps - 1);
            }
          }
        }
      }
    };

    const drawLossCurve = () => {
      const c = lossRef.current;
      if (!c) return;
      const { ctx, W, H } = fitCanvas(c, 300, 34);
      const hist = trainerRef.current?.lossHistory ?? [];
      const pad = 3;

      ctx.strokeStyle = "#efefef";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(0, H - pad);
      ctx.lineTo(W, H - pad);
      ctx.stroke();
      if (hist.length < 2) return;

      // scale against the worst loss seen so the real MSE (whatever its
      // magnitude) always slopes top-left → bottom-right as it converges
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

    let raf = 0;
    const loop = (now: number) => {
      layoutWires();
      drawDemo(now);
      drawArm(now);
      drawVision();
      drawLossCurve();
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

  // ---- the one interactive control: Start Training / Reset ----
  const onToggle = () => {
    const trainer = (trainerRef.current ??= new VLATrainer());
    if (trainingRef.current) {
      trainer.reset();
      trainingRef.current = false;
      trailRef.current = [];
      deltaRef.current = null;
      armState.current = { a1: REST[0], a2: REST[1] };
      episodeRef.current = newEpisode(exampleRef.current);
      setTraining(false);
      setStatus("idle");
      setHud({ lossText: "—", progress: 0, samples: 0 });
    } else {
      trainingRef.current = true;
      trailRef.current = [];
      deltaRef.current = null;
      episodeRef.current = newEpisode(exampleRef.current);
      setTraining(true);
      void trainer.start(() => {
        setStatus(trainer.status);
        const now = performance.now();
        if (now - lastHudRef.current > 150 && !Number.isNaN(trainer.loss)) {
          lastHudRef.current = now;
          setHud({
            lossText: trainer.loss.toFixed(3),
            progress: 1 - trainer.lossNorm(),
            samples: trainer.samples,
          });
        }
      });
    }
  };

  // live attention weights over the CURRENT example's tokens — verbs decay,
  // the object words ramp up as the real loss converges
  const p = training ? hud.progress : 0;
  const pct = (v: number) => `${(Math.max(0.04, Math.min(1, v)) * 100).toFixed(0)}%`;

  const statusText =
    status === "idle" ? "Idle" : status === "loading" ? "Loading" : "Training";

  return (
    <header className={`hero${training ? " is-training" : ""}`} ref={stageRef}>
      {/* orthogonal data-pipeline rails, measured from the real box positions */}
      <svg
        className="vla-wires"
        ref={svgRef}
        viewBox="0 0 1000 1000"
        preserveAspectRatio="none"
        aria-hidden="true"
      >
        <g className="vla-wire-base">
          <path data-wire="top" vectorEffect="non-scaling-stroke" />
          <path data-wire="bottom" vectorEffect="non-scaling-stroke" />
        </g>
        <g className="vla-wire-signal">
          <g className="vla-wire-glow">
            <path data-wire="top" vectorEffect="non-scaling-stroke" />
            <path data-wire="bottom" vectorEffect="non-scaling-stroke" />
          </g>
          <g className="vla-wire-dash">
            <path
              data-wire="top"
              vectorEffect="non-scaling-stroke"
              className="vla-dash-a"
            />
            <path
              data-wire="bottom"
              vectorEffect="non-scaling-stroke"
              className="vla-dash-b"
            />
          </g>
        </g>
      </svg>

      {/* Demonstration — far-left anchor; prompt floats above it */}
      <div className="vla-node vla-input" ref={inputRef}>
        <div className="vla-prompt vla-detail" ref={promptRef}>
          {example.text}
          <span className="vla-grip" aria-hidden="true" />
        </div>
        <div className="vla-label vla-label-demo">Demonstration</div>
        <canvas className="vla-canvas" ref={demoRef} />
        <div className="vla-joints" ref={demoJointsRef}>
          <span>θ1 —</span>
          <span>θ2 —</span>
        </div>
      </div>

      {/* Vision Encoder — 4x4 patch grid over the demonstration's 16x16 view */}
      <div className="vla-node vla-vision" ref={visionCardRef}>
        <div className="vla-label">Vision Encoder</div>
        <canvas
          className="vla-vision-canvas vla-detail"
          ref={visionRef}
          width={176}
          height={176}
        />
      </div>

      {/* Language Encoder — the current prompt tokenized into chips */}
      <div className="vla-node vla-lang" ref={langCardRef}>
        <div className="vla-label">Language Encoder</div>
        <div className="vla-detail">
          <div className="vla-prompt-echo">&quot;{example.text}&quot;</div>
          <div className="vla-chip-row">
            {example.words.map((w, i) => (
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
                      style={{ width: pct(attentionWeight(w, p)) }}
                    />
                  </div>
                </div>
              </Fragment>
            ))}
          </div>
          <div className="vla-attn-caption">↳ live attention weight</div>
        </div>
      </div>

      {/* Rollout — repeated policy-driven grasp attempts */}
      <div className="vla-node vla-output" ref={outputCardRef}>
        <div className="vla-label">Rollout</div>
        <canvas className="vla-canvas vla-detail" ref={armRef} />
        <div className="vla-joints vla-detail" ref={armJointsRef}>
          <span>θ1 —</span>
          <span>θ2 —</span>
        </div>
      </div>

      {/* training control bar */}
      <div className="vla-bar">
        <div className="vla-status">
          <span className={`vla-dot${training ? " is-on" : ""}`} />
          <div className="vla-status-col">
            <span className="vla-status-text">{statusText}</span>
            {training && hud.samples > 0 && (
              <span className="vla-status-sub">
                {hud.samples.toLocaleString()} demos seen
              </span>
            )}
          </div>
        </div>
        <div className="vla-loss">
          <div className="vla-loss-label">MSE Loss</div>
          <canvas className="vla-loss-canvas" ref={lossRef} />
        </div>
        <div className="vla-loss-val">{hud.lossText}</div>
        <button className="vla-btn" onClick={onToggle} type="button">
          {training ? "Reset" : "Start Training"}
        </button>
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
