"use client";

import { useEffect, useRef, useState } from "react";
import { ARSTrainer } from "@/lib/rl/ars";
import { drawSparkline } from "@/lib/render/sparkline";
import { sizeCanvas } from "@/lib/render/canvas";
import { useFloatDrag, FLOAT_PARAMS, type Corner } from "@/components/useFloatDrag";
import type { EnvFactory } from "@/lib/rl/types";

const STEP_HZ = 48; // rendered-agent steps per second (real-time)
const STEP_MS = 1000 / STEP_HZ;
const MAX_STEPS_PER_FRAME = 4;
const TRAIN_STEPS_PER_SEC = 1600; // background ARS training. Fast enough that several
//                                   generations complete within ~5-10s, so the rendered
//                                   agent visibly improves while a viewer watches.
const TRAIN_CAP_PER_FRAME = 240; // don't binge-train after a stall / hidden tab
const HUD_EVERY = 10; // only refresh the episode/return readout every N episodes so the
//                       numbers tick at a readable pace instead of flickering every frame

const pad = (n: number) => String(Math.max(0, Math.round(n))).padStart(3, "0");

export default function MiniDemo({
  make,
  corner,
  displayDisturb = 0,
  showFall = false,
  variant = "float",
}: {
  make: EnvFactory;
  corner: Corner;
  displayDisturb?: number;
  showFall?: boolean;
  // "float" is the draggable ring panel (desktop); "static" is the single
  // in-flow demo shown below the name on mobile — no drag, HUD always visible.
  variant?: "float" | "static";
}) {
  const simRef = useRef<HTMLCanvasElement>(null);
  const chartRef = useRef<HTMLCanvasElement>(null);
  const statsRef = useRef<HTMLDivElement>(null);
  const trainerRef = useRef<ARSTrainer | null>(null);
  const [title] = useState(() => make().title);
  const [running, setRunning] = useState(false);
  const runningRef = useRef(false);
  const isStatic = variant === "static";
  const { ref: dragRef, wasDraggedRef, revealed } = useFloatDrag(FLOAT_PARAMS[corner]);

  const toggle = () => {
    if (wasDraggedRef.current) return;
    runningRef.current = !runningRef.current;
    setRunning(runningRef.current);
  };

  const reset = (e: React.MouseEvent) => {
    e.stopPropagation();
    trainerRef.current?.manualReset();
  };

  useEffect(() => {
    const sim = simRef.current;
    const chart = chartRef.current;
    if (!sim) return;
    const ctx = sim.getContext("2d");
    if (!ctx) return;

    const reduceMotion = window.matchMedia(
      "(prefers-reduced-motion: reduce)"
    ).matches;

    const env = make();
    env.disturbAmp = displayDisturb;
    env.showFall = showFall;
    if (isStatic) {
      // perched-on-the-name mobile mini: drop the agent to the canvas floor and
      // hide the env's own ground line so the page (the name) reads as the floor
      env.bare = true;
      env.groundFrac = 0.9;
    }
    const trainer = new ARSTrainer(make);
    trainerRef.current = trainer;
    const first = sizeCanvas(sim);
    env.setSize(first.w, first.h);
    trainer.setSize(first.w, first.h);
    env.reset();

    let raf = 0;
    let acc = 0; // real-time display-step accumulator (ms)
    let trainAcc = 0; // training-step accumulator (ms)
    let lastT = performance.now();
    let lastHudEp = -HUD_EVERY; // force a first HUD paint
    const trainDur = 1000 / TRAIN_STEPS_PER_SEC;

    const frame = () => {
      const { w, h } = sizeCanvas(sim);
      if (w > 0 && (Math.round(w) !== Math.round(env.w) || Math.round(h) !== Math.round(env.h))) {
        env.setSize(w, h);
        trainer.setSize(w, h);
        env.reset();
      }
      if (env.w > 0) {
        const now = performance.now();
        let elapsed = now - lastT;
        lastT = now;
        if (elapsed > 250) elapsed = STEP_MS;

        if (runningRef.current && !reduceMotion) {
          // background training, paced to TRAIN_STEPS_PER_SEC
          trainAcc += elapsed;
          let tsteps = 0;
          while (trainAcc >= trainDur && tsteps < TRAIN_CAP_PER_FRAME) {
            trainAcc -= trainDur;
            tsteps++;
          }
          if (tsteps > 0) trainer.advance(tsteps);
          // real-time display of the current mean policy
          acc += elapsed;
          let budget = 0;
          while (acc >= STEP_MS && budget < MAX_STEPS_PER_FRAME) {
            acc -= STEP_MS;
            budget++;
          }
          for (let s = 0; s < budget; s++) {
            const { done } = env.step(trainer.actMean(env.getObs()));
            if (done) env.reset();
          }
        } else {
          acc = 0;
          trainAcc = 0;
        }

        ctx.clearRect(0, 0, w, h);
        env.draw(ctx);
        if (chart) {
          const c = sizeCanvas(chart);
          const cctx = chart.getContext("2d");
          if (cctx && c.w > 0) drawSparkline(cctx, c.w, c.h, trainer.history);
        }
        if (statsRef.current) {
          const ep = trainer.episode;
          if (ep < lastHudEp || ep - lastHudEp >= HUD_EVERY) {
            lastHudEp = ep;
            statsRef.current.textContent = `Episode ${pad(ep)}   Return ${pad(
              trainer.genMean
            )}   Gen ${trainer.run}`;
          }
        }
      }
      raf = requestAnimationFrame(frame);
    };
    raf = requestAnimationFrame(frame);
    return () => {
      cancelAnimationFrame(raf);
      trainerRef.current = null;
    };
  }, [make, displayDisturb, showFall, isStatic]);

  return (
    <div
      ref={isStatic ? undefined : dragRef}
      className={
        isStatic
          ? `mini-mobile${running ? " is-running" : ""}`
          : `mini mini-${corner}${running ? " is-running" : ""}${revealed ? "" : " mini-pre"}`
      }
      onClick={toggle}
    >
      <canvas ref={simRef} className="mini-sim" />
      <div className="mini-hud">
        <div className="mini-meta">
          <div className="mini-row">
            <div className={running ? "mini-training" : "mini-training mini-paused"}>
              {running ? (
                "● TRAINING"
              ) : (
                <>
                  <span className="verb-click">○ CLICK FOR LIVE TRAINING</span>
                  <span className="verb-tap">○ TAP FOR LIVE TRAINING</span>
                </>
              )}
            </div>
            <button
              type="button"
              className="mini-reset"
              onClick={reset}
              aria-label="Reset training"
              title="Reset training"
            >
              ↺
            </button>
          </div>
          <div className="mini-title">{title}</div>
          <div ref={statsRef} className="mini-stats">
            Episode 000&nbsp;&nbsp;&nbsp;Return 000&nbsp;&nbsp;&nbsp;Gen 1
          </div>
        </div>
        <canvas ref={chartRef} className="mini-chart" />
      </div>
    </div>
  );
}
