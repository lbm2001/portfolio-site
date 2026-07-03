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
const TRAIN_STEPS_PER_SEC = 260; // background ARS training — deliberately slow so
//                                  convergence is watchable, not instant
const TRAIN_CAP_PER_FRAME = 40; // don't binge-train after a stall / hidden tab

const pad = (n: number) => String(Math.max(0, Math.round(n))).padStart(3, "0");

export default function MiniDemo({
  make,
  corner,
  displayDisturb = 0,
}: {
  make: EnvFactory;
  corner: Corner;
  displayDisturb?: number;
}) {
  const simRef = useRef<HTMLCanvasElement>(null);
  const chartRef = useRef<HTMLCanvasElement>(null);
  const statsRef = useRef<HTMLDivElement>(null);
  const [title] = useState(() => make().title);
  const [running, setRunning] = useState(false);
  const runningRef = useRef(false);
  const { ref: dragRef, wasDraggedRef } = useFloatDrag(FLOAT_PARAMS[corner]);

  const toggle = () => {
    if (wasDraggedRef.current) return;
    runningRef.current = !runningRef.current;
    setRunning(runningRef.current);
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
    const trainer = new ARSTrainer(make);
    const first = sizeCanvas(sim);
    env.setSize(first.w, first.h);
    trainer.setSize(first.w, first.h);
    env.reset();

    let raf = 0;
    let acc = 0; // real-time display-step accumulator (ms)
    let trainAcc = 0; // training-step accumulator (ms)
    let lastT = performance.now();
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
          statsRef.current.textContent = `episode ${pad(trainer.episode)}   return ${pad(
            trainer.genMean
          )}   gen ${trainer.run}`;
        }
      }
      raf = requestAnimationFrame(frame);
    };
    raf = requestAnimationFrame(frame);
    return () => cancelAnimationFrame(raf);
  }, [make, displayDisturb]);

  return (
    <div
      ref={dragRef}
      className={`mini mini-${corner}${running ? " is-running" : ""}`}
      onClick={toggle}
    >
      <canvas ref={simRef} className="mini-sim" />
      <div className="mini-meta">
        <div className={running ? "mini-training" : "mini-training mini-paused"}>
          {running ? "● TRAINING" : "○ CLICK TO TRAIN"}
        </div>
        <div className="mini-title">{title}</div>
        <div ref={statsRef} className="mini-stats">
          episode 000&nbsp;&nbsp;&nbsp;return 000&nbsp;&nbsp;&nbsp;gen 1
        </div>
      </div>
      <canvas ref={chartRef} className="mini-chart" />
    </div>
  );
}
