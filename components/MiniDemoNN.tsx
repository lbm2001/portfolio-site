"use client";

import { useEffect, useRef, useState } from "react";
import { NNTrainer } from "@/lib/nn/trainer";
import { drawSparkline } from "@/lib/render/sparkline";
import { sizeCanvas } from "@/lib/render/canvas";
import { useFloatDrag, FLOAT_PARAMS, type Corner } from "@/components/useFloatDrag";
import type { NNTaskFactory } from "@/lib/nn/types";

const TRAIN_STEPS_PER_SEC = 5; // each step already covers a mini-batch; kept slow so the loss
//                                curve ticks down at a watchable pace
const TRAIN_CAP_PER_FRAME = 4;

const pad = (n: number) => String(Math.max(0, Math.round(n))).padStart(3, "0");

export default function MiniDemoNN({
  make,
  corner,
}: {
  make: NNTaskFactory;
  corner: Corner;
}) {
  const simRef = useRef<HTMLCanvasElement>(null);
  const chartRef = useRef<HTMLCanvasElement>(null);
  const statsRef = useRef<HTMLDivElement>(null);
  const [title] = useState(() => make().title);
  const [running, setRunning] = useState(false);
  const runningRef = useRef(false);
  const { ref: dragRef, wasDraggedRef } = useFloatDrag(FLOAT_PARAMS[corner]);

  const toggleRun = () => {
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

    const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

    const trainer = new NNTrainer(make);
    const first = sizeCanvas(sim);
    trainer.setSize(first.w, first.h);
    trainer.task.reset();

    let raf = 0;
    let trainAcc = 0;
    let lastT = performance.now();
    const trainDur = 1000 / TRAIN_STEPS_PER_SEC;

    const frame = () => {
      const { w, h } = sizeCanvas(sim);
      if (w > 0 && (Math.round(w) !== Math.round(trainer.task.w) || Math.round(h) !== Math.round(trainer.task.h))) {
        trainer.setSize(w, h);
        trainer.task.reset();
      }
      if (trainer.ready()) {
        const now = performance.now();
        let elapsed = now - lastT;
        lastT = now;
        if (elapsed > 250) elapsed = trainDur;

        if (runningRef.current && !reduceMotion) {
          trainAcc += elapsed;
          let tsteps = 0;
          while (trainAcc >= trainDur && tsteps < TRAIN_CAP_PER_FRAME) {
            trainAcc -= trainDur;
            tsteps++;
          }
          if (tsteps > 0) trainer.advance(tsteps);
        }

        ctx.clearRect(0, 0, w, h);
        trainer.draw(ctx);
        if (chart) {
          const c = sizeCanvas(chart);
          const cctx = chart.getContext("2d");
          if (cctx && c.w > 0) drawSparkline(cctx, c.w, c.h, trainer.history, "LOSS / STEP");
        }
        if (statsRef.current) {
          statsRef.current.textContent = `step ${pad(trainer.step)}   loss ${trainer.lastLoss.toFixed(
            3
          )}   run ${trainer.run}`;
        }
      }
      raf = requestAnimationFrame(frame);
    };
    raf = requestAnimationFrame(frame);
    return () => cancelAnimationFrame(raf);
  }, [make]);

  return (
    <div
      ref={dragRef}
      className={`mini mini-${corner}${running ? " is-running" : ""}`}
      onClick={toggleRun}
    >
      <canvas ref={simRef} className="mini-sim" />
      <div className="mini-meta">
        <div className={running ? "mini-training" : "mini-training mini-paused"}>
          {running ? "● TRAINING" : "○ CLICK TO TRAIN"}
        </div>
        <div className="mini-title">{title}</div>
        <div ref={statsRef} className="mini-stats">
          step 000&nbsp;&nbsp;&nbsp;loss 0.000&nbsp;&nbsp;&nbsp;run 1
        </div>
      </div>
      <canvas ref={chartRef} className="mini-chart" />
    </div>
  );
}
