"use client";

import { useEffect, useRef, useState } from "react";
import { NNTrainer } from "@/lib/nn/trainer";
import { drawSparkline } from "@/lib/render/sparkline";
import { sizeCanvas } from "@/lib/render/canvas";
import { useFloatDrag, FLOAT_PARAMS, type Corner } from "@/components/useFloatDrag";
import type { NNTaskFactory } from "@/lib/nn/types";

const TRAIN_STEPS_PER_SEC = 24; // each step already covers a mini-batch. Paced so a
//                                 full learn (blob split / curve fit / digit lock-in)
//                                 is clearly visible within ~5-10s of clicking.
const TRAIN_CAP_PER_FRAME = 12;
const HUD_EVERY = 10; // only refresh the step/loss readout every N steps so the numbers
//                       tick at a readable pace instead of flickering every frame

const pad = (n: number) => String(Math.max(0, Math.round(n))).padStart(3, "0");

export default function MiniDemoNN({
  make,
  corner,
  dropSmall = false,
  smallSlot,
}: {
  make: NNTaskFactory;
  corner: Corner;
  // Small-mobile (≤600px) layout only: dropSmall hides this panel there (one
  // random NN demo is dropped each load); smallSlot places the survivors in the
  // compact 2-up band instead of their normal ring corner. See globals.css.
  dropSmall?: boolean;
  smallSlot?: string;
}) {
  const simRef = useRef<HTMLCanvasElement>(null);
  const chartRef = useRef<HTMLCanvasElement>(null);
  const statsRef = useRef<HTMLDivElement>(null);
  const trainerRef = useRef<NNTrainer | null>(null);
  const [title] = useState(() => make().title);
  const [running, setRunning] = useState(false);
  const runningRef = useRef(false);
  const { ref: dragRef, wasDraggedRef, revealed } = useFloatDrag(FLOAT_PARAMS[corner]);

  const toggleRun = () => {
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

    const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

    const trainer = new NNTrainer(make);
    trainerRef.current = trainer;
    const first = sizeCanvas(sim);
    trainer.setSize(first.w, first.h);
    trainer.task.reset();

    let raf = 0;
    let trainAcc = 0;
    let lastT = performance.now();
    let lastHudStep = -HUD_EVERY; // force a first HUD paint
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
          const st = trainer.step;
          if (st < lastHudStep || st - lastHudStep >= HUD_EVERY) {
            lastHudStep = st;
            statsRef.current.textContent = `Step ${pad(st)}   Loss ${trainer.lastLoss.toFixed(
              3
            )}`;
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
  }, [make]);

  return (
    <div
      ref={dragRef}
      className={`mini mini-${corner}${running ? " is-running" : ""}${revealed ? "" : " mini-pre"}${dropSmall ? " mini-drop-sm" : ""}${smallSlot ? ` mini-sm-${smallSlot}` : ""}`}
      onClick={toggleRun}
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
            Step 000&nbsp;&nbsp;&nbsp;Loss 0.000
          </div>
        </div>
        <canvas ref={chartRef} className="mini-chart" />
      </div>
    </div>
  );
}
