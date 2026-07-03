// Curve fitting: an MLP regresses a noisy 1D function. The simplest of the
// four hello-worlds — a scatter of data points and a curve that bends from a
// flat random line into the fit.

import type { NNTask } from "./types";
import { MLP, type Sample } from "./mlp";

// A smooth, low-frequency target the small tanh MLP can actually fit well.
// (An earlier version had a sin(7x) term that the net couldn't resolve —
// spectral bias — so the curve never matched the points.)
function targetFn(x: number): number {
  return 0.5 * Math.sin(2.2 * x) + 0.14 * Math.sin(3.6 * x + 0.7);
}

const N_POINTS = 40;

export class CurveFitTask implements NNTask {
  readonly title = "Regression · curve fit · backprop";
  readonly netSizes = [1, 16, 16, 1];
  readonly lossType = "mse" as const;
  w = 0;
  h = 0;
  net!: MLP;
  private data: Sample[] = [];
  private probe!: Sample;
  private stepCount = 0;

  setSize(w: number, h: number) {
    this.w = w;
    this.h = h;
  }

  reset() {
    this.net = new MLP(this.netSizes, ["tanh", "tanh", "linear"]);
    this.data = [];
    for (let i = 0; i < N_POINTS; i++) {
      const x = -1 + (2 * i) / (N_POINTS - 1);
      const noise = (Math.random() - 0.5) * 0.08;
      this.data.push({ x: [x], y: [targetFn(x) + noise] });
    }
    this.probe = this.data[0];
    this.stepCount = 0;
  }

  trainStep(): number {
    const loss = this.net.trainStep(this.data, 0.5, this.lossType);
    this.stepCount++;
    if (this.stepCount % 40 === 0) {
      this.probe = this.data[(Math.random() * this.data.length) | 0];
    }
    return loss;
  }

  currentSample(): Sample {
    return this.probe;
  }

  converged(loss: number, step: number): boolean {
    return loss < 0.003 || step > 500;
  }

  private toPx(x: number, y: number): [number, number] {
    const cx = 6 + ((x + 1) / 2) * (this.w - 12);
    const cy = this.h * 0.5 - y * this.h * 0.72;
    return [cx, cy];
  }

  draw(ctx: CanvasRenderingContext2D) {
    if (!this.net) return;
    for (const { x, y } of this.data) {
      const [cx, cy] = this.toPx(x[0], y[0]);
      ctx.fillStyle = "rgba(17,17,17,0.35)";
      ctx.beginPath();
      ctx.arc(cx, cy, 1.6, 0, 6.283);
      ctx.fill();
    }
    ctx.beginPath();
    const steps = 60;
    for (let i = 0; i <= steps; i++) {
      const x = -1 + (2 * i) / steps;
      const yhat = this.net.predict([x])[0];
      const [cx, cy] = this.toPx(x, yhat);
      if (i === 0) ctx.moveTo(cx, cy);
      else ctx.lineTo(cx, cy);
    }
    ctx.strokeStyle = "#E12D1A";
    ctx.lineWidth = 1.6;
    ctx.stroke();
  }
}
