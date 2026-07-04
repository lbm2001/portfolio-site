// Curve fitting: an MLP regresses a noisy 1D function. The simplest of the
// four hello-worlds — a scatter of data points and a curve that bends from a
// flat random line into the fit.

import type { NNTask } from "./types";
import { MLP, type Sample } from "./mlp";

// A smooth, low-frequency target the small tanh MLP can actually fit well.
// (An earlier version had a sin(7x) term that the net couldn't resolve —
// spectral bias — so the curve never matched the points.)
// Shape params are re-rolled each reset() (within a low-frequency range) so
// the curve isn't identical every load or click of the reset button.
interface CurveParams {
  a1: number;
  f1: number;
  a2: number;
  f2: number;
  phase: number;
}

function targetFn(x: number, p: CurveParams): number {
  return p.a1 * Math.sin(p.f1 * x) + p.a2 * Math.sin(p.f2 * x + p.phase);
}

function randomParams(): CurveParams {
  return {
    a1: 0.4 + Math.random() * 0.2,
    f1: 1.8 + Math.random() * 0.8,
    a2: 0.08 + Math.random() * 0.1,
    f2: 3.0 + Math.random() * 1.2,
    phase: Math.random() * Math.PI * 2,
  };
}

const N_POINTS = 40;
const NOISE_AMP = 0.12;

export class CurveFitTask implements NNTask {
  readonly title = "Regression · Mean Squared Error Loss";
  // deliberately over-parameterized for a 1D fit → the curve wiggles through the
  // noisy points (visible overfitting) rather than settling on a smooth trend
  readonly netSizes = [1, 32, 32, 1];
  readonly lossType = "mse" as const;
  w = 0;
  h = 0;
  net!: MLP;
  private data: Sample[] = [];
  private probe!: Sample;
  private stepCount = 0;
  private params!: CurveParams;

  setSize(w: number, h: number) {
    this.w = w;
    this.h = h;
  }

  reset() {
    this.net = new MLP(this.netSizes, ["tanh", "tanh", "linear"]);
    this.params = randomParams();
    this.data = [];
    for (let i = 0; i < N_POINTS; i++) {
      const x = -1 + (2 * i) / (N_POINTS - 1);
      const noise = (Math.random() - 0.5) * NOISE_AMP;
      this.data.push({ x: [x], y: [targetFn(x, this.params) + noise] });
    }
    this.probe = this.data[0];
    this.stepCount = 0;
  }

  trainStep(): number {
    // Low LR for two reasons: the wide net's gradients are large (0.5 → NaN), and
    // a gentle rate lets the curve visibly bend from a flat line into the fit over
    // ~5-6s rather than snapping in under a second.
    const loss = this.net.trainStep(this.data, 0.035, this.lossType);
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
    return loss < 0.004 || step > 320;
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
