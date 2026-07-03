// Classic "neural net hello world": two Gaussian point clouds that ARE linearly
// separable, and a small MLP that finds a dividing line via plain gradient
// descent. The decision boundary is rendered as a soft heatmap that sharpens
// from a fuzzy blob into a clean split as training progresses.

import type { NNTask } from "./types";
import { MLP, type Sample } from "./mlp";

const DATA_X0 = -1.3;
const DATA_X1 = 2.3;
const DATA_Y0 = -1.6;
const DATA_Y1 = 1.3;

// two well-separated blobs → a single straight boundary suffices
const BLOB_A = { mx: -0.1, my: -0.7, sd: 0.22 };
const BLOB_B = { mx: 1.1, my: 0.6, sd: 0.22 };

// standard normal via Box–Muller
function randn(): number {
  const u = Math.random() || 1e-9;
  const v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

function makeBlobs(n: number): Sample[] {
  const data: Sample[] = [];
  for (let i = 0; i < n; i++) {
    const b = i % 2 === 0 ? BLOB_A : BLOB_B;
    data.push({
      x: [b.mx + randn() * b.sd, b.my + randn() * b.sd],
      y: [i % 2 === 0 ? 0 : 1],
    });
  }
  return data;
}

export class ClassificationTask implements NNTask {
  readonly title = "Classifier · linear split · backprop";
  readonly netSizes = [2, 8, 8, 1];
  readonly lossType = "bce" as const;
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
    this.net = new MLP(this.netSizes, ["tanh", "tanh", "sigmoid"]);
    this.data = makeBlobs(120);
    this.probe = this.data[0];
    this.stepCount = 0;
  }

  trainStep(): number {
    const batch: Sample[] = [];
    for (let i = 0; i < 16; i++) batch.push(this.data[(Math.random() * this.data.length) | 0]);
    // low LR on purpose: the blobs are linearly separable, so a high rate solves
    // them almost instantly — this paces the boundary sharpening to ~20s so it's
    // watchable before the task resets with a fresh cloud.
    const loss = this.net.trainStep(batch, 0.06, this.lossType);
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
    return loss < 0.03 || step > 600;
  }

  private toPx(px: number, py: number): [number, number] {
    const x = ((px - DATA_X0) / (DATA_X1 - DATA_X0)) * this.w;
    const y = this.h - ((py - DATA_Y0) / (DATA_Y1 - DATA_Y0)) * this.h;
    return [x, y];
  }

  private toData(cx: number, cy: number): [number, number] {
    const px = DATA_X0 + (cx / this.w) * (DATA_X1 - DATA_X0);
    const py = DATA_Y0 + ((this.h - cy) / this.h) * (DATA_Y1 - DATA_Y0);
    return [px, py];
  }

  draw(ctx: CanvasRenderingContext2D) {
    if (!this.net) return;
    const cell = 6;
    for (let cy = 0; cy < this.h; cy += cell) {
      for (let cx = 0; cx < this.w; cx += cell) {
        const [px, py] = this.toData(cx + cell / 2, cy + cell / 2);
        const p = this.net.predict([px, py])[0];
        const alpha = 0.08 + 0.14 * Math.abs(p - 0.5) * 2;
        ctx.fillStyle = p > 0.5 ? `rgba(225,45,26,${alpha})` : `rgba(17,17,17,${alpha})`;
        ctx.fillRect(cx, cy, cell, cell);
      }
    }
    for (const { x, y } of this.data) {
      const [cx, cy] = this.toPx(x[0], x[1]);
      ctx.fillStyle = y[0] === 1 ? "#E12D1A" : "#111";
      ctx.beginPath();
      ctx.arc(cx, cy, 2, 0, 6.283);
      ctx.fill();
    }
  }
}
