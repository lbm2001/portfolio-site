// Classic "neural net hello world": two Gaussian point clouds that ARE linearly
// separable, and a small MLP that finds a dividing line via plain gradient
// descent.

import type { NNTask } from "./types";
import { MLP, type Sample } from "./mlp";

const DATA_X0 = -1.3;
const DATA_X1 = 2.3;
const DATA_Y0 = -1.6;
const DATA_Y1 = 1.3;

// two well-separated blobs → a single straight boundary suffices
const BLOB_A = { mx: -0.1, my: -0.7, sd: 0.22 };
const BLOB_B = { mx: 1.1, my: 0.6, sd: 0.22 };
// re-rolled each reset() so the cloud isn't identical every load — kept small
// enough that the ~1.77-unit gap between centers (≈8sd) never collapses below
// a cleanly separable margin.
const CENTER_JITTER = 0.18;

// standard normal via Box–Muller
function randn(): number {
  const u = Math.random() || 1e-9;
  const v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

function jitteredBlob(b: { mx: number; my: number; sd: number }) {
  return {
    mx: b.mx + (Math.random() - 0.5) * 2 * CENTER_JITTER,
    my: b.my + (Math.random() - 0.5) * 2 * CENTER_JITTER,
    sd: b.sd,
  };
}

function makeBlobs(n: number): Sample[] {
  const a = jitteredBlob(BLOB_A);
  const b = jitteredBlob(BLOB_B);
  const data: Sample[] = [];
  for (let i = 0; i < n; i++) {
    const blob = i % 2 === 0 ? a : b;
    data.push({
      x: [blob.mx + randn() * blob.sd, blob.my + randn() * blob.sd],
      y: [i % 2 === 0 ? 0 : 1],
    });
  }
  return data;
}

export class ClassificationTask implements NNTask {
  readonly title = "Classification · Binary Cross-Entropy Loss";
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
    // paced so the boundary visibly sweeps from a fuzzy blob into a clean split
    // over ~6s at the panel's step rate (not a sub-second snap), then resets.
    const loss = this.net.trainStep(batch, 0.05, this.lossType);
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
    return loss < 0.03 || step > 220;
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

  // probability of class 1 at a canvas pixel
  private pAt(cx: number, cy: number): number {
    const [px, py] = this.toData(cx, cy);
    return this.net.predict([px, py])[0];
  }

  draw(ctx: CanvasRenderingContext2D) {
    if (!this.net) return;
    // solid decision boundary: the p = 0.5 iso-line via marching squares
    const gs = 9;
    ctx.strokeStyle = "#E12D1A";
    ctx.lineWidth = 2;
    ctx.lineCap = "round";
    ctx.beginPath();
    for (let y = 0; y + gs <= this.h; y += gs) {
      for (let x = 0; x + gs <= this.w; x += gs) {
        const a = this.pAt(x, y) - 0.5; // top-left
        const b = this.pAt(x + gs, y) - 0.5; // top-right
        const c = this.pAt(x + gs, y + gs) - 0.5; // bottom-right
        const d = this.pAt(x, y + gs) - 0.5; // bottom-left
        const pts: [number, number][] = [];
        if (a * b < 0) pts.push([x + (gs * -a) / (b - a), y]);
        if (b * c < 0) pts.push([x + gs, y + (gs * -b) / (c - b)]);
        if (d * c < 0) pts.push([x + (gs * -d) / (c - d), y + gs]);
        if (a * d < 0) pts.push([x, y + (gs * -a) / (d - a)]);
        for (let i = 0; i + 1 < pts.length; i += 2) {
          ctx.moveTo(pts[i][0], pts[i][1]);
          ctx.lineTo(pts[i + 1][0], pts[i + 1][1]);
        }
      }
    }
    ctx.stroke();

    // data points
    for (const { x, y } of this.data) {
      const [cx, cy] = this.toPx(x[0], x[1]);
      ctx.fillStyle = y[0] === 1 ? "#E12D1A" : "#111";
      ctx.beginPath();
      ctx.arc(cx, cy, 2, 0, 6.283);
      ctx.fill();
    }
  }
}
