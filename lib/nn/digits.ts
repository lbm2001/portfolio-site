// Digit recognizer — the spirit of MNIST without bundling a dataset. Instead of
// a hard binary bitmap font, each sample is procedurally rendered like real
// MNIST: a vector glyph drawn with a thick round brush onto a 28×28 canvas,
// under a random affine warp (rotation / shear / scale / shift) + a slight blur,
// then read back as GRAYSCALE pixel intensities (0..1). So the strokes have soft
// anti-aliased edges and every sample varies — close to the real thing. A small
// softmax MLP classifies the 784-pixel vectors via categorical cross-entropy.

import type { NNTask } from "./types";
import { MLP, type Sample } from "./mlp";

const DIM = 28;
const N_IN = DIM * DIM;

// Each glyph is drawn in a normalized 0..1 box (x right, y down); the caller has
// already applied the pixel scale + random warp to the context. We only append
// path segments here — the caller strokes once.
function glyphPath(ctx: CanvasRenderingContext2D, d: number) {
  const q = (x1: number, y1: number, x2: number, y2: number) => ctx.quadraticCurveTo(x1, y1, x2, y2);
  const m = (x: number, y: number) => ctx.moveTo(x, y);
  const l = (x: number, y: number) => ctx.lineTo(x, y);
  const ring = (cx: number, cy: number, rx: number, ry: number) => {
    ctx.moveTo(cx + rx, cy);
    ctx.ellipse(cx, cy, rx, ry, 0, 0, Math.PI * 2);
  };
  switch (d) {
    case 0:
      ring(0.5, 0.5, 0.23, 0.35);
      break;
    case 1:
      m(0.37, 0.29); l(0.5, 0.18); l(0.5, 0.82);
      m(0.36, 0.82); l(0.64, 0.82);
      break;
    case 2:
      m(0.3, 0.32); q(0.34, 0.16, 0.5, 0.16); q(0.7, 0.16, 0.67, 0.36);
      q(0.64, 0.54, 0.3, 0.82); l(0.72, 0.82);
      break;
    case 3:
      m(0.31, 0.24); q(0.6, 0.11, 0.63, 0.32); q(0.64, 0.49, 0.45, 0.5);
      q(0.67, 0.5, 0.64, 0.69); q(0.6, 0.88, 0.31, 0.78);
      break;
    case 4:
      m(0.58, 0.16); l(0.27, 0.63); l(0.75, 0.63);
      m(0.61, 0.16); l(0.61, 0.84);
      break;
    case 5:
      m(0.66, 0.18); l(0.36, 0.18); l(0.34, 0.46);
      q(0.5, 0.39, 0.6, 0.5); q(0.69, 0.6, 0.6, 0.72); q(0.5, 0.86, 0.31, 0.78);
      break;
    case 6:
      m(0.61, 0.2); q(0.33, 0.2, 0.32, 0.55); q(0.31, 0.84, 0.5, 0.84);
      q(0.69, 0.84, 0.68, 0.65); q(0.67, 0.48, 0.5, 0.49); q(0.38, 0.5, 0.33, 0.6);
      break;
    case 7:
      m(0.3, 0.18); l(0.72, 0.18); l(0.44, 0.84);
      break;
    case 8:
      ring(0.5, 0.33, 0.16, 0.17);
      ring(0.5, 0.65, 0.19, 0.19);
      break;
    case 9:
      m(0.4, 0.82); q(0.67, 0.82, 0.68, 0.47); q(0.69, 0.16, 0.5, 0.16);
      q(0.31, 0.16, 0.32, 0.35); q(0.33, 0.52, 0.5, 0.51); q(0.62, 0.5, 0.67, 0.4);
      break;
  }
}

function oneHot(label: number): number[] {
  const v = new Array(10).fill(0);
  v[label] = 1;
  return v;
}

const rnd = (a: number, b: number) => a + Math.random() * (b - a);

export class DigitTask implements NNTask {
  readonly title = "Digit recognizer · 28×28 · backprop";
  readonly netSizes = [N_IN, 32, 10];
  readonly lossType = "ce" as const;
  w = 0;
  h = 0;
  net!: MLP;
  private oc: HTMLCanvasElement | null = null;
  private octx: CanvasRenderingContext2D | null = null;
  private sample!: { vec: number[]; label: number };
  private probs: number[] = new Array(10).fill(0.1);
  private stepCount = 0;

  setSize(w: number, h: number) {
    this.w = w;
    this.h = h;
  }

  reset() {
    this.net = new MLP(this.netSizes, ["tanh", "softmax"]);
    this.stepCount = 0;
    this.ensureCanvas();
    this.newSample();
  }

  // lazily created so the module stays SSR-safe (no `document` at import time)
  private ensureCanvas() {
    if (this.oc) return;
    this.oc = document.createElement("canvas");
    this.oc.width = DIM;
    this.oc.height = DIM;
    this.octx = this.oc.getContext("2d");
  }

  // Render one warped, blurred, grayscale sample of `label` → 784 intensities.
  private raster(label: number): number[] {
    const ctx = this.octx;
    const out = new Array(N_IN).fill(0);
    if (!ctx) return out;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.fillStyle = "#000";
    ctx.fillRect(0, 0, DIM, DIM);
    ctx.save();
    // random affine warp about the centre → natural per-sample variation
    ctx.translate(DIM / 2 + rnd(-3, 3), DIM / 2 + rnd(-3, 3));
    ctx.rotate(rnd(-0.26, 0.26));
    ctx.transform(1, 0, rnd(-0.32, 0.32), 1, 0, 0); // horizontal shear
    const s = rnd(0.82, 1.12) * DIM;
    ctx.scale(s, s);
    ctx.translate(-0.5, -0.5);
    ctx.strokeStyle = "#fff";
    ctx.lineJoin = "round";
    ctx.lineCap = "round";
    ctx.lineWidth = rnd(0.085, 0.13); // in normalized units → ~2.5–3.5px stroke
    ctx.filter = "blur(0.6px)"; // soft anti-aliased edges, MNIST-style
    ctx.beginPath();
    glyphPath(ctx, label);
    ctx.stroke();
    ctx.restore();
    ctx.filter = "none";
    const data = ctx.getImageData(0, 0, DIM, DIM).data;
    for (let i = 0; i < N_IN; i++) out[i] = data[i * 4] / 255; // white-on-black → R channel
    return out;
  }

  private newSample() {
    const label = (Math.random() * 10) | 0;
    this.sample = { vec: this.raster(label), label };
  }

  currentSample(): Sample {
    return { x: this.sample.vec, y: oneHot(this.sample.label) };
  }

  trainStep(): number {
    const batch: Sample[] = [];
    for (let i = 0; i < 16; i++) {
      const label = (Math.random() * 10) | 0;
      batch.push({ x: this.raster(label), y: oneHot(label) });
    }
    const loss = this.net.trainStep(batch, 0.3, this.lossType);
    this.stepCount++;
    if (this.stepCount % 12 === 0) this.newSample();
    this.probs = this.net.predict(this.sample.vec);
    return loss;
  }

  converged(loss: number, step: number): boolean {
    return loss < 0.1 || step > 800;
  }

  draw(ctx: CanvasRenderingContext2D) {
    if (!this.net) return;
    const cell = Math.min((this.w * 0.4) / DIM, (this.h * 0.9) / DIM);
    const ox = 8;
    const oy = (this.h - cell * DIM) / 2;
    // grayscale digit (soft, anti-aliased) rather than hard on/off cells
    for (let r = 0; r < DIM; r++) {
      for (let c = 0; c < DIM; c++) {
        const v = this.sample.vec[r * DIM + c];
        if (v <= 0.02) continue;
        ctx.fillStyle = `rgba(17,17,17,${v})`;
        ctx.fillRect(ox + c * cell, oy + r * cell, cell, cell);
      }
    }
    const bx = ox + cell * DIM + 14;
    const barAreaW = Math.max(1, this.w - bx - 6);
    const barH = (this.h - 4) / 10;
    for (let d = 0; d < 10; d++) {
      const p = this.probs[d];
      const by = 2 + d * barH;
      ctx.fillStyle = d === this.sample.label ? "rgba(225,45,26,0.9)" : "rgba(17,17,17,0.55)";
      ctx.fillRect(bx, by, Math.max(1, barAreaW * p), Math.max(1, barH - 1.5));
    }
  }
}
