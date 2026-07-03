// Digit recognizer: the spirit of MNIST without bundling a dataset. Ten
// hand-authored 10x10 bitmap digits, corrupted with pixel noise on every draw,
// classified by a small softmax MLP trained with categorical cross-entropy.

import type { NNTask } from "./types";
import { MLP, type Sample } from "./mlp";

const DIGIT_ROWS = 10;
const DIGIT_COLS = 10;

// Bold ~2px strokes so the glyphs stay legible at panel size (thin 1px strokes
// read as broken, especially once pixel noise is added).
// prettier-ignore
const DIGIT_FONT: string[] = [
  "..######.." + ".##....##." + "##......##" + "##......##" + "##......##" +
  "##......##" + "##......##" + "##......##" + ".##....##." + "..######..", // 0
  "...###...." + "..####...." + ".##.##...." + "....##...." + "....##...." +
  "....##...." + "....##...." + "....##...." + "..######.." + "..######..", // 1
  ".######..." + "##....##.." + "......##.." + ".....##..." + "....##...." +
  "...##....." + "..##......" + ".##......." + "########.." + "########..", // 2
  ".######..." + "##....##.." + "......##.." + "...####..." + "...####..." +
  "......##.." + "......##.." + "##....##.." + ".######..." + ".######...", // 3
  "....###..." + "...####..." + "..##.##..." + ".##..##..." + "##...##..." +
  "#########." + "#########." + ".....##..." + ".....##..." + ".....##...", // 4
  "########.." + "########.." + "##........" + "##........" + "#######..." +
  "......##.." + "......##.." + "##....##.." + ".######..." + ".######...", // 5
  "..#####..." + ".##...##.." + "##........" + "##........" + "#######..." +
  "###...##.." + "##.....##." + "##.....##." + ".##...##.." + "..#####...", // 6
  "#########." + "#########." + ".......##." + "......##.." + ".....##..." +
  "....##...." + "...##....." + "...##....." + "...##....." + "...##.....", // 7
  "..#####..." + ".##...##.." + "##.....##." + ".##...##.." + "..#####..." +
  ".##...##.." + "##.....##." + "##.....##." + ".##...##.." + "..#####...", // 8
  "..#####..." + ".##...##.." + "##.....##." + "##.....##." + ".##...###." +
  "..#######." + "......##.." + ".....##..." + "....##...." + "..###.....", // 9
];

function digitToVec(pattern: string): number[] {
  return pattern.split("").map((c) => (c === "#" ? 1 : 0));
}

const DIGIT_VECS = DIGIT_FONT.map(digitToVec);

function noisyDigit(label: number, flipProb = 0.03): number[] {
  return DIGIT_VECS[label].map((v) => (Math.random() < flipProb ? 1 - v : v));
}

function oneHot(label: number): number[] {
  const v = new Array(10).fill(0);
  v[label] = 1;
  return v;
}

export class DigitTask implements NNTask {
  readonly title = "Digit recognizer · 10×10 · backprop";
  readonly netSizes = [DIGIT_ROWS * DIGIT_COLS, 24, 10];
  readonly lossType = "ce" as const;
  w = 0;
  h = 0;
  net!: MLP;
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
    this.newSample();
  }

  currentSample(): Sample {
    return { x: this.sample.vec, y: oneHot(this.sample.label) };
  }

  private newSample() {
    const label = (Math.random() * 10) | 0;
    this.sample = { vec: noisyDigit(label), label };
  }

  trainStep(): number {
    const batch: Sample[] = [];
    for (let i = 0; i < 16; i++) {
      const label = (Math.random() * 10) | 0;
      batch.push({ x: noisyDigit(label), y: oneHot(label) });
    }
    const loss = this.net.trainStep(batch, 0.12, this.lossType);
    this.stepCount++;
    if (this.stepCount % 20 === 0) this.newSample();
    this.probs = this.net.predict(this.sample.vec);
    return loss;
  }

  converged(loss: number, step: number): boolean {
    return loss < 0.05 || step > 700;
  }

  draw(ctx: CanvasRenderingContext2D) {
    if (!this.net) return;
    const cell = Math.min((this.w * 0.34) / DIGIT_COLS, (this.h * 0.8) / DIGIT_ROWS);
    const ox = 8;
    const oy = (this.h - cell * DIGIT_ROWS) / 2;
    for (let r = 0; r < DIGIT_ROWS; r++) {
      for (let c = 0; c < DIGIT_COLS; c++) {
        const v = this.sample.vec[r * DIGIT_COLS + c];
        ctx.fillStyle = v ? "#111" : "rgba(17,17,17,0.06)";
        ctx.fillRect(ox + c * cell, oy + r * cell, cell - 1, cell - 1);
      }
    }
    const bx = ox + cell * DIGIT_COLS + 14;
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
