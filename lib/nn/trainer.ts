// Mirrors ARSTrainer's shape (run/history/advance) so the two hero modes read
// as siblings, but there's no train/display split here — supervised learning
// has no exploration noise to hide, so we just draw the current weights.

import type { NNTask } from "./types";

const HISTORY = 60; // loss samples kept for the sparkline
const RECORD_EVERY = 6; // thin the per-step loss into a smoother trace

export class NNTrainer {
  task: NNTask;
  step = 0;
  run = 1; // how many times training has restarted from scratch
  lastLoss = 0;
  history: number[] = [];

  private acc = 0;

  constructor(make: () => NNTask) {
    this.task = make();
  }

  setSize(w: number, h: number) {
    this.task.setSize(w, h);
  }

  ready() {
    return this.task.w > 0;
  }

  manualReset() {
    this.task.reset();
    this.run++;
    this.step = 0;
    this.lastLoss = 0;
    this.history = [];
    this.acc = 0;
  }

  advance(stepsPerFrame: number) {
    if (!this.ready()) return;
    for (let i = 0; i < stepsPerFrame; i++) {
      const loss = this.task.trainStep();
      this.lastLoss = loss;
      this.step++;
      this.acc++;
      if (this.acc >= RECORD_EVERY) {
        this.acc = 0;
        this.history.push(loss);
        if (this.history.length > HISTORY) this.history.shift();
      }
      if (this.task.converged(loss, this.step)) {
        this.task.reset();
        this.run++;
        this.step = 0;
        this.history = [];
      }
    }
  }

  draw(ctx: CanvasRenderingContext2D) {
    this.task.draw(ctx);
  }
}
