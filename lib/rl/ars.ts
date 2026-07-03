// Augmented Random Search (Mania, Guy & Recht, 2018) — a real, published RL
// method. We optimise a linear policy by sampling pairs of weight perturbations,
// scoring each by the *actual return* of a full environment rollout, and stepping
// the weights toward the perturbations that earned more reward.
//
// Training runs on an internal environment, decoupled from what's drawn on
// screen: `advance()` performs real ARS rollouts (paired, from the env's fixed
// start so the gradient estimate is valid), while the rendered agent plays the
// current *mean* policy via `actMean()`. That way the visible agent smoothly
// improves as training converges instead of flickering through the noisy ±
// exploration rollouts.

import type { EnvFactory, RLEnv } from "./types";
import { RunningNorm, act, policySize } from "./policy";

const N_DIRS = 12; // directions sampled per update
const TOP_B = 6; // best directions actually used
const NU = 0.05; // perturbation std
const ALPHA = 0.03; // step size
const HISTORY = 60; // returns kept for the sparkline

function randn(): number {
  let u1 = 0;
  let u2 = 0;
  while (u1 === 0) u1 = Math.random();
  while (u2 === 0) u2 = Math.random();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

interface RolloutSlot {
  dir: number;
  sign: 1 | -1;
}

export class ARSTrainer {
  env: RLEnv; // internal training env (not rendered)
  readonly obsDim: number;
  readonly actDim: number;
  readonly size: number;
  W: Float64Array;
  norm: RunningNorm;

  run = 1; // ARS generation (HUD "run")
  episode = 1; // total training rollouts (HUD "episode")
  lastReturn = 0; // last single rollout return (raw, noisy)
  genMean = 0; // mean return of the most recent generation (smooth metric)
  history: number[] = []; // per-generation mean returns → sparkline / HUD

  private dirs: Float64Array[] = [];
  private rPlus: number[] = [];
  private rMinus: number[] = [];
  private queue: RolloutSlot[] = [];
  private qi = 0;
  private params: Float64Array;
  private epReturn = 0;
  private active = false;

  constructor(makeEnv: EnvFactory) {
    this.env = makeEnv();
    this.obsDim = this.env.obsDim;
    this.actDim = this.env.actDim;
    this.size = policySize(this.actDim, this.obsDim);
    this.W = new Float64Array(this.size);
    this.norm = new RunningNorm(this.obsDim);
    this.seedWeights();
    this.params = new Float64Array(this.size);
    this.newGeneration();
  }

  setSize(w: number, h: number) {
    this.env.setSize(w, h);
  }

  ready() {
    return this.env.w > 0;
  }

  // Action from the current mean policy — used to drive the rendered agent.
  actMean(obs: number[]): number[] {
    return act(this.W, this.norm.normalize(obs), this.actDim, this.obsDim);
  }

  private seedWeights() {
    for (let i = 0; i < this.size; i++) this.W[i] = (Math.random() - 0.5) * 0.1;
  }

  private newGeneration() {
    this.dirs = [];
    this.rPlus = new Array(N_DIRS).fill(0);
    this.rMinus = new Array(N_DIRS).fill(0);
    this.queue = [];
    for (let d = 0; d < N_DIRS; d++) {
      const dir = new Float64Array(this.size);
      for (let i = 0; i < this.size; i++) dir[i] = randn();
      this.dirs.push(dir);
      this.queue.push({ dir: d, sign: 1 }, { dir: d, sign: -1 });
    }
    this.qi = 0;
  }

  private beginRollout() {
    const slot = this.queue[this.qi];
    const dir = this.dirs[slot.dir];
    for (let i = 0; i < this.size; i++)
      this.params[i] = this.W[i] + slot.sign * NU * dir[i];
    // Every rollout starts from the SAME deterministic state so the paired
    // difference (r+ − r−) reflects the perturbation, not start-state luck.
    this.env.reset();
    this.epReturn = 0;
    this.active = true;
  }

  private finishRollout() {
    const slot = this.queue[this.qi];
    if (slot.sign === 1) this.rPlus[slot.dir] = this.epReturn;
    else this.rMinus[slot.dir] = this.epReturn;

    this.lastReturn = this.epReturn;
    this.episode++;
    this.active = false;
    this.qi++;

    if (this.qi >= this.queue.length) {
      this.applyUpdate();
      this.run++;
      this.newGeneration();
    }
  }

  private applyUpdate() {
    const idx = Array.from({ length: N_DIRS }, (_, i) => i);
    idx.sort(
      (a, b) =>
        Math.max(this.rPlus[b], this.rMinus[b]) -
        Math.max(this.rPlus[a], this.rMinus[a])
    );
    const top = idx.slice(0, TOP_B);

    const used: number[] = [];
    for (const d of top) used.push(this.rPlus[d], this.rMinus[d]);
    const mean = used.reduce((s, v) => s + v, 0) / used.length;
    const varr =
      used.reduce((s, v) => s + (v - mean) * (v - mean), 0) / used.length;
    const sigmaR = Math.sqrt(varr) || 1;

    // record a smooth per-generation progress metric for the HUD + sparkline
    let genSum = 0;
    for (let d = 0; d < N_DIRS; d++) genSum += this.rPlus[d] + this.rMinus[d];
    this.genMean = genSum / (2 * N_DIRS);
    this.history.push(this.genMean);
    if (this.history.length > HISTORY) this.history.shift();

    const scale = ALPHA / (TOP_B * sigmaR);
    for (const d of top) {
      const diff = this.rPlus[d] - this.rMinus[d];
      const dir = this.dirs[d];
      for (let i = 0; i < this.size; i++) this.W[i] += scale * diff * dir[i];
    }
  }

  // Advance training by up to `budget` internal environment steps, crossing
  // rollout and generation boundaries as needed.
  advance(budget: number) {
    if (!this.ready()) return;
    let used = 0;
    while (used < budget) {
      if (!this.active) this.beginRollout();
      const obs = this.env.getObs();
      this.norm.observe(obs);
      const action = act(this.params, this.norm.normalize(obs), this.actDim, this.obsDim);
      const { reward, done } = this.env.step(action);
      this.epReturn += reward;
      used++;
      if (done) this.finishRollout();
    }
  }
}
