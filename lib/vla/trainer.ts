// The real thing behind the hero's "Start Training" button: an asynchronous
// behavioral-cloning loop. Each batch synthesizes 32 (pose, command) states —
// the command drawn from every phrasing x color in examples.ts — renders each
// pose through the same 16x16 silhouette pipeline the live rollout uses,
// labels it with the analytical-IK expert's joint delta toward the commanded
// block's center, and runs one trainOnBatch step. The plotted MSE loss is the
// genuine training loss, and the Rollout arm is driven purely by
// model.predict on its own live 16x16 view — nothing is simulated.
//
// tfjs is dynamically imported on first start() so the landing page loads
// without the ~1MB bundle; batches are paced with awaits between steps so
// the rendering thread stays at 60fps.

import {
  BASE,
  BLACK_BLOCK,
  RED_BLOCK,
  THETA1_RANGE,
  THETA2_RANGE,
  clamp,
  graspTarget,
  ikToBlock,
  solveIK,
} from "./geometry";
import { paintSilhouette } from "./scene";
import { MAX_SEQ_LEN, buildVLAModel, type TF } from "./model";
import { EXAMPLES } from "./examples";

import type * as tfType from "@tensorflow/tfjs";

const BATCH_SIZE = 32;
const HISTORY_LEN = 90;
/** Silhouettes are drawn at 64px then averaged down to 16 — at 16px directly
    the sub-pixel arm strokes alias away entirely. */
const RENDER_SIZE = 64;
/** Minimum gap between batches, so training never starves the rAF loop. */
const BATCH_GAP_MS = 30;
/** Fraction of samples drawn NEAR the commanded block's own IK solution
    (rest are uniform). Densifying data where converged rollouts end up gives
    the policy a sharp zero-crossing at the target instead of a mushy
    regression-to-the-mean plateau — this is what makes displayed episodes
    actually succeed within ~10-15s. */
const NEAR_TARGET_FRAC = 0.35;
const NEAR_TARGET_STD = 0.5;

export type TrainerStatus = "idle" | "loading" | "training";

export class VLATrainer {
  status: TrainerStatus = "idle";
  /** Real MSE from the latest trainOnBatch (NaN before the first batch). */
  loss = NaN;
  /** First recorded loss — the normalization anchor for the UI. */
  initialLoss = NaN;
  lossHistory: number[] = [];
  batches = 0;

  private tf: TF | null = null;
  private model: tfType.LayersModel | null = null;
  private running = false;
  /** Guards against overlapping start() calls after a quick reset+restart. */
  private runId = 0;
  private sceneCanvas: HTMLCanvasElement | null = null;
  private sceneCtx: CanvasRenderingContext2D | null = null;
  private thumbCanvas: HTMLCanvasElement | null = null;
  private thumbCtx: CanvasRenderingContext2D | null = null;

  /** Total expert-labeled examples consumed so far. */
  get samples() {
    return this.batches * BATCH_SIZE;
  }

  /** IK expert: exact joint delta from a pose toward the commanded block. */
  private expertDelta(a1: number, a2: number, red: boolean): [number, number] {
    const t = graspTarget(red ? RED_BLOCK : BLACK_BLOCK);
    const [t1, t2] = solveIK(t.x - BASE.x, t.y - BASE.y);
    return [t1 - a1, t2 - a2];
  }

  private ensureCanvases() {
    if (this.sceneCtx && this.thumbCtx) return;
    this.sceneCanvas = document.createElement("canvas");
    this.sceneCanvas.width = RENDER_SIZE;
    this.sceneCanvas.height = RENDER_SIZE;
    this.sceneCtx = this.sceneCanvas.getContext("2d", {
      willReadFrequently: true,
    });
    this.thumbCanvas = document.createElement("canvas");
    this.thumbCanvas.width = 16;
    this.thumbCanvas.height = 16;
    this.thumbCtx = this.thumbCanvas.getContext("2d", {
      willReadFrequently: true,
    });
  }

  /** Render a pose through the silhouette pipeline; returns 16x16 RGBA. */
  private renderPose(a1: number, a2: number): ImageData {
    this.ensureCanvases();
    const sctx = this.sceneCtx!;
    const tctx = this.thumbCtx!;
    paintSilhouette(sctx, RENDER_SIZE, a1, a2);
    tctx.imageSmoothingEnabled = true;
    tctx.clearRect(0, 0, 16, 16);
    tctx.drawImage(this.sceneCanvas!, 0, 0, RENDER_SIZE, RENDER_SIZE, 0, 0, 16, 16);
    return tctx.getImageData(0, 0, 16, 16);
  }

  /** Roughly-Gaussian noise in [-1,1]*std*2 (sum of two uniforms). */
  private gauss(std: number) {
    return (Math.random() + Math.random() - 1) * std * 2;
  }

  /**
   * One gradient step on a freshly synthesized micro-batch.
   * Returns the batch MSE loss.
   */
  private async trainStep(): Promise<number> {
    const tf = this.tf!;
    const vis = new Float32Array(BATCH_SIZE * 16 * 16 * 3);
    const lang = new Int32Array(BATCH_SIZE * MAX_SEQ_LEN);
    const ys = new Float32Array(BATCH_SIZE * 2);

    for (let i = 0; i < BATCH_SIZE; i++) {
      const ex = EXAMPLES[Math.floor(Math.random() * EXAMPLES.length)];
      const red = ex.color === "red";

      let a1: number;
      let a2: number;
      if (Math.random() < NEAR_TARGET_FRAC) {
        const [t1, t2] = ikToBlock(red ? RED_BLOCK : BLACK_BLOCK);
        a1 = clamp(t1 + this.gauss(NEAR_TARGET_STD), THETA1_RANGE[0], THETA1_RANGE[1]);
        a2 = clamp(t2 + this.gauss(NEAR_TARGET_STD), THETA2_RANGE[0], THETA2_RANGE[1]);
      } else {
        a1 = THETA1_RANGE[0] + Math.random() * (THETA1_RANGE[1] - THETA1_RANGE[0]);
        a2 = THETA2_RANGE[0] + Math.random() * (THETA2_RANGE[1] - THETA2_RANGE[0]);
      }

      const [d1, d2] = this.expertDelta(a1, a2, red);
      ys[i * 2] = d1;
      ys[i * 2 + 1] = d2;

      // INVERTED intensities (background 0, arm/blocks sparse positive).
      // Fed raw, the near-all-white image saturates the conv branch and the
      // model collapses onto language-only predictions — verified offline:
      // raw plateaus at MSE ~1.7 while inverted converges to ~0.02.
      const img = this.renderPose(a1, a2).data;
      const base = i * 16 * 16 * 3;
      for (let p = 0; p < 256; p++) {
        vis[base + p * 3] = 1 - img[p * 4] / 255;
        vis[base + p * 3 + 1] = 1 - img[p * 4 + 1] / 255;
        vis[base + p * 3 + 2] = 1 - img[p * 4 + 2] / 255;
      }

      lang.set(ex.tokens, i * MAX_SEQ_LEN);
    }

    const xsVision = tf.tensor4d(vis, [BATCH_SIZE, 16, 16, 3]);
    const xsLang = tf.tensor2d(lang, [BATCH_SIZE, MAX_SEQ_LEN], "int32");
    const ysT = tf.tensor2d(ys, [BATCH_SIZE, 2]);

    try {
      const h = await this.model!.trainOnBatch([xsVision, xsLang], ysT);
      return Array.isArray(h) ? h[0] : (h as number);
    } finally {
      xsVision.dispose();
      xsLang.dispose();
      ysT.dispose();
    }
  }

  /**
   * Load tfjs (first call only), build a fresh model and run batches until
   * reset(). onUpdate fires after every batch with the new loss recorded.
   */
  async start(onUpdate?: () => void) {
    if (this.running) return;
    const myRun = ++this.runId;
    this.running = true;
    this.status = "loading";
    onUpdate?.();

    if (!this.tf) {
      const tf = await import("@tensorflow/tfjs");
      await tf.ready();
      this.tf = tf;
    }
    if (!this.running || this.runId !== myRun) return; // reset while loading

    this.model?.dispose();
    this.model = buildVLAModel(this.tf);
    this.status = "training";
    this.loss = NaN;
    this.initialLoss = NaN;
    this.lossHistory = [];
    this.batches = 0;
    onUpdate?.();

    while (this.running && this.runId === myRun) {
      const t0 = performance.now();
      const loss = await this.trainStep();
      if (!this.running || this.runId !== myRun) break;
      this.loss = loss;
      if (Number.isNaN(this.initialLoss)) this.initialLoss = loss;
      this.lossHistory.push(loss);
      if (this.lossHistory.length > HISTORY_LEN) this.lossHistory.shift();
      this.batches++;
      onUpdate?.();
      const gap = Math.max(8, BATCH_GAP_MS - (performance.now() - t0));
      await new Promise((r) => setTimeout(r, gap));
    }
  }

  /** Stop training and discard the learned weights (fresh model next start). */
  reset() {
    this.running = false;
    this.runId++;
    this.status = "idle";
    this.model?.dispose();
    this.model = null;
    this.loss = NaN;
    this.initialLoss = NaN;
    this.lossHistory = [];
    this.batches = 0;
  }

  get ready() {
    return this.status === "training" && this.model !== null;
  }

  /**
   * Policy inference for the live rollout: render the arm's current pose to
   * the same 16x16 view the training samples use, run the model, return the
   * predicted joint deltas for the given command tokens.
   */
  predictDelta(a1: number, a2: number, tokens: number[]): [number, number] | null {
    if (!this.ready || this.batches === 0) return null;
    const tf = this.tf!;
    const img = this.renderPose(a1, a2);
    const out = tf.tidy(() => {
      // same inverted-intensity preprocessing as the training batches
      const v = tf
        .sub(1, tf.browser.fromPixels(img, 3).toFloat().div(255))
        .expandDims(0);
      const l = tf.tensor2d([tokens], [1, MAX_SEQ_LEN], "int32");
      const y = this.model!.predict([v, l]) as tfType.Tensor;
      return y.dataSync();
    });
    return [out[0], out[1]];
  }

  /** Loss normalized against the first batch, clamped to [0,1]. */
  lossNorm(): number {
    if (Number.isNaN(this.loss) || Number.isNaN(this.initialLoss)) return 1;
    if (this.initialLoss <= 0) return 0;
    return Math.max(0, Math.min(1, this.loss / this.initialLoss));
  }
}
