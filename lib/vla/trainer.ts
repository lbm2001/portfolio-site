// The real thing behind the hero's "Start Training" button: an asynchronous
// behavioral-cloning loop over the 2-block task (colors drawn from a 4-color
// palette). Each batch synthesizes 16 (scene layout, pose, command) states —
// random block placements, sentences from the slot grammar with ~10%
// word-dropout to <unk> — renders each state through the same 32x32
// silhouette pipeline the live rollout uses, labels it with the analytical-
// IK expert's ABSOLUTE target joint angles for the commanded block (plus the
// target color for the auxiliary head), and runs one trainOnBatch step. The
// label does NOT depend on the (randomized, for robustness) pose the arm is
// rendered at — only on which side the named color is on — so the network
// only has to learn "given this color and this scene, where does the arm
// need to end up," not also implicitly infer the current pose and subtract.
// The Rollout arm computes its own delta from the predicted target against
// its actual known current pose (see Hero.tsx's drawArm) and steps toward
// it; the plotted curve is the genuine action loss (Huber regression to that
// absolute target), and the Rollout is driven purely by model.predict on
// its own live 32x32 view.
//
// Training stops by itself once the trailing-window action loss crosses the convergence
// threshold — that switches the hero into "try it" mode where user-typed
// sentences drive the policy. It can also be paused/resumed.
//
// tfjs is dynamically imported on first start() so the landing page loads
// without the ~1MB bundle; batches are paced with awaits between steps so
// the rendering thread stays at 60fps.

import { THETA1_RANGE, THETA2_RANGE, clamp, ikToX } from "./geometry";
import { paintSilhouette } from "./scene";
import {
  COLORS,
  COLOR_TOKEN_IDS,
  MAX_SEQ_LEN,
  PAD,
  UNK,
  blockOfColor,
  presentColor,
  randomLayout,
  sampleSentence,
  type Layout,
} from "./examples";
import { IMG_SIZE, buildVLAModel, type VLAModels, type TF } from "./model";

import type * as tfType from "@tensorflow/tfjs";

// Halved from 32: fewer canvas renders + smaller tensors per batch means
// roughly 2x the gradient steps per wall-clock second — for a task this
// small the noisier per-step gradient estimate is a good trade for more
// frequent updates within a ~15s budget.
const BATCH_SIZE = 16;
/** Silhouettes are drawn at 128px then averaged down to 32 — rendered at
    32px directly the sub-pixel arm strokes alias away. */
const RENDER_SIZE = 128;
/** Minimum gap between batches, so training never starves the rAF loop. The
    render loop runs on its own rAF and only needs a sliver of main-thread time
    yielded back between gradient steps — 30ms here left ~1/3 of every second
    idle. At ~8ms the rAF loop still gets its slice while ~25% more gradient
    steps fit into the same ~15s budget. */
const BATCH_GAP_MS = 8;
/** Fraction of samples drawn NEAR the commanded block's own IK solution
    (rest are uniform over the full pose range). The label no longer depends
    on this sampled pose (it's an absolute target now), but the rendered
    silhouette still does — this keeps vision well-trained on what the scene
    looks like as the rollout closes in near the target, not just far away. */
const NEAR_TARGET_FRAC = 0.35;
const NEAR_TARGET_STD = 0.5;
/** Training word-dropout: chance a non-color token becomes <unk>. */
const WORD_DROPOUT = 0.1;

// Convergence: the mean action loss over a short trailing WINDOW of batches
// stays under CONVERGE_LOSS for CONVERGE_STREAK consecutive batches (after a
// minimum warmup) → training ends and unlocks "try it" mode. MAX_BATCHES is
// the fixed-budget fallback: whatever the loss floor turns out to be,
// training ends and the policy becomes usable.
//
// This used to judge convergence on an EMA with alpha=0.05 — a ~20-batch
// time constant, so ~1/alpha lag before the smoothed value even crossed the
// threshold. Work the step response: if the true loss dropped to ~0.05, the
// EMA needed ~67 more batches just to cross 0.08 (0.95^n < 0.03/0.95), i.e.
// ~6s of pure DETECTION latency after the policy had already learned the
// task. A trailing-window mean of the RAW loss crosses within ~WINDOW
// batches as the old high values roll off — an order of magnitude less lag
// for the same stability, since the streak still guards against a lucky dip.
//
// MIN_BATCHES/CONVERGE_STREAK used to be 300/20 — a 320-batch FLOOR before
// convergence could ever be flagged, regardless of how good the loss got.
// At ~60-100ms/batch that alone is 16-32s, which structurally ruled out
// ever hitting a ~15s convergence target no matter how learnable the task
// was. Lowered so the floor (60 batches) leaves real headroom in the budget.
//
// Threshold is on the HUBER action loss now (see model.ts), not raw MSE:
// with a wrong-side pick capped at ~2.5 instead of ~9.3, the same ~1%
// misclassification tail floors the loss near ~0.025 rather than ~0.09.
// 0.07 sits above that floor with room to spare — the side-classification is
// solved well before the loss grinds down to the floor, and the last stretch
// of descent is just fine-regression polish. Handing off at 0.07 (was 0.045)
// enters "try it" mode noticeably sooner while the policy is already reliably
// picking the right block — the remaining fine-regression gain isn't worth
// stalling the demo for. Raise further to hand off even earlier.
export const CONVERGE_LOSS = 0.07;
const CONVERGE_WINDOW = 10;
const CONVERGE_STREAK = 5;
const MIN_BATCHES = 60;
const MAX_BATCHES = 1800;

export type TrainerStatus =
  | "idle"
  | "loading"
  | "training"
  | "paused"
  | "converged";

export class VLATrainer {
  status: TrainerStatus = "idle";
  /** Real action loss (Huber) from the latest trainOnBatch (NaN before the first). */
  loss = NaN;
  /** Mean action loss over the last CONVERGE_WINDOW batches — the low-lag
      signal convergence is judged on (NaN before the first batch). */
  smoothLoss = NaN;
  /** First recorded loss — the normalization anchor for the UI. */
  initialLoss = NaN;
  lossHistory: number[] = [];
  batches = 0;

  private tf: TF | null = null;
  private models: VLAModels | null = null;
  /** A separate inference model holding a FROZEN copy of the policy weights,
      refreshed only at snapshotPolicy() calls (each demo-cycle boundary and on
      convergence). The live rollout attempt runs against this so it sees a
      fixed policy for its whole cycle while the main model keeps training. */
  private frozenModels: VLAModels | null = null;
  private running = false;
  private paused = false;
  private convergeStreak = 0;
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

  get ready() {
    return (
      (this.status === "training" ||
        this.status === "paused" ||
        this.status === "converged") &&
      this.models !== null &&
      this.batches > 0
    );
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
    this.thumbCanvas.width = IMG_SIZE;
    this.thumbCanvas.height = IMG_SIZE;
    this.thumbCtx = this.thumbCanvas.getContext("2d", {
      willReadFrequently: true,
    });
  }

  /** Render a state through the silhouette pipeline; returns 32x32 RGBA. */
  private renderPose(a1: number, a2: number, layout: Layout): ImageData {
    this.ensureCanvases();
    paintSilhouette(this.sceneCtx!, RENDER_SIZE, a1, a2, layout);
    const tctx = this.thumbCtx!;
    tctx.imageSmoothingEnabled = true;
    tctx.clearRect(0, 0, IMG_SIZE, IMG_SIZE);
    tctx.drawImage(
      this.sceneCanvas!,
      0,
      0,
      RENDER_SIZE,
      RENDER_SIZE,
      0,
      0,
      IMG_SIZE,
      IMG_SIZE
    );
    return tctx.getImageData(0, 0, IMG_SIZE, IMG_SIZE);
  }

  /** Roughly-Gaussian noise (sum of two uniforms). */
  private gauss(std: number) {
    return (Math.random() + Math.random() - 1) * std * 2;
  }

  /**
   * One gradient step on a freshly synthesized micro-batch.
   * Returns the batch's action loss (Huber).
   */
  private async trainStep(): Promise<number> {
    const tf = this.tf!;
    const px = IMG_SIZE * IMG_SIZE * 3;
    const vis = new Float32Array(BATCH_SIZE * px);
    const lang = new Int32Array(BATCH_SIZE * MAX_SEQ_LEN);
    const ysA = new Float32Array(BATCH_SIZE * 2);
    const ysC = new Float32Array(BATCH_SIZE * COLORS.length);

    for (let i = 0; i < BATCH_SIZE; i++) {
      const layout = randomLayout();
      // command one of the two colors actually present in this scene
      const sentence = sampleSentence(presentColor(layout));
      const targetX = blockOfColor(layout, sentence.color).x;

      let a1: number;
      let a2: number;
      if (Math.random() < NEAR_TARGET_FRAC) {
        const [t1, t2] = ikToX(targetX);
        a1 = clamp(t1 + this.gauss(NEAR_TARGET_STD), THETA1_RANGE[0], THETA1_RANGE[1]);
        a2 = clamp(t2 + this.gauss(NEAR_TARGET_STD), THETA2_RANGE[0], THETA2_RANGE[1]);
      } else {
        a1 = THETA1_RANGE[0] + Math.random() * (THETA1_RANGE[1] - THETA1_RANGE[0]);
        a2 = THETA2_RANGE[0] + Math.random() * (THETA2_RANGE[1] - THETA2_RANGE[0]);
      }

      // ABSOLUTE target joint angles, not a delta from the sampled pose:
      // the label no longer depends on the (randomized, for robustness)
      // pose the arm is rendered at — only on which side the commanded
      // color is on. That removes the need for the network to also read
      // the current pose out of the image and implicitly subtract; the
      // rollout computes its own delta from this against its actual known
      // current pose (see Hero.tsx).
      const [t1, t2] = ikToX(targetX);
      ysA[i * 2] = t1;
      ysA[i * 2 + 1] = t2;
      ysC[i * COLORS.length + sentence.color] = 1;

      // INVERTED intensities (background 0, content sparse positive) — fed
      // raw, the near-all-white image saturates the conv branch and the
      // model collapses onto language-only predictions.
      const img = this.renderPose(a1, a2, layout).data;
      const base = i * px;
      for (let p = 0; p < IMG_SIZE * IMG_SIZE; p++) {
        vis[base + p * 3] = 1 - img[p * 4] / 255;
        vis[base + p * 3 + 1] = 1 - img[p * 4 + 1] / 255;
        vis[base + p * 3 + 2] = 1 - img[p * 4 + 2] / 255;
      }

      // word-dropout: non-color tokens occasionally become <unk> so the
      // encoder learns to handle unknown words in free user text
      for (let s = 0; s < MAX_SEQ_LEN; s++) {
        let id = sentence.tokens[s];
        if (id !== PAD && !COLOR_TOKEN_IDS.has(id) && Math.random() < WORD_DROPOUT)
          id = UNK;
        lang[i * MAX_SEQ_LEN + s] = id;
      }
    }

    const xsVision = tf.tensor4d(vis, [BATCH_SIZE, IMG_SIZE, IMG_SIZE, 3]);
    const xsLang = tf.tensor2d(lang, [BATCH_SIZE, MAX_SEQ_LEN], "int32");
    const yAction = tf.tensor2d(ysA, [BATCH_SIZE, 2]);
    const yColor = tf.tensor2d(ysC, [BATCH_SIZE, COLORS.length]);

    try {
      const h = await this.models!.model.trainOnBatch(
        [xsVision, xsLang],
        [yAction, yColor]
      );
      // multi-output: [totalLoss, actionLoss, colorLoss]
      return Array.isArray(h) ? (h[1] as number) : (h as number);
    } finally {
      xsVision.dispose();
      xsLang.dispose();
      yAction.dispose();
      yColor.dispose();
    }
  }

  /**
   * Load tfjs (first call only), build a fresh model and run batches until
   * pause/reset or convergence. onUpdate fires after every batch.
   */
  async start(onUpdate?: () => void) {
    if (this.running) return;
    const myRun = ++this.runId;
    this.running = true;
    this.paused = false;
    this.status = "loading";
    onUpdate?.();

    if (!this.tf) {
      // Import the umbrella "@tensorflow/tfjs" package. A prior attempt to
      // import core+layers+webgl-backend separately (to shed the unused
      // converter/data/cpu-backend weight) measured ~0KB real savings in a
      // production build (core's op/gradient library dominates regardless)
      // AND broke at runtime: tfjs-layers imports its own copy of
      // tfjs-core internally, and the bundler didn't dedupe it against the
      // one imported here, so tensors crossing between "our" core and
      // layers' internal core lacked expected prototype methods (surfaced
      // as "rMat.flatten is not a function" deep inside GRU). The umbrella
      // package guarantees a single shared core instance — not worth
      // reintroducing that class of bug for zero measured benefit.
      const tf = await import("@tensorflow/tfjs");
      await tf.ready();
      this.tf = tf;
    }
    if (!this.running || this.runId !== myRun) return; // reset while loading

    this.disposeModels();
    this.models = buildVLAModel(this.tf);
    this.loss = NaN;
    this.smoothLoss = NaN;
    this.initialLoss = NaN;
    this.lossHistory = [];
    this.batches = 0;
    this.convergeStreak = 0;

    // WebGL compiles each distinct kernel shader (conv2d, pooling, embedding
    // gather, GRU internals, the losses, Adam's update ops) the first time
    // it's used — a one-time cost that would otherwise stall the FIRST
    // visible batch right after the status flips to "Training". Pay it here
    // instead, while the UI still reads "Loading" (a state the user already
    // expects to wait through), so training visibly moves at full speed
    // from the first rendered batch.
    const warmupLoss = await this.trainStep();
    if (!this.running || this.runId !== myRun) return;
    this.loss = warmupLoss;
    this.smoothLoss = warmupLoss;
    this.initialLoss = warmupLoss;
    this.lossHistory.push(warmupLoss);
    this.batches = 1;

    this.status = "training";
    onUpdate?.();

    while (this.running && this.runId === myRun) {
      if (this.paused) {
        await new Promise((r) => setTimeout(r, 120));
        continue;
      }
      const t0 = performance.now();
      const loss = await this.trainStep();
      if (!this.running || this.runId !== myRun) break;
      this.loss = loss;
      if (Number.isNaN(this.initialLoss)) this.initialLoss = loss;
      // keep the FULL curve from batch 0 → now (capped only by MAX_BATCHES),
      // so the plot shows the whole training development, not a trailing slice
      this.lossHistory.push(loss);
      this.batches++;
      // low-lag convergence signal: mean of the last CONVERGE_WINDOW raw
      // losses (read straight off the tail of lossHistory — no extra buffer)
      const window = this.lossHistory.slice(-CONVERGE_WINDOW);
      this.smoothLoss = window.reduce((a, b) => a + b, 0) / window.length;

      // converged? training's job is done — keep the model, stop the loop
      if (this.batches >= MIN_BATCHES && this.smoothLoss < CONVERGE_LOSS) {
        this.convergeStreak++;
      } else {
        this.convergeStreak = 0;
      }
      if (this.convergeStreak >= CONVERGE_STREAK || this.batches >= MAX_BATCHES) {
        this.snapshotPolicy(); // freeze the final weights for "try it" mode
        this.status = "converged";
        this.running = false;
        onUpdate?.();
        return;
      }

      onUpdate?.();
      const gap = Math.max(8, BATCH_GAP_MS - (performance.now() - t0));
      await new Promise((r) => setTimeout(r, gap));
    }
  }

  /** Halt gradient steps without touching the model (Resume continues). */
  pause() {
    if (this.status !== "training") return;
    this.paused = true;
    this.status = "paused";
  }

  resume() {
    if (this.status !== "paused") return;
    this.paused = false;
    this.status = "training";
  }

  /** Stop training and discard the learned weights (fresh model next start). */
  reset() {
    this.running = false;
    this.paused = false;
    this.runId++;
    this.status = "idle";
    this.disposeModels();
    this.loss = NaN;
    this.smoothLoss = NaN;
    this.initialLoss = NaN;
    this.lossHistory = [];
    this.batches = 0;
    this.convergeStreak = 0;
  }

  private disposeModels() {
    // the viz sub-models share layers with the main model — disposing the
    // main graph frees the shared weights; dispose() on the others only
    // drops their container objects
    this.models?.model.dispose();
    this.models = null;
    this.frozenModels?.model.dispose();
    this.frozenModels = null;
  }

  /** Preprocess a 32x32 RGBA thumb into the model's inverted input tensor. */
  private visionTensor(img: ImageData): tfType.Tensor4D {
    const tf = this.tf!;
    return tf.tidy(() =>
      tf.sub(1, tf.browser.fromPixels(img, 3).toFloat().div(255)).expandDims(0)
    ) as tfType.Tensor4D;
  }

  /**
   * Policy inference for the live rollout: render the arm's current state
   * (pose + the rollout's own block layout) to the same 32x32 view the
   * training samples use, run the model, return the predicted ABSOLUTE
   * target joint angles (not a delta — the caller subtracts its own known
   * current pose to get the step direction; see Hero.tsx's drawArm).
   */
  predictTarget(
    a1: number,
    a2: number,
    tokens: number[],
    layout: Layout
  ): [number, number] | null {
    if (!this.ready) return null;
    return this.inferTarget(this.models!.model, a1, a2, tokens, layout);
  }

  /**
   * Freeze the current policy weights into the separate inference model, so a
   * rollout attempt can run a FIXED policy for its whole cycle (matching how a
   * real rollout uses frozen weights) while background training keeps updating
   * the main model. Called at each demo-cycle boundary and on convergence.
   * No-op until the first batch has built the main model.
   */
  snapshotPolicy() {
    if (!this.tf || !this.models) return;
    if (!this.frozenModels) this.frozenModels = buildVLAModel(this.tf);
    // getWeights() returns the live variables' current values; setWeights
    // copies them into the frozen model's own variables, so the snapshot holds
    // steady as the main model trains on. Same architecture → identical weight
    // ordering. The returned tensors are the main model's — do NOT dispose.
    this.frozenModels.model.setWeights(this.models.model.getWeights());
  }

  /**
   * Like predictTarget, but runs the FROZEN snapshot from the last
   * snapshotPolicy() call, so a rollout attempt sees one fixed policy for its
   * whole cycle. Falls back to the live model if no snapshot exists yet.
   */
  predictFrozenTarget(
    a1: number,
    a2: number,
    tokens: number[],
    layout: Layout
  ): [number, number] | null {
    if (!this.ready) return null;
    const model = this.frozenModels?.model ?? this.models!.model;
    return this.inferTarget(model, a1, a2, tokens, layout);
  }

  /** Render the state, run the given model, return predicted target angles. */
  private inferTarget(
    model: tfType.LayersModel,
    a1: number,
    a2: number,
    tokens: number[],
    layout: Layout
  ): [number, number] {
    const tf = this.tf!;
    const img = this.renderPose(a1, a2, layout);
    const out = tf.tidy(() => {
      const v = this.visionTensor(img);
      const l = tf.tensor2d([tokens], [1, MAX_SEQ_LEN], "int32");
      const [action] = model.predict([v, l]) as tfType.Tensor[];
      return action.dataSync();
    });
    return [out[0], out[1]];
  }

  /**
   * Decode which color a token sequence names, via the auxiliary head (which
   * reads only the language branch — vision input is zeros).
   */
  decodeColor(tokens: number[]): { color: number; prob: number } | null {
    if (!this.ready) return null;
    const tf = this.tf!;
    const probs = tf.tidy(() => {
      const v = tf.zeros([1, IMG_SIZE, IMG_SIZE, 3]);
      const l = tf.tensor2d([tokens], [1, MAX_SEQ_LEN], "int32");
      const [, color] = this.models!.model.predict([v, l]) as tfType.Tensor[];
      return color.dataSync();
    });
    let best = 0;
    for (let i = 1; i < probs.length; i++) if (probs[i] > probs[best]) best = i;
    return { color: best, prob: probs[best] };
  }

  /**
   * Exact per-token contribution to the decoded color's logit — the live
   * per-chip bars. Because the color head is a single linear layer on a
   * MEAN-pooled bag-of-embeddings, that logit is precisely
   * mean_t(embedding[token_t] · colorWeights[:, color]): no forward pass
   * needed, just two small weight-matrix lookups + a dot product per token.
   */
  tokenContributions(tokens: number[], color: number): number[] | null {
    if (!this.ready) return null;
    const embedTable = this.models!.model
      .getLayer("text_embedding")
      .getWeights()[0]
      .arraySync() as number[][];
    const colorWeights = this.models!.model
      .getLayer("color")
      .getWeights()[0]
      .arraySync() as number[][];
    const dim = colorWeights.length;
    const contributions = tokens.map((tok) => {
      let dot = 0;
      for (let d = 0; d < dim; d++) dot += embedTable[tok][d] * colorWeights[d][color];
      return dot;
    });
    const max = Math.max(...contributions.map(Math.abs), 1e-6);
    return contributions.map((c) => Math.abs(c) / max);
  }

  /** Loss normalized against the first batch, clamped to [0,1]. */
  lossNorm(): number {
    if (Number.isNaN(this.loss) || Number.isNaN(this.initialLoss)) return 1;
    if (this.initialLoss <= 0) return 0;
    return Math.max(0, Math.min(1, this.loss / this.initialLoss));
  }
}
