// The real thing behind the hero's "Start Training" button: an asynchronous
// behavioral-cloning loop over the run-config's task set (lift/touch/stack
// on 2..4 blocks from a 2/4/8-color palette — see lib/vla/run-config.ts).
// Each batch synthesizes (scene layout, pose, command) states — random block
// placements, per-task sentences from the slot grammar with ~10% word-dropout
// to <unk> — renders each state through the same silhouette pipeline the live
// rollout uses, labels it with the analytical-IK expert's ABSOLUTE target
// joint angles + desired gripper state (plus color/ref-color/task for the
// auxiliary heads), and runs one trainOnBatch step.
//
// The carry phase is policy-driven, so lift/stack samples are CARRY-
// CONDITIONED: a carryFrac share render the commanded block IN THE GRIPPER
// of the sampled pose, and the label flips to the carry-phase target —
// lift → REST with the grip closed, stack → holding the block seated on the
// reference block with the grip open (release). "Is a block in my hand" is
// only visible in the image, which is exactly the state cue the network has
// to learn. Apart from the carried block riding at the effector, labels do
// NOT depend on the (randomized, for robustness) rendered pose — the network
// learns "given this scene, command and carry state, where does the arm need
// to end up," not also implicitly infer the current pose and subtract. The
// Rollout arm computes its own delta from the predicted target against its
// actual known current pose (see Hero.tsx's drawArm) and steps toward it;
// the plotted curve is the genuine action loss (Huber regression to that
// absolute target), and the Rollout is driven purely by model.predict on
// its own live silhouette view.
//
// Training stops by itself once the trailing-window action loss crosses the
// convergence threshold — that switches the hero into "try it" mode where
// user-typed sentences drive the policy. It can also be paused/resumed.
//
// THREADING: this class is environment-agnostic — it normally runs inside
// lib/vla/trainer.worker.ts (OffscreenCanvas, off the main thread, so the
// gradient loop never contends with the 60fps display), with lib/vla/
// trainer.ts as the main-thread proxy Hero talks to. On browsers without
// module-worker/OffscreenCanvas support the proxy instantiates it inline on
// the main thread instead (DOM canvases, the pre-worker behavior).
//
// tfjs is dynamically imported on first start() so the landing page loads
// without the ~1MB bundle; batches are paced with awaits between steps so
// whichever thread hosts the loop stays responsive.

import { CONFIG } from "./config";
import {
  REST,
  THETA1_RANGE,
  THETA2_RANGE,
  clamp,
  fk,
  ikToPlace,
  ikToX,
} from "./geometry";
import { paintSilhouette } from "./scene";
import {
  COLORS,
  LABEL_TOKEN_IDS,
  MAX_SEQ_LEN,
  PAD,
  UNK,
  blockOfColor,
  randomLayout,
  sampleCommand,
  type Layout,
} from "./examples";
import { TASKS, type TaskKind } from "./run-config";
import {
  ATTN_GRID,
  IMG_SIZE,
  REF_NONE,
  buildVLAModel,
  type VLAModels,
  type TF,
} from "./model";
import { EMBED_DIM } from "./vocab.gen";
import { embeddingMatrix, loadEmbeddings } from "./embeddings";

import type * as tfType from "@tensorflow/tfjs";

// All tuning knobs live in lib/vla/config.ts — the rationale for each value is
// documented there. Aliased to locals so the loop body below reads the same.
const BATCH_SIZE = CONFIG.trainer.batchSize;
const RENDER_SIZE = CONFIG.trainer.renderSize;
const BATCH_GAP_MS = CONFIG.trainer.batchGapMs;
const NEAR_TARGET_FRAC = CONFIG.trainer.nearTargetFrac;
const NEAR_TARGET_STD = CONFIG.trainer.nearTargetStd;
const WORD_DROPOUT = CONFIG.trainer.wordDropout;
const CARRY_FRAC = CONFIG.trainer.carryFrac;

// Convergence: the mean action loss over a short trailing WINDOW of batches
// stays under CONVERGE_LOSS for CONVERGE_STREAK consecutive batches (after
// MIN_BATCHES warmup) → training ends and unlocks "try it" mode. MAX_BATCHES is
// the fixed-budget fallback. Threshold is on the HUBER action loss (model.ts).
export const CONVERGE_LOSS = CONFIG.trainer.converge.loss;
const CONVERGE_WINDOW = CONFIG.trainer.converge.window;
const CONVERGE_STREAK = CONFIG.trainer.converge.streak;
const MIN_BATCHES = CONFIG.trainer.converge.minBatches;
const MAX_BATCHES = CONFIG.trainer.converge.maxBatches;

export type TrainerStatus =
  | "idle"
  | "loading"
  | "training"
  | "paused"
  | "converged";

/** One policy inference: the action plus the "where the model looks" viz,
    all read from a single forward pass of the viz twin (see model.ts). */
export interface PredictResult {
  /** Predicted ABSOLUTE target joint angles. */
  target: [number, number];
  /** Predicted DESIRED gripper state at that target, sigmoid in [0,1]
      (≥0.5 = closed). Drives the grasp/release decisions in Hero's episode
      machine and the Action Head readout. */
  grip: number;
  /** Spatial attention over the ATTN_GRID×ATTN_GRID vision cells, row-major,
      normalized so the peak cell is 1 (ready to use as overlay alpha). */
  attn: number[];
  /** The map's soft-argmax — where the model looks, in [0,1] image coords of
      the silhouette view (x right, y down). */
  xy: [number, number];
}

/** What the language heads decode from a token sequence (vision zeroed) —
    the "decoded target" readout and try-it routing. refColor is null when
    the ref head picks its "none" class (lift/touch commands). */
export interface DecodedCommand {
  task: TaskKind;
  taskProb: number;
  color: number;
  colorProb: number;
  refColor: number | null;
  refProb: number;
}

/** 2D context in either environment (worker OffscreenCanvas / DOM canvas). */
type Ctx2D = CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D;

/** A square 2D canvas in whichever flavor this environment provides. */
function make2d(size: number): { canvas: CanvasImageSource; ctx: Ctx2D } {
  if (typeof OffscreenCanvas !== "undefined") {
    const canvas = new OffscreenCanvas(size, size);
    return {
      canvas,
      ctx: canvas.getContext("2d", {
        willReadFrequently: true,
      }) as OffscreenCanvasRenderingContext2D,
    };
  }
  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;
  return {
    canvas,
    ctx: canvas.getContext("2d", { willReadFrequently: true })!,
  };
}

export class VLATrainerCore {
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
  private sceneCanvas: CanvasImageSource | null = null;
  private sceneCtx: Ctx2D | null = null;
  private thumbCtx: Ctx2D | null = null;

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
    const scene = make2d(RENDER_SIZE);
    this.sceneCanvas = scene.canvas;
    this.sceneCtx = scene.ctx;
    this.thumbCtx = make2d(IMG_SIZE).ctx;
  }

  /** Render a state through the silhouette pipeline; returns IMG_SIZE RGBA.
      `carry` draws that block at the effector instead of its rest spot —
      the carry-phase state cue. */
  private renderPose(
    a1: number,
    a2: number,
    layout: Layout,
    carry: number | null = null
  ): ImageData {
    this.ensureCanvases();
    paintSilhouette(this.sceneCtx!, RENDER_SIZE, a1, a2, layout, carry);
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

  /** Write the attention-map label for one sample into ysM: BILINEAR weights
      over the (up to) 4 grid cells around a point given in the silhouette
      view's [0,1] (u, v) image coords. Soft on purpose — a hard one-hot label
      would train the map toward a one-hot, quantizing the soft-argmax readout
      to cell centers (measured: ~0.15 median reach error); the bilinear
      label's CE optimum is a distribution whose EXPECTATION is the point's
      continuous position, so sub-cell precision survives. */
  private writeMapLabel(ysM: Float32Array, base: number, u: number, v: number) {
    const G = ATTN_GRID;
    // continuous cell coords (cell centers sit at integer +0.5 / G)
    const cx = u * G - 0.5;
    const cy = v * G - 0.5;
    const j0 = Math.floor(cx);
    const i0 = Math.floor(cy);
    const fx = cx - j0;
    const fy = cy - i0;
    for (const [i, j, w] of [
      [i0, j0, (1 - fy) * (1 - fx)],
      [i0, j0 + 1, (1 - fy) * fx],
      [i0 + 1, j0, fy * (1 - fx)],
      [i0 + 1, j0 + 1, fy * fx],
    ]) {
      if (i < 0 || i >= G || j < 0 || j >= G || w === 0) continue;
      ysM[base + i * G + j] += w;
    }
  }

  /** A resting block's visual center in silhouette [0,1] (u, v) coords.
      Mirrors paintSilhouette's geometry: sceneMap placement, the block's rest
      height, plus the silBlockScale boost — so the label points at the pixels
      the model actually sees. */
  private blockUV(x: number, size: number, rest = 0): [number, number] {
    const u = 0.5 + (x - 0.5) * CONFIG.render.sceneScale;
    const yCenter = rest + (size * CONFIG.render.silBlockScale) / 2;
    const v = CONFIG.render.floorY - yCenter * CONFIG.render.sceneScale;
    return [u, v];
  }

  /** The effector position in silhouette [0,1] (u, v) coords — where a
      CARRIED block renders, so it's the attention anchor for lift's carry
      phase (the only pose-DEPENDENT label; the action label stays REST). */
  private effectorUV(a1: number, a2: number): [number, number] {
    const e = fk(a1, a2);
    const u = 0.5 + (e.ex - 0.5) * CONFIG.render.sceneScale;
    const v = CONFIG.render.floorY - e.ey * CONFIG.render.sceneScale;
    return [u, v];
  }

  /**
   * One gradient step on a freshly synthesized micro-batch.
   * Returns the batch's action loss (Huber).
   */
  private async trainStep(): Promise<number> {
    const tf = this.tf!;
    const px = IMG_SIZE * IMG_SIZE * 3;
    const cells = ATTN_GRID * ATTN_GRID;
    const refClasses = COLORS.length + 1; // + the "none" class
    const vis = new Float32Array(BATCH_SIZE * px);
    const lang = new Int32Array(BATCH_SIZE * MAX_SEQ_LEN);
    const ysA = new Float32Array(BATCH_SIZE * 2);
    const ysG = new Float32Array(BATCH_SIZE);
    const ysC = new Float32Array(BATCH_SIZE * COLORS.length);
    const ysR = new Float32Array(BATCH_SIZE * refClasses);
    const ysT = new Float32Array(BATCH_SIZE * TASKS.length);
    const ysM = new Float32Array(BATCH_SIZE * cells);

    for (let i = 0; i < BATCH_SIZE; i++) {
      const layout = randomLayout();
      // an executable command for a run-config task: the acted-on color is
      // present; stack also names a second present color to stack onto
      const sentence = sampleCommand(layout);
      const target = blockOfColor(layout, sentence.color);
      const ref =
        sentence.task === "stack"
          ? blockOfColor(layout, sentence.refColor!)
          : null;

      // carry-conditioned label (see the file header): a CARRY_FRAC share of
      // lift/stack samples render the commanded block in the gripper and
      // label the carry-phase target instead of the grasp
      const midCarry = sentence.task !== "touch" && Math.random() < CARRY_FRAC;
      let t1: number;
      let t2: number;
      let grip: number;
      if (!midCarry) {
        // reach phase: the grasp point (the IK label depends on the block's
        // SIZE too — grasp height = size/2 — so a bigger commanded block
        // resolves to a higher reach). Desired grip AT the target: touch
        // stays open, lift/stack close to pick the block up.
        [t1, t2] = ikToX(target.x, target.size, target.y ?? 0);
        grip = sentence.task === "touch" ? 0 : 1;
      } else if (sentence.task === "lift") {
        // carrying → bring it home, keep holding
        [t1, t2] = REST;
        grip = 1;
      } else {
        // carrying → hold the block seated on the reference block, let go
        [t1, t2] = ikToPlace(ref!.x, (ref!.y ?? 0) + ref!.size, target.size);
        grip = 0;
      }

      let a1: number;
      let a2: number;
      if (Math.random() < NEAR_TARGET_FRAC) {
        a1 = clamp(t1 + this.gauss(NEAR_TARGET_STD), THETA1_RANGE[0], THETA1_RANGE[1]);
        a2 = clamp(t2 + this.gauss(NEAR_TARGET_STD), THETA2_RANGE[0], THETA2_RANGE[1]);
      } else {
        a1 = THETA1_RANGE[0] + Math.random() * (THETA1_RANGE[1] - THETA1_RANGE[0]);
        a2 = THETA2_RANGE[0] + Math.random() * (THETA2_RANGE[1] - THETA2_RANGE[0]);
      }

      // ABSOLUTE target joint angles, not a delta from the sampled pose:
      // the label doesn't depend on the (randomized, for robustness) pose
      // the arm is rendered at — only on the scene, the command and the
      // carry state. That removes the need for the network to also read the
      // current pose out of the image and implicitly subtract; the rollout
      // computes its own delta from this against its actual known current
      // pose (see Hero.tsx).
      ysA[i * 2] = t1;
      ysA[i * 2 + 1] = t2;
      ysG[i] = grip;
      ysC[i * COLORS.length + sentence.color] = 1;
      ysR[i * refClasses + (sentence.refColor ?? REF_NONE)] = 1;
      ysT[i * TASKS.length + TASKS.indexOf(sentence.task)] = 1;
      // attention supervision: bilinear soft label on the block the policy
      // should currently home on (see model.ts — the action loss alone
      // cannot sharpen the map). Reach phase → the commanded block at its
      // rest spot; stack's carry → the REFERENCE block; lift's carry → the
      // carried block itself, riding at the effector of the rendered pose.
      const [u, v] = !midCarry
        ? this.blockUV(target.x, target.size, target.y ?? 0)
        : sentence.task === "stack"
          ? this.blockUV(ref!.x, ref!.size, ref!.y ?? 0)
          : this.effectorUV(a1, a2);
      this.writeMapLabel(ysM, i * cells, u, v);

      // INVERTED intensities (background 0, content sparse positive) — fed
      // raw, the near-all-white image saturates the conv branch and the
      // model collapses onto language-only predictions.
      const img = this.renderPose(
        a1,
        a2,
        layout,
        midCarry ? sentence.color : null
      ).data;
      const base = i * px;
      for (let p = 0; p < IMG_SIZE * IMG_SIZE; p++) {
        vis[base + p * 3] = 1 - img[p * 4] / 255;
        vis[base + p * 3 + 1] = 1 - img[p * 4 + 1] / 255;
        vis[base + p * 3 + 2] = 1 - img[p * 4 + 2] / 255;
      }

      // word-dropout: tokens that don't carry label information (articles/
      // nouns/fillers — NOT color words or task verbs/preps, see
      // LABEL_TOKEN_IDS) occasionally become <unk> so the encoder learns to
      // handle unknown words in free user text
      for (let s = 0; s < MAX_SEQ_LEN; s++) {
        let id = sentence.tokens[s];
        if (id !== PAD && !LABEL_TOKEN_IDS.has(id) && Math.random() < WORD_DROPOUT)
          id = UNK;
        lang[i * MAX_SEQ_LEN + s] = id;
      }
    }

    const xsVision = tf.tensor4d(vis, [BATCH_SIZE, IMG_SIZE, IMG_SIZE, 3]);
    const xsLang = tf.tensor2d(lang, [BATCH_SIZE, MAX_SEQ_LEN], "int32");
    const yAction = tf.tensor2d(ysA, [BATCH_SIZE, 2]);
    const yGrip = tf.tensor2d(ysG, [BATCH_SIZE, 1]);
    const yColor = tf.tensor2d(ysC, [BATCH_SIZE, COLORS.length]);
    const yRef = tf.tensor2d(ysR, [BATCH_SIZE, refClasses]);
    const yTask = tf.tensor2d(ysT, [BATCH_SIZE, TASKS.length]);
    const yMap = tf.tensor2d(ysM, [BATCH_SIZE, cells]);

    try {
      const h = await this.models!.model.trainOnBatch(
        [xsVision, xsLang],
        [yAction, yGrip, yColor, yRef, yTask, yMap]
      );
      // multi-output: [total, action, grip, color, refColor, task, map] —
      // index 1 stays the Huber ACTION loss the convergence logic watches
      return Array.isArray(h) ? (h[1] as number) : (h as number);
    } finally {
      xsVision.dispose();
      xsLang.dispose();
      yAction.dispose();
      yGrip.dispose();
      yColor.dispose();
      yRef.dispose();
      yTask.dispose();
      yMap.dispose();
    }
  }

  /**
   * Load tfjs (first call only), build a fresh model and run batches until
   * pause/reset or convergence. onUpdate fires after every batch.
   */
  async start(onUpdate?: () => void): Promise<void> {
    if (this.running) return;
    const myRun = ++this.runId;
    this.running = true;
    this.paused = false;
    this.status = "loading";
    onUpdate?.();

    // the pretrained GloVe assets (~1MB) load in parallel with tfjs; both
    // are cached after the first start, so a reset+restart resolves instantly
    let embed: Float32Array;
    try {
      const embedP = loadEmbeddings();
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
      embed = await embedP;
    } catch (err) {
      // network failure on the embedding fetch — back to idle so the button
      // becomes "Start Training" again (loadEmbeddings un-caches the
      // rejection, so the retry refetches)
      console.error("VLA trainer failed to load", err);
      this.running = false;
      this.status = "idle";
      onUpdate?.();
      return;
    }
    if (!this.running || this.runId !== myRun) return; // reset while loading

    this.disposeModels();
    this.models = buildVLAModel(this.tf, embed);
    this.loss = NaN;
    this.smoothLoss = NaN;
    this.initialLoss = NaN;
    this.lossHistory = [];
    this.batches = 0;
    this.convergeStreak = 0;

    try {
      // WebGL compiles each distinct kernel shader (conv2d, pooling, embedding
      // gather, the losses, Adam's update ops) the first time it's used — a
      // one-time cost that would otherwise stall the FIRST visible batch right
      // after the status flips to "Training". Pay it here instead, while the UI
      // still reads "Loading" (a state the user already expects to wait
      // through), so training visibly moves at full speed from the first
      // rendered batch.
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
    } catch (err) {
      // A batch threw mid-training (e.g. the WebGL backend lost its GPU context
      // and an op couldn't recover). Uncaught, this rejection has nobody
      // awaiting it — trainer.worker.ts calls start() fire-and-forget — so
      // status would sit stuck on "loading" forever with no visible signal.
      // Fall back to the cpu backend once (this small CNN trains fine without a
      // GPU, just slower) and retry from a clean model; give up to idle only if
      // cpu ALSO fails.
      if (!this.running || this.runId !== myRun) return;
      console.error("VLA trainer step failed", err);
      if (this.tf.getBackend() !== "cpu") {
        console.warn("VLA trainer falling back to the cpu backend");
        this.running = false;
        await this.tf.setBackend("cpu");
        return this.start(onUpdate);
      }
      this.running = false;
      this.status = "idle";
      this.disposeModels();
      onUpdate?.();
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
   * Policy inference on the LIVE (still-training) model: render the given
   * state (pose + block layout + carried block) to the same model's-eye view
   * the training samples use, run the viz twin, return the predicted
   * ABSOLUTE target joint angles + desired gripper state plus the spatial-
   * attention readout (not a delta — the caller subtracts its own known
   * current pose to get the step direction; see Hero.tsx's drawArm). Also
   * serves the Vision Encoder panel's live "where the model looks" heatmap
   * during training.
   */
  predictTarget(
    a1: number,
    a2: number,
    tokens: number[],
    layout: Layout,
    carry: number | null = null
  ): PredictResult | null {
    if (!this.ready) return null;
    return this.inferTarget(this.models!, a1, a2, tokens, layout, carry);
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
    // embeddingMatrix() is non-null whenever models exist (start() awaits it)
    if (!this.frozenModels)
      this.frozenModels = buildVLAModel(this.tf, embeddingMatrix()!);
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
    layout: Layout,
    carry: number | null = null
  ): PredictResult | null {
    if (!this.ready) return null;
    return this.inferTarget(
      this.frozenModels ?? this.models!,
      a1,
      a2,
      tokens,
      layout,
      carry
    );
  }

  /** Render the state, run the given models' viz twin, return the predicted
      target angles + gripper + attention readout (one forward pass for all). */
  private inferTarget(
    models: VLAModels,
    a1: number,
    a2: number,
    tokens: number[],
    layout: Layout,
    carry: number | null = null
  ): PredictResult {
    const tf = this.tf!;
    const img = this.renderPose(a1, a2, layout, carry);
    const out = tf.tidy(() => {
      const v = this.visionTensor(img);
      const l = tf.tensor2d([tokens], [1, MAX_SEQ_LEN], "int32");
      const [action, grip, attn] = models.viz.predict([v, l]) as tfType.Tensor[];
      return {
        action: action.dataSync(),
        grip: grip.dataSync()[0],
        attn: attn.dataSync(),
      };
    });
    // the UI's gaze point: the map's expectation in plain [0,1] image coords,
    // computed here CPU-side (the in-graph attn_grid kernel feeds the action
    // head CENTERED+GAINED coords — see config.ts attnCoordGain — so it isn't
    // directly displayable). Must use the RAW softmax map, before the peak
    // normalization below.
    const G = ATTN_GRID;
    let ux = 0;
    let vy = 0;
    let peak = 0;
    for (let i = 0; i < out.attn.length; i++) {
      const w = out.attn[i];
      ux += w * ((i % G) + 0.5);
      vy += w * (Math.floor(i / G) + 0.5);
      if (w > peak) peak = w;
    }
    // normalize the map so the peak cell is 1 — the UI uses it as alpha
    const inv = peak > 0 ? 1 / peak : 0;
    return {
      target: [out.action[0], out.action[1]],
      grip: out.grip,
      attn: Array.from(out.attn, (a) => a * inv),
      xy: [ux / G, vy / G],
    };
  }

  /**
   * Decode what a token sequence asks for — task, acted-on color and (for
   * stack) the reference color — via the auxiliary language heads (which
   * read only the language branch — vision input is zeros).
   */
  decodeCommand(tokens: number[]): DecodedCommand | null {
    if (!this.ready) return null;
    const tf = this.tf!;
    const out = tf.tidy(() => {
      const v = tf.zeros([1, IMG_SIZE, IMG_SIZE, 3]);
      const l = tf.tensor2d([tokens], [1, MAX_SEQ_LEN], "int32");
      const [, , color, ref, task] = this.models!.model.predict([
        v,
        l,
      ]) as tfType.Tensor[];
      return {
        color: color.dataSync(),
        ref: ref.dataSync(),
        task: task.dataSync(),
      };
    });
    const argmax = (a: Float32Array | Int32Array | Uint8Array) => {
      let best = 0;
      for (let i = 1; i < a.length; i++) if (a[i] > a[best]) best = i;
      return best;
    };
    const c = argmax(out.color);
    const r = argmax(out.ref);
    const t = argmax(out.task);
    return {
      task: TASKS[t],
      taskProb: out.task[t],
      color: c,
      colorProb: out.color[c],
      refColor: r === REF_NONE ? null : r,
      refProb: out.ref[r],
    };
  }

  /**
   * The language encoder's per-token ATTENTION weights — the live per-chip
   * bars. The pooling scorer is a single LINEAR dense layer over the frozen
   * embeddings, so the exact masked-softmax weights the model computes
   * internally can be recomputed CPU-side here: score each token as
   * embedding[token] · scoreKernel + scoreBias, drop padding, softmax over
   * the rest. No forward pass — just the scorer's two small weight tensors
   * plus a dot product per token against the frozen table. Weights are
   * normalized so the most-attended token fills its bar.
   */
  attentionWeights(tokens: number[]): number[] | null {
    if (!this.ready) return null;
    // the embedding table is FROZEN pretrained GloVe — read it from the
    // CPU-side copy loadEmbeddings() kept (syncing the 20k x 50 table off
    // the GPU every refresh would dwarf the readout it powers); only the
    // small trainable scorer weights are pulled from the model
    const embedTable = embeddingMatrix();
    if (!embedTable) return null;
    const w = this.models!.model.getLayer("attn_score").getWeights();
    const kernel = w[0].arraySync() as number[][]; // [EMBED_DIM][1]
    const bias = (w[1].arraySync() as number[])[0];
    // raw scores; padding is excluded from the softmax entirely
    const scores = tokens.map((tok) => {
      if (tok === PAD) return -Infinity;
      let s = bias;
      for (let d = 0; d < EMBED_DIM; d++)
        s += embedTable[tok * EMBED_DIM + d] * kernel[d][0];
      return s;
    });
    const finite = scores.filter((s) => Number.isFinite(s));
    const maxS = finite.length ? Math.max(...finite) : 0;
    let sum = 0;
    const exps = scores.map((s) => {
      if (!Number.isFinite(s)) return 0;
      const e = Math.exp(s - maxS);
      sum += e;
      return e;
    });
    const weights = exps.map((e) => (sum > 0 ? e / sum : 0));
    const maxW = Math.max(...weights, 1e-6);
    return weights.map((wt) => wt / maxW);
  }

  /** Loss normalized against the first batch, clamped to [0,1]. */
  lossNorm(): number {
    if (Number.isNaN(this.loss) || Number.isNaN(this.initialLoss)) return 1;
    if (this.initialLoss <= 0) return 0;
    return Math.max(0, Math.min(1, this.loss / this.initialLoss));
  }
}
