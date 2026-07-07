// Main-thread proxy for the VLA trainer. The actual training loop lives in
// lib/vla/trainer.core.ts and normally runs inside lib/vla/trainer.worker.ts
// (a module Web Worker with OffscreenCanvas + its own WebGL context), so
// gradient steps never contend with the hero's 60fps rAF loop. This class is
// what Hero.tsx talks to; it keeps the pre-worker VLATrainer surface:
//
//  - control (start/pause/resume/reset/snapshotPolicy) is fire-and-forget:
//    the proxy applies the same optimistic state transition the old in-thread
//    class did synchronously, posts the command, and re-syncs from the
//    worker's authoritative {t:"state"} echo.
//  - telemetry (status/loss/lossHistory/batches/ready/lossNorm) stays a
//    plain synchronous read, served from a mirror updated per batch message.
//    lossHistory is reconstructed by appending each batch's loss — only a
//    few numbers cross the thread boundary per step, never the full curve.
//  - inference (predictFrozenTarget/decodeColor/attentionWeights) is now
//    ASYNC (Promise), matched to worker replies by request id. The rollout
//    already re-predicts on a throttle and steps toward its LAST target every
//    frame, so the added round-trip latency is invisible (see Hero.tsx).
//
// A `gen` counter increments on every reset and is echoed on each state post,
// so batch updates still in flight when the mirror was cleared are dropped
// instead of repopulating it.
//
// On browsers without module-worker/OffscreenCanvas support the proxy falls
// back to running VLATrainerCore inline on the main thread (the pre-worker
// behavior, jank and all) behind the same async API.

import { CONFIG } from "./config";
import { registerFullVocab, type Layout } from "./examples";
import {
  VLATrainerCore,
  type PredictResult,
  type TrainerStatus,
} from "./trainer.core";
import type { WorkerRequest, WorkerResponse } from "./trainer.worker";

export type { PredictResult, TrainerStatus };

const BATCH_SIZE = CONFIG.trainer.batchSize;

/** Omit that distributes over a union (plain Omit collapses it to the
    members' common properties, losing the per-variant fields). */
type DistributiveOmit<T, K extends keyof T> = T extends unknown
  ? Omit<T, K>
  : never;

/** True when the trainer can host itself off the main thread. */
function workerSupported() {
  return (
    typeof Worker !== "undefined" && typeof OffscreenCanvas !== "undefined"
  );
}

export class VLATrainer {
  /** Inline fallback core (only when workerSupported() is false). */
  private core: VLATrainerCore | null = null;
  private worker: Worker | null = null;
  private onUpdate: (() => void) | undefined;
  /** Reset generation — state posts from an older gen are stale, drop them. */
  private gen = 0;
  private nextId = 1;
  private pending = new Map<number, (result: never) => void>();

  // Mirror of the worker-side core's telemetry (worker mode only; in inline
  // mode every accessor delegates to the core directly).
  private m = {
    status: "idle" as TrainerStatus,
    loss: NaN,
    smoothLoss: NaN,
    initialLoss: NaN,
    lossHistory: [] as number[],
    batches: 0,
  };

  get status() {
    return this.core ? this.core.status : this.m.status;
  }
  /** Real action loss (Huber) from the latest batch (NaN before the first). */
  get loss() {
    return this.core ? this.core.loss : this.m.loss;
  }
  /** Trailing-window mean action loss (the convergence signal). */
  get smoothLoss() {
    return this.core ? this.core.smoothLoss : this.m.smoothLoss;
  }
  /** First recorded loss — the normalization anchor for the UI. */
  get initialLoss() {
    return this.core ? this.core.initialLoss : this.m.initialLoss;
  }
  get lossHistory() {
    return this.core ? this.core.lossHistory : this.m.lossHistory;
  }
  get batches() {
    return this.core ? this.core.batches : this.m.batches;
  }
  /** Total expert-labeled examples consumed so far. */
  get samples() {
    return this.core ? this.core.samples : this.m.batches * BATCH_SIZE;
  }
  get ready() {
    if (this.core) return this.core.ready;
    return (
      (this.m.status === "training" ||
        this.m.status === "paused" ||
        this.m.status === "converged") &&
      this.m.batches > 0
    );
  }

  /** Loss normalized against the first batch, clamped to [0,1]. */
  lossNorm(): number {
    if (this.core) return this.core.lossNorm();
    const { loss, initialLoss } = this.m;
    if (Number.isNaN(loss) || Number.isNaN(initialLoss)) return 1;
    if (initialLoss <= 0) return 0;
    return Math.max(0, Math.min(1, loss / initialLoss));
  }

  private ensureBackend() {
    if (this.core || this.worker) return;
    if (!workerSupported()) {
      this.core = new VLATrainerCore();
      return;
    }
    this.worker = new Worker(
      new URL("./trainer.worker.ts", import.meta.url),
      { type: "module" }
    );
    // If the worker script fails to load/evaluate (stale chunk, unsupported
    // browser, etc.) or posts something structured-clone can't handle, there
    // is otherwise no signal at all — status would sit at "loading" forever.
    // Recover the same way a failed embeddings fetch does: back to idle so
    // "Start Training" reappears and a retry gets a fresh worker.
    this.worker.onerror = (ev) => {
      console.error("VLA trainer worker failed:", ev.message || ev);
      this.handleWorkerFailure();
    };
    this.worker.onmessageerror = () => {
      console.error("VLA trainer worker sent an unclonable message");
      this.handleWorkerFailure();
    };
    this.worker.onmessage = (e: MessageEvent<WorkerResponse>) => {
      const msg = e.data;
      switch (msg.t) {
        case "state": {
          if (msg.gen !== this.gen) return; // pre-reset message in flight
          if (msg.batches > this.m.batches) this.m.lossHistory.push(msg.loss);
          this.m.status = msg.status;
          this.m.loss = msg.loss;
          this.m.smoothLoss = msg.smoothLoss;
          this.m.initialLoss = msg.initialLoss;
          this.m.batches = msg.batches;
          this.onUpdate?.();
          break;
        }
        case "vocab":
          // the worker's embedding load registered the 20k-word list with ITS
          // examples.ts; mirror that into this thread's tokenizer so typed
          // near-synonyms resolve to real token ids in "try it" mode
          registerFullVocab(msg.words);
          break;
        case "predict":
        case "predictLive":
          this.take(msg.id)?.(msg.result as never);
          break;
        case "decode":
          this.take(msg.id)?.(msg.result as never);
          break;
        case "attention":
          this.take(msg.id)?.(msg.result as never);
          break;
      }
    };
  }

  private take(id: number) {
    const resolve = this.pending.get(id);
    this.pending.delete(id);
    return resolve;
  }

  /** Worker died (load error or unclonable message) — drop it, resolve any
      in-flight requests with null so callers don't hang, and go back to idle
      so the UI recovers instead of sitting on "loading" forever. */
  private handleWorkerFailure() {
    this.worker?.terminate();
    this.worker = null;
    this.gen++; // orphan any reply that somehow still lands
    this.pending.forEach((resolve) => resolve(null as never));
    this.pending.clear();
    this.m.status = "idle";
    this.m.loss = NaN;
    this.m.smoothLoss = NaN;
    this.m.initialLoss = NaN;
    this.m.lossHistory = [];
    this.m.batches = 0;
    this.onUpdate?.();
  }

  private send(msg: WorkerRequest) {
    this.worker!.postMessage(msg);
  }

  /** Ask the worker and get a typed reply matched by request id. */
  private request<T>(
    msg: DistributiveOmit<Extract<WorkerRequest, { id: number }>, "id" | "gen">
  ): Promise<T> {
    return new Promise<T>((resolve) => {
      const id = this.nextId++;
      this.pending.set(id, resolve as (result: never) => void);
      this.send({ ...msg, id, gen: this.gen } as WorkerRequest);
    });
  }

  /**
   * Kick off training (loads tfjs + embeddings in the worker on first call).
   * onUpdate fires after every batch and on status changes, same as before.
   */
  start(onUpdate?: () => void) {
    this.ensureBackend();
    this.onUpdate = onUpdate;
    if (this.core) {
      void this.core.start(onUpdate);
      return;
    }
    if (this.m.status !== "idle") return;
    this.m.status = "loading";
    this.send({ t: "start", gen: this.gen });
    onUpdate?.();
  }

  /** Halt gradient steps without touching the model (Resume continues). */
  pause() {
    if (this.core) return this.core.pause();
    if (this.m.status !== "training") return;
    this.m.status = "paused";
    this.send({ t: "pause", gen: this.gen });
  }

  resume() {
    if (this.core) return this.core.resume();
    if (this.m.status !== "paused") return;
    this.m.status = "training";
    this.send({ t: "resume", gen: this.gen });
  }

  /** Stop training and discard the learned weights (fresh model next start).
      The worker itself stays warm — tfjs + embeddings stay cached, so a
      restart skips straight to model build. */
  reset() {
    if (this.core) return this.core.reset();
    this.gen++;
    this.m.status = "idle";
    this.m.loss = NaN;
    this.m.smoothLoss = NaN;
    this.m.initialLoss = NaN;
    this.m.lossHistory = [];
    this.m.batches = 0;
    // in-flight prediction replies will still arrive and resolve; anything
    // queued after this reset computes against a disposed model → null
    if (this.worker) this.send({ t: "reset", gen: this.gen });
  }

  /** Tear the worker down entirely (component unmount). */
  destroy() {
    this.reset();
    this.worker?.terminate();
    this.worker = null;
    this.pending.clear();
    this.core = null;
  }

  /** Freeze the current policy weights for the per-cycle rollout snapshot.
      Fire-and-forget; postMessage FIFO guarantees it lands before any predict
      request posted after it. */
  snapshotPolicy() {
    if (this.core) return this.core.snapshotPolicy();
    if (!this.ready) return;
    this.send({ t: "snapshot", gen: this.gen });
  }

  /**
   * Policy inference against the frozen per-cycle snapshot — async (the
   * render + forward pass happen in the worker). Replies with the target
   * angles PLUS the spatial-attention readout from the same pass. The caller
   * keeps stepping toward its previous target until the reply lands; see
   * Hero.tsx.
   */
  predictFrozenTarget(
    a1: number,
    a2: number,
    tokens: number[],
    layout: Layout
  ): Promise<PredictResult | null> {
    if (this.core)
      return Promise.resolve(this.core.predictFrozenTarget(a1, a2, tokens, layout));
    if (!this.ready) return Promise.resolve(null);
    return this.request({ t: "predict", a1, a2, tokens, layout });
  }

  /**
   * Same inference, but on the LIVE (still-training) model — powers the
   * Vision Encoder panel's "where the model looks" heatmap during training,
   * queried against the demonstration's current state.
   */
  predictLive(
    a1: number,
    a2: number,
    tokens: number[],
    layout: Layout
  ): Promise<PredictResult | null> {
    if (this.core)
      return Promise.resolve(this.core.predictTarget(a1, a2, tokens, layout));
    if (!this.ready) return Promise.resolve(null);
    return this.request({ t: "predictLive", a1, a2, tokens, layout });
  }

  /** Decode which color a token sequence names (auxiliary language head). */
  decodeColor(tokens: number[]): Promise<{ color: number; prob: number } | null> {
    if (this.core) return Promise.resolve(this.core.decodeColor(tokens));
    if (!this.ready) return Promise.resolve(null);
    return this.request({ t: "decode", tokens });
  }

  /** The language encoder's per-token attention weights (live token bars). */
  attentionWeights(tokens: number[]): Promise<number[] | null> {
    if (this.core) return Promise.resolve(this.core.attentionWeights(tokens));
    if (!this.ready) return Promise.resolve(null);
    return this.request({ t: "attention", tokens });
  }
}
