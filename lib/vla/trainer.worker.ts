// Web Worker entry hosting the VLA training loop (lib/vla/trainer.core.ts),
// so gradient steps, silhouette rendering (OffscreenCanvas) and inference all
// run OFF the main thread — the 60fps hero display never contends with
// training, which unblocks raising imgSize/batch throughput without jank.
//
// Counterpart: lib/vla/trainer.ts (the main-thread proxy Hero talks to).
// Protocol (all messages are plain structured-clone-able objects):
//
//   main → worker                        worker → main
//   {t:"start", gen, cfg}                {t:"state", gen, status, loss,
//   {t:"pause"|"resume"|"reset", gen}      smoothLoss, initialLoss, batches}
//   {t:"snapshot", gen}                    — after every batch + control msg
//   {t:"predict", id, a1, a2,            {t:"predict", id, result}
//     tokens, layout, carry, gen}        {t:"predictLive", id, result}
//   {t:"predictLive", id, a1, a2,        {t:"decode", id, result}
//     tokens, layout, carry, gen}        {t:"attention", id, result}
//   {t:"decode", id, tokens, gen}        {t:"vocab", words}
//   {t:"attention", id, tokens, gen}
//
// "start" ships the user's ⚙ RunConfig (task set / palette / density): the
// worker holds its own copy of the run-config module's state, so it must be
// installed here before the training loop samples anything. "predict" runs
// the FROZEN per-cycle snapshot (the rollout's policy); "predictLive" runs
// the still-training model (the Vision Encoder panel's live gaze heatmap).
// Both carry the rollout's carried-block state (rendered at the effector in
// the model's-eye view) and reply with a full PredictResult — target angles
// plus the spatial-attention map — from one forward pass. "decode" replies
// with the language heads' full DecodedCommand (task + color + ref color).
//
// `gen` is the proxy's reset-generation counter: it's echoed on every state
// post so the proxy can drop state messages that were already in flight when
// a reset cleared its mirror (otherwise a stale batch update would repopulate
// it). Request/response pairs are matched by `id` instead and need no gen.
// postMessage delivery is FIFO, so a "snapshot" posted before a "predict" is
// applied first — the ordering the per-cycle frozen-policy rollout relies on.
//
// tfjs + the GloVe embeddings load inside the worker on the first "start"
// (both stay lazy: the page never pays for them until the user clicks). The
// embeddings' word list is posted back as {t:"vocab"} because the main
// thread's tokenizer (examples.ts) needs registerFullVocab too — worker and
// page each have their own copy of that module's state.

import {
  VLATrainerCore,
  type DecodedCommand,
  type PredictResult,
} from "./trainer.core";
import { loadEmbeddings, vocabWords } from "./embeddings";
import type { Layout } from "./examples";
import { setRunConfig, type RunConfig } from "./run-config";

export type WorkerRequest =
  | { t: "start"; gen: number; cfg: RunConfig }
  | { t: "pause"; gen: number }
  | { t: "resume"; gen: number }
  | { t: "reset"; gen: number }
  | { t: "snapshot"; gen: number }
  | {
      t: "predict";
      id: number;
      a1: number;
      a2: number;
      tokens: number[];
      layout: Layout;
      carry: number | null;
      gen: number;
    }
  | {
      t: "predictLive";
      id: number;
      a1: number;
      a2: number;
      tokens: number[];
      layout: Layout;
      carry: number | null;
      gen: number;
    }
  | { t: "decode"; id: number; tokens: number[]; gen: number }
  | { t: "attention"; id: number; tokens: number[]; gen: number };

export type WorkerResponse =
  | {
      t: "state";
      gen: number;
      status: VLATrainerCore["status"];
      loss: number;
      smoothLoss: number;
      initialLoss: number;
      batches: number;
    }
  | { t: "predict"; id: number; result: PredictResult | null }
  | { t: "predictLive"; id: number; result: PredictResult | null }
  | { t: "decode"; id: number; result: DecodedCommand | null }
  | { t: "attention"; id: number; result: number[] | null }
  | { t: "vocab"; words: string[] };

const core = new VLATrainerCore();
const post = (m: WorkerResponse) => postMessage(m);

/** The latest generation seen from the proxy, echoed on state posts. */
let gen = 0;
let vocabSent = false;

const postState = () =>
  post({
    t: "state",
    gen,
    status: core.status,
    loss: core.loss,
    smoothLoss: core.smoothLoss,
    initialLoss: core.initialLoss,
    batches: core.batches,
  });

onmessage = (e: MessageEvent<WorkerRequest>) => {
  const msg = e.data;
  gen = msg.gen;
  switch (msg.t) {
    case "start":
      // install the run config BEFORE the loop samples anything — this
      // worker's run-config module state is separate from the main thread's
      setRunConfig(msg.cfg);
      // core.start resolves only when training ends; postState is its
      // per-batch onUpdate. The vocab rides along once the embeddings land.
      void core.start(postState);
      if (!vocabSent)
        void loadEmbeddings()
          .then(() => {
            const words = vocabWords();
            if (words) {
              vocabSent = true;
              post({ t: "vocab", words });
            }
          })
          .catch(() => {}); // start() itself surfaces load failures via status
      postState();
      break;
    case "pause":
      core.pause();
      postState();
      break;
    case "resume":
      core.resume();
      postState();
      break;
    case "reset":
      core.reset();
      postState();
      break;
    case "snapshot":
      core.snapshotPolicy();
      break;
    case "predict":
      post({
        t: "predict",
        id: msg.id,
        result: core.predictFrozenTarget(
          msg.a1,
          msg.a2,
          msg.tokens,
          msg.layout,
          msg.carry
        ),
      });
      break;
    case "predictLive":
      post({
        t: "predictLive",
        id: msg.id,
        result: core.predictTarget(
          msg.a1,
          msg.a2,
          msg.tokens,
          msg.layout,
          msg.carry
        ),
      });
      break;
    case "decode":
      post({ t: "decode", id: msg.id, result: core.decodeCommand(msg.tokens) });
      break;
    case "attention":
      post({
        t: "attention",
        id: msg.id,
        result: core.attentionWeights(msg.tokens),
      });
      break;
  }
};
