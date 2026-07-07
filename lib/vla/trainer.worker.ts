// Web Worker entry hosting the VLA training loop (lib/vla/trainer.core.ts),
// so gradient steps, silhouette rendering (OffscreenCanvas) and inference all
// run OFF the main thread — the 60fps hero display never contends with
// training, which unblocks raising imgSize/batch throughput without jank.
//
// Counterpart: lib/vla/trainer.ts (the main-thread proxy Hero talks to).
// Protocol (all messages are plain structured-clone-able objects):
//
//   main → worker                        worker → main
//   {t:"start", gen}                     {t:"state", gen, status, loss,
//   {t:"pause"|"resume"|"reset", gen}      smoothLoss, initialLoss, batches}
//   {t:"snapshot", gen}                    — after every batch + control msg
//   {t:"predict", id, a1, a2,            {t:"predict", id, target}
//     tokens, layout, gen}               {t:"decode", id, result}
//   {t:"decode", id, tokens, gen}        {t:"attention", id, result}
//   {t:"attention", id, tokens, gen}     {t:"vocab", words}
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

import { VLATrainerCore } from "./trainer.core";
import { loadEmbeddings, vocabWords } from "./embeddings";
import type { Layout } from "./examples";

export type WorkerRequest =
  | { t: "start"; gen: number }
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
  | { t: "predict"; id: number; target: [number, number] | null }
  | { t: "decode"; id: number; result: { color: number; prob: number } | null }
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
        target: core.predictFrozenTarget(msg.a1, msg.a2, msg.tokens, msg.layout),
      });
      break;
    case "decode":
      post({ t: "decode", id: msg.id, result: core.decodeColor(msg.tokens) });
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
