// VLA network definition. TensorFlow.js is passed in (and only ever loaded
// via dynamic import in trainer.ts) so this module stays SSR-safe and the
// ~1MB tfjs bundle is fetched lazily on "Start Training".
//
// Vision→action is a language-conditioned SPATIAL ATTENTION readout, not a
// flatten→dense fusion. The conv stack produces a G×G feature map; the
// language vector is projected to a query which dot-product-scores every map
// cell ("does this cell look like what the command asked for?"); a spatial
// softmax turns the scores into an attention map; and the readout is the
// map's SOFT-ARGMAX — the expected (x, y) image coordinate under the map —
// plus the attention-weighted feature vector (block size / local shape). A
// small dense head then regresses the target joint angles from those.
//
// Why this shape: the previous architecture (FiLM-modulated CNN → flatten →
// dense) buried the vision→language binding inside a flatten that destroys
// spatial structure. The dense layer could satisfy the loss early by ignoring
// the command and regressing per-side mean targets — a shortcut basin that
// the 2026-07 sweep measured as ~9% of inits collapsing onto an always-one-
// side policy (plus a long "slow seed" tail), unfixable by any swept
// hyperparameter. Here the ONLY path from pixels to action runs through the
// attention map, and the map is driven by a language·feature dot product, so
// the "ignore the command" shortcut doesn't exist. The soft-argmax is an
// expectation, not a cell index, so position accuracy is no longer quantized
// by the feature-map resolution (the old ~0.03 reach-error floor).
//
// Language: an ATTENTION-POOLED bag-of-embeddings. Embed each token, score
// each one with a small learned linear scorer, then combine them with a
// masked softmax (padding forced to zero attention) into a single weighted-
// sum sentence vector. This replaces a plain mean-pool, which diluted the
// one word that matters (the color/verb) under filler + padding — in a
// 12-slot mean the color token is only ~1/12 of the result, and short
// commands are watered down further by the empty pad slots. Attention lets
// the encoder learn to keep the content words and ignore the scaffolding
// (still robust to WHERE the color word appears — free user text at
// inference; word order carries no signal, so no recurrence is needed). The
// embedding table is PRETRAINED (a ~20k-word GloVe 50d slice, see
// lib/vla/embeddings.ts) and FROZEN: only the scorer/heads/fusion fine-tune
// on top of it. Frozen is the point — Adam would only update rows for words
// seen in training, so a trainable table would drift "golden" away from the
// untouched "gold" and destroy exactly the near-synonym generalization the
// pretrained geometry provides. The linear color head learns a map from
// GloVe space using only the grammar's synonyms, and unseen neighbors
// ("gold", "violet") ride along. An auxiliary color-classification head on
// the pooled language vector shapes it to be color-decodable and powers the
// live "decoded target" readout.
// The token-attention scorer is LINEAR, so trainer.ts's attentionWeights()
// can recompute the exact same per-token weights CPU-side (from the scorer's
// two small weight tensors + the frozen embedding table) for the live
// per-token bars — no extra sub-model or forward pass required.
//
// Everything below is STANDARD tfjs layers (reshape/dot/softmax activation/
// dense) plus the one custom AttentionPooling layer the language branch
// already had — deliberately: custom layers under a dynamically-imported
// tfjs are the highest-runtime-risk construct in this stack. The spatial
// soft-argmax needs no custom code because the coordinate expectation is
// just a Dense with a FROZEN kernel holding each cell's (x, y) center.

import type * as tfType from "@tensorflow/tfjs";
import { CONFIG, type ConvLayer } from "./config";
import { MAX_SEQ_LEN, VOCAB_SIZE, COLORS } from "./examples";
import { TASKS } from "./run-config";
import { EMBED_DIM } from "./vocab.gen";

export type TF = typeof tfType;

// Every architecture/optimizer knob below is tuned in lib/vla/config.ts; the
// rationale for each value is documented there.
export const IMG_SIZE = CONFIG.model.imgSize;
export const LEARNING_RATE = CONFIG.model.learningRate;
export const COLOR_LOSS_WEIGHT = CONFIG.model.colorLossWeight;
export const MAP_LOSS_WEIGHT = CONFIG.model.mapLossWeight;
export const ACTION_HUBER_DELTA = CONFIG.model.actionHuberDelta;
export const TASK_LOSS_WEIGHT = CONFIG.model.taskLossWeight;
export const REF_COLOR_LOSS_WEIGHT = CONFIG.model.refColorLossWeight;

/** The ref-color head’s "none" class index (lift has no reference). */
export const REF_NONE = COLORS.length;

/** Spatial size after one conv stage (+ optional pool). */
function convOutSize(size: number, l: ConvLayer): number {
  const stride = l.stride ?? 1;
  size =
    l.padding === "valid"
      ? Math.floor((size - l.kernel) / stride) + 1
      : Math.ceil(size / stride);
  if (l.pool) size = Math.floor(size / 2);
  return size;
}

/** Side length G of the attention grid — the final conv map's spatial size.
    The attention map posted to the UI is G*G values, row-major. */
export const ATTN_GRID = CONFIG.model.conv.reduce(
  (s, l) => convOutSize(s, l),
  IMG_SIZE
);

export interface VLAModels {
  /** Main policy (the one that trains): [vision, tokens] →
      [action angles (2), color softmax, ref-color softmax (COLORS+1, last =
      "none"), task softmax, attention map [G*G]]. The map is a trained
      OUTPUT, not just a readout — see mapLossWeight in config.ts for why the
      action loss alone can't train the attention. color/refColor/task read
      only langPooled, so they stay pure text decoders. */
  model: tfType.LayersModel;
  /** Inference/readout twin sharing every layer (no weights of its own):
      [vision, tokens] → [action angles (2), spatial attention map [G*G]].
      One predict on this yields the action AND the "where is the model
      looking" viz in a single pass (the UI's gaze point is the map's
      expectation, computed CPU-side in trainer.core). */
  viz: tfType.LayersModel;
}

/**
 * A masked attention-pooling layer: given the token embeddings [B, T, D],
 * their per-token scores [B, T, 1], and the raw token ids [B, T], it masks
 * out padding (id 0), softmaxes the scores over the token axis, and returns
 * the attention-weighted sum [B, D]. Built as a factory (rather than a
 * top-level class) because it must subclass the `tf.layers.Layer` from the
 * dynamically-imported tfjs instance. It holds no weights of its own — the
 * trainable scorer is a separate Dense layer — so the two models built per
 * session (live + frozen snapshot) stay weight-order-compatible.
 */
function makeAttentionPooling(tf: TF) {
  return class AttentionPooling extends tf.layers.Layer {
    static className = "AttentionPooling";

    computeOutputShape(
      inputShape: tfType.Shape | tfType.Shape[]
    ): tfType.Shape | tfType.Shape[] {
      const emb = (inputShape as tfType.Shape[])[0] as number[];
      return [emb[0], emb[emb.length - 1]]; // [B, D]
    }

    call(
      inputs: tfType.Tensor | tfType.Tensor[]
    ): tfType.Tensor | tfType.Tensor[] {
      return tf.tidy(() => {
        const [embedded, scores, tokenIds] = inputs as tfType.Tensor[];
        const mask = tf.cast(tf.notEqual(tokenIds, 0), "float32"); // [B, T]
        const s = tf.squeeze(scores, [2]); // [B, T]
        // pad positions get -1e9 before the softmax, so they take ~0 weight
        const masked = tf.add(s, tf.mul(tf.sub(mask, 1), 1e9));
        const weights = tf.softmax(masked, -1); // [B, T]
        return tf.sum(tf.mul(embedded, tf.expandDims(weights, -1)), 1); // [B, D]
      });
    }
  };
}

/**
 * @param embedMatrix Dequantized pretrained GloVe table,
 *   [VOCAB_SIZE, EMBED_DIM] row-major (from lib/vla/embeddings.ts).
 */
export function buildVLAModel(tf: TF, embedMatrix: Float32Array): VLAModels {
  // Language branch first — the vision branch's attention query consumes
  // langPooled, so the sentence vector has to exist before the CNN is wired.
  const langInput = tf.input({
    shape: [MAX_SEQ_LEN],
    name: "language_tokens",
    dtype: "int32",
  });
  const embedded = tf.layers
    .embedding({
      inputDim: VOCAB_SIZE,
      outputDim: EMBED_DIM,
      trainable: false, // frozen pretrained backbone (see header)
      name: "text_embedding",
    })
    .apply(langInput) as tfType.SymbolicTensor; // [T, EMBED_DIM]
  // per-token importance score; LINEAR (no activation) so attentionWeights()
  // can recompute the same scores CPU-side from this layer's two weights
  const scores = tf.layers
    .dense({ units: 1, name: "attn_score" })
    .apply(embedded) as tfType.SymbolicTensor; // [T, 1]
  const AttentionPooling = makeAttentionPooling(tf);
  const langPooled = new AttentionPooling({ name: "lang_vector" }).apply([
    embedded,
    scores,
    langInput,
  ]) as tfType.SymbolicTensor; // EMBED_DIM

  // Vision CNN. The stack is data-driven from CONFIG.model.conv (edit that to
  // change depth / kernel sizes / channels); plain relu convs — the language
  // conditioning happens in the attention readout below, not mid-CNN.
  const visionInput = tf.input({
    shape: [IMG_SIZE, IMG_SIZE, 3],
    name: "vision_pixels",
  });
  let v: tfType.SymbolicTensor = visionInput;
  CONFIG.model.conv.forEach((layer, i) => {
    v = tf.layers
      .conv2d({
        filters: layer.filters,
        kernelSize: layer.kernel,
        strides: layer.stride ?? 1,
        padding: layer.padding ?? "same",
        activation: "relu",
        name: `conv${i + 1}`,
      })
      .apply(v) as tfType.SymbolicTensor;
    if (layer.pool)
      v = tf.layers
        .maxPooling2d({ poolSize: 2 })
        .apply(v) as tfType.SymbolicTensor;
  });

  // ── language-conditioned spatial attention + soft-argmax readout ────────
  const G = ATTN_GRID;
  const C = CONFIG.model.conv[CONFIG.model.conv.length - 1].filters;
  // the G×G map as a sequence of cells, row-major (i*G + j), features last
  const cells = tf.layers
    .reshape({ targetShape: [G * G, C], name: "vision_cells" })
    .apply(v) as tfType.SymbolicTensor; // [B, G*G, C]
  // the command as a query over feature space. LINEAR: the dot-product score
  // is then bilinear in (features, language) — the simplest learnable "does
  // this cell match the command" test. Kernel is rescaled by 1/√C post-build
  // so the softmax starts soft (see below).
  const query = tf.layers
    .dense({ units: C, name: "attn_query" })
    .apply(langPooled) as tfType.SymbolicTensor; // [B, C]
  // per-cell match score: contract the feature axis of both inputs
  const cellScores = tf.layers
    .dot({ axes: [2, 1], name: "attn_cell_scores" })
    .apply([cells, query]) as tfType.SymbolicTensor; // [B, G*G]
  // spatial softmax — "where the model looks", also posted to the UI
  const attnMap = tf.layers
    .activation({ activation: "softmax", name: "attn_map" })
    .apply(cellScores) as tfType.SymbolicTensor; // [B, G*G]
  // soft-argmax: expected (x, y) coordinate under the map. A Dense with a
  // FROZEN kernel holding each cell's center (seeded post-build) IS that
  // expectation — no custom layer needed, and gradients still flow through
  // the attention map to the convs/query. The kernel stores CENTERED, GAINED
  // coords ((c − 0.5) × attnCoordGain), not raw [0,1] — see config.ts.
  const attnXY = tf.layers
    .dense({ units: 2, useBias: false, trainable: false, name: "attn_grid" })
    .apply(attnMap) as tfType.SymbolicTensor; // [B, 2], gained image coords
  // attention-weighted feature readout: block size / local shape at the
  // attended spot (the grasp height depends on block size, which (x̂, ŷ)
  // alone doesn't carry)
  const attended = tf.layers
    .dot({ axes: [1, 1], name: "attn_read" })
    .apply([attnMap, cells]) as tfType.SymbolicTensor; // [B, C]

  // fusion + heads. langPooled rides along so the head keeps a direct
  // language path; the color aux head reads ONLY langPooled, so it stays a
  // pure text decoder and the live "decoded target" readout is unchanged.
  const fused = tf.layers
    .concatenate()
    .apply([attnXY, attended, langPooled] as tfType.SymbolicTensor[]);
  const dense1 = tf.layers
    .dense({ units: CONFIG.model.fusionUnits, activation: "relu" })
    .apply(fused);
  const actionOutput = tf.layers
    .dense({ units: 2, activation: "linear", name: "action" })
    .apply(dense1) as tfType.SymbolicTensor;
  const colorOutput = tf.layers
    .dense({ units: COLORS.length, activation: "softmax", name: "color" })
    .apply(langPooled) as tfType.SymbolicTensor;
  // stack’s second referent ("…on the X block"); REF_NONE for lift
  const refColorOutput = tf.layers
    .dense({ units: COLORS.length + 1, activation: "softmax", name: "ref_color" })
    .apply(langPooled) as tfType.SymbolicTensor;
  const taskOutput = tf.layers
    .dense({ units: TASKS.length, activation: "softmax", name: "task" })
    .apply(langPooled) as tfType.SymbolicTensor;

  const model = tf.model({
    inputs: [visionInput, langInput],
    outputs: [actionOutput, colorOutput, refColorOutput, taskOutput, attnMap],
  });
  // readout twin: same graph nodes, so it shares every weight with `model`
  const viz = tf.model({
    inputs: [visionInput, langInput],
    outputs: [actionOutput, attnMap],
  });

  // load the pretrained GloVe vectors into the (frozen) embedding table.
  // setWeights copies the values into the layer's variable, so the temp
  // tensor is disposed right after.
  const embedInit = tf.tensor2d(embedMatrix, [VOCAB_SIZE, EMBED_DIM]);
  model.getLayer("text_embedding").setWeights([embedInit]);
  embedInit.dispose();

  // seed the frozen soft-argmax kernel: row i*G+j holds cell (i, j)'s center,
  // centered and gained (see attnCoordGain in config.ts) — column 0 = x
  // (j across), column 1 = y (i down, matching the silhouette's canvas
  // orientation)
  const gain = CONFIG.model.attnCoordGain;
  const grid = new Float32Array(G * G * 2);
  for (let i = 0; i < G; i++)
    for (let j = 0; j < G; j++) {
      grid[(i * G + j) * 2] = ((j + 0.5) / G - 0.5) * gain;
      grid[(i * G + j) * 2 + 1] = ((i + 0.5) / G - 0.5) * gain;
    }
  const gridInit = tf.tensor2d(grid, [G * G, 2]);
  model.getLayer("attn_grid").setWeights([gridInit]);
  gridInit.dispose();

  // temper the attention at init: scale the query kernel by 1/√C so the
  // initial cell scores are small and the softmax starts near-uniform —
  // a peaked random map at batch 0 would gradient-starve the losing cells.
  // Only the INIT is scaled; the kernel itself stays fully trainable.
  // (getWeights returns the layer's LIVE variable tensors — same rule as
  // snapshotPolicy: never dispose them; only the derived temp is ours.)
  const q = model.getLayer("attn_query");
  const [qKernel, qBias] = q.getWeights();
  const qScaled = tf.tidy(() => qKernel.mul(1 / Math.sqrt(C)));
  q.setWeights([qScaled, qBias]); // copies the values into the variables
  qScaled.dispose();

  // tfjs-layers doesn't implement compile({lossWeights}) — scale the aux
  // color loss inside a custom per-output loss function instead (and the
  // loss array must then be all-functions, so the action loss is a function
  // too). The action loss is Huber, not MSE: the wrong-side outlier tail (see
  // ACTION_HUBER_DELTA) otherwise dominates both the floor and the gradient.
  const actionLoss = (yTrue: tfType.Tensor, yPred: tfType.Tensor) =>
    tf.losses.huberLoss(yTrue, yPred, undefined, ACTION_HUBER_DELTA);
  const weightedColorLoss = (yTrue: tfType.Tensor, yPred: tfType.Tensor) =>
    tf.tidy(() =>
      tf.metrics.categoricalCrossentropy(yTrue, yPred).mul(COLOR_LOSS_WEIGHT)
    );
  const weightedRefColorLoss = (yTrue: tfType.Tensor, yPred: tfType.Tensor) =>
    tf.tidy(() =>
      tf.metrics.categoricalCrossentropy(yTrue, yPred).mul(REF_COLOR_LOSS_WEIGHT)
    );
  const weightedTaskLoss = (yTrue: tfType.Tensor, yPred: tfType.Tensor) =>
    tf.tidy(() =>
      tf.metrics.categoricalCrossentropy(yTrue, yPred).mul(TASK_LOSS_WEIGHT)
    );
  // the attention supervision: which grid cell holds the block the policy
  // should currently be homing on (trainer.core builds the label from the
  // same layout the IK label comes from — the commanded block, or the
  // reference block during stack's carry phase). attnMap is already a
  // softmax, so plain categorical CE.
  const weightedMapLoss = (yTrue: tfType.Tensor, yPred: tfType.Tensor) =>
    tf.tidy(() =>
      tf.metrics.categoricalCrossentropy(yTrue, yPred).mul(MAP_LOSS_WEIGHT)
    );
  model.compile({
    optimizer: tf.train.adam(LEARNING_RATE),
    loss: [
      actionLoss,
      weightedColorLoss,
      weightedRefColorLoss,
      weightedTaskLoss,
      weightedMapLoss,
    ],
  });

  return { model, viz };
}
