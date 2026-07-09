// VLA network definition. TensorFlow.js is passed in (and only ever loaded
// via dynamic import in trainer.ts) so this module stays SSR-safe and the
// ~1MB tfjs bundle is fetched lazily on "Start Training".
//
// Vision→action is a language-conditioned SPATIAL ATTENTION readout, not a
// flatten→dense fusion. The conv stack produces a G×G feature map; a single
// language query dot-product-scores every map cell ("does this cell look
// like the commanded block?"); a spatial softmax turns the score row into an
// attention map; and the readout is the map's SOFT-ARGMAX — the expected
// (x, y) image coordinate under the map — plus its attention-weighted feature
// vector (block size / local shape). A small dense head then regresses the
// target joint angles from that readout.
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
// Language: ONE attention-pooled slot with a LOCAL-CONTEXT (conv) scorer.
// A conv1d scorer (kernel 7 over the token axis, linear) scores every token
// from itself PLUS its ±3-token neighborhood, and attention-pools the
// sequence into a single vector the color head decodes and the attention
// query reads. An attention pool (not a mean-pool) because a mean dilutes
// the one word that matters under filler + padding; a conv (not a bare
// per-token dense) so the scorer generalizes across compressed forms
// ("grab red", bare "red") the dropVerb/Article/NounProb grammar
// augmentation in config.ts trains it on, rather than overfitting the full
// grammar's fixed color-word position.
// The embedding table is PRETRAINED (a ~20k-word GloVe 50d slice, see
// lib/vla/embeddings.ts) and FROZEN: only the scorer/head/fusion fine-tune
// on top of it. Frozen is the point — Adam would only update rows for words
// seen in training, so a trainable table would drift "golden" away from the
// untouched "gold" and destroy exactly the near-synonym generalization the
// pretrained geometry provides. The linear color head learns a map from
// GloVe space using only the grammar's synonyms, and unseen neighbors
// ("gold", "violet") ride along, powering the live "decoded target" readout.
// The token-attention scorer is LINEAR (conv1d, no activation), so
// trainer.core's attentionWeights() can recompute the exact same per-token
// weights CPU-side (the small conv kernel + the frozen embedding table —
// PAD rows are zero, matching the conv's "same" zero padding) for the live
// per-token bars — no extra sub-model or forward pass required.
//
// PROPRIOCEPTIVE CARRY FLAG: a third input (1 scalar) tells the network
// whether the gripper is currently holding a block. WHY: the same (scene,
// sentence) has two far-apart action targets — the grasp point when empty-
// handed, the carry-home (REST) target when holding — and pre-flag the ONLY
// disambiguator was the carried block's pixels at the effector; when vision
// missed that cue the action head averaged the modes (a 0.12-0.43 loss
// oscillation). The flag feeds the attention QUERY too: "block at the
// effector" changes what the commanded block looks like. Honesty note: this
// is gripper proprioception (a sensed STATE), which a real robot has — and it
// is distinct from the gripper COMMAND head (see gripperOutput below): the
// policy commands the gripper to close AND senses whether it is holding,
// exactly as a real robot does. No expert-side information crosses at inference.
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
import { EMBED_DIM } from "./vocab.gen";

export type TF = typeof tfType;

// Every architecture/optimizer knob below is tuned in lib/vla/config.ts; the
// rationale for each value is documented there.
export const IMG_SIZE = CONFIG.model.imgSize;
export const LEARNING_RATE = CONFIG.model.learningRate;
export const COLOR_LOSS_WEIGHT = CONFIG.model.colorLossWeight;
export const MAP_LOSS_WEIGHT = CONFIG.model.mapLossWeight;
export const GRIPPER_LOSS_WEIGHT = CONFIG.model.gripperLossWeight;
export const ACTION_HUBER_DELTA = CONFIG.model.actionHuberDelta;

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
  /** Main policy (the one that trains): [vision, tokens, carry] →
      [action angles (2), color softmax, attention map [G*G], gripper (1)]. The
      map is a trained OUTPUT, not just a readout — see mapLossWeight in
      config.ts for why the action loss alone can't train the attention. color
      reads the pooled language slot — a pure text decoder. gripper is the
      learned "close now" command (last output; see below). */
  model: tfType.LayersModel;
  /** Inference/readout twin sharing every layer (no weights of its own):
      same inputs → [action angles (2), attention map [G*G], gripper (1)]. One
      predict on this yields the action, the "where is the model looking" viz
      AND the gripper command in a single pass (trainer.core computes the map
      expectation CPU-side). */
  viz: tfType.LayersModel;
  /** Language-only training twin: [tokens] → [color]. Shares the embedding +
      conv scorer + color head with `model` (no weights of its own), but its
      graph EXCLUDES the vision branch, so trainOnBatch on it updates ONLY the
      language weights — the basis for the Loading-phase warm-up (see
      languageWarmup in trainer.core). It carries its OWN optimizer (compiled
      below), independent of the main model's. */
  lang: tfType.LayersModel;
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
  // the pooled language slots, so they have to exist before the CNN is wired.
  const langInput = tf.input({
    shape: [MAX_SEQ_LEN],
    name: "language_tokens",
    dtype: "int32",
  });
  // proprioceptive "gripper is holding a block" flag (see header)
  const carryInput = tf.input({ shape: [1], name: "carry_flag" });
  const embedded = tf.layers
    .embedding({
      inputDim: VOCAB_SIZE,
      outputDim: EMBED_DIM,
      trainable: false, // frozen pretrained backbone (see header)
      name: "text_embedding",
    })
    .apply(langInput) as tfType.SymbolicTensor; // [T, EMBED_DIM]
  // per-token scores from the token PLUS its ±3 neighborhood (see header —
  // kernel 7 covers the widest grammar gap with headroom for free text).
  // LINEAR (no activation) so attentionWeights() can recompute the same
  // scores CPU-side from the conv kernel. "attn_score"/"lang_vector" keep
  // their names so the existing per-token-bars readout stays wired.
  const scoresTarget = tf.layers
    .conv1d({ filters: 1, kernelSize: 7, padding: "same", name: "attn_score" })
    .apply(embedded) as tfType.SymbolicTensor; // [T, 1]
  const AttentionPooling = makeAttentionPooling(tf);
  const langTarget = new AttentionPooling({ name: "lang_vector" }).apply([
    embedded,
    scoresTarget,
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
  // ONE language-conditioned query over feature space — the attention map.
  // LINEAR query: the dot-product score is bilinear in (features, language) —
  // the simplest learnable "does this cell match the commanded block" test;
  // the kernel is rescaled by 1/√C post-build so the softmax starts soft (see
  // below). The map's job is phase-INDEPENDENT: it tracks the commanded block
  // wherever it renders (floor spot, or the effector while carried). The
  // carry flag rides into the query: "block at the effector" changes what the
  // commanded block looks like.
  const pickQuery = tf.layers
    .dense({ units: C, name: "pick_query" })
    .apply(
      tf.layers
        .concatenate({ name: "pick_query_in" })
        .apply([langTarget, carryInput]) as tfType.SymbolicTensor
    ) as tfType.SymbolicTensor; // [B, C]
  // per-cell match scores → spatial softmax ("where the model looks", also
  // posted to the UI)
  const pickMap = tf.layers
    .activation({ activation: "softmax", name: "pick_map" })
    .apply(
      tf.layers
        .dot({ axes: [2, 1], name: "pick_cell_scores" })
        .apply([cells, pickQuery]) as tfType.SymbolicTensor
    ) as tfType.SymbolicTensor; // [B, G*G]
  // soft-argmax: expected (x, y) coordinate under the map. A Dense with a
  // FROZEN kernel holding each cell's center (seeded post-build) IS that
  // expectation — no custom layer needed, and gradients still flow through
  // the attention map to the convs/query. The kernel stores CENTERED, GAINED
  // coords ((c − 0.5) × attnCoordGain) — see config.ts.
  const pickXY = tf.layers
    .dense({ units: 2, useBias: false, trainable: false, name: "pick_grid" })
    .apply(pickMap) as tfType.SymbolicTensor; // [B, 2], gained image coords
  // attention-weighted feature readout: block size / local shape at the
  // attended spot.
  const pickFeat = tf.layers
    .dot({ axes: [1, 1], name: "pick_read" })
    .apply([pickMap, cells]) as tfType.SymbolicTensor; // [B, C]

  // fusion + heads. The map readout feeds the action head; langTarget rides
  // along so the head keeps a direct language path; the carry flag rides
  // along so the action head needn't re-derive the phase from pixels (the
  // pre-flag mode-averaging failure). The color head reads the pooled
  // language slot — a pure text decoder powering the decoded-target readout.
  const fused = tf.layers
    .concatenate()
    .apply([
      pickXY,
      pickFeat,
      langTarget,
      carryInput,
    ] as tfType.SymbolicTensor[]);
  const dense1 = tf.layers
    .dense({ units: CONFIG.model.fusionUnits, activation: "relu" })
    .apply(fused);
  const actionOutput = tf.layers
    .dense({ units: 2, activation: "linear", name: "action" })
    .apply(dense1) as tfType.SymbolicTensor;
  const colorOutput = tf.layers
    .dense({ units: COLORS.length, activation: "softmax", name: "color" })
    .apply(langTarget) as tfType.SymbolicTensor;
  // gripper COMMAND: a learned "close now" head (0=open → 1=closed). It reads
  // the fused hidden dense1 — being "over the block" is a VISUAL fact, so it
  // needs the vision→attention pathway, not the language slot alone. Trained by
  // BCE against the shared effectorOverBlock predicate (trainer.core), it fires
  // ~0 while approaching and ~1 only when the effector is fully over the block;
  // the rollout turns its rising edge into the physical grasp (Hero.tsx). It is
  // APPENDED LAST so every existing positional output read stays valid. Kept OUT
  // of the lang twin (that graph has no vision node → no "over the block" fact).
  const gripperOutput = tf.layers
    .dense({ units: 1, activation: "sigmoid", name: "gripper" })
    .apply(dense1) as tfType.SymbolicTensor;

  const model = tf.model({
    inputs: [visionInput, langInput, carryInput],
    outputs: [actionOutput, colorOutput, pickMap, gripperOutput],
  });
  // readout twin: same graph nodes, so it shares every weight with `model`
  const viz = tf.model({
    inputs: [visionInput, langInput, carryInput],
    outputs: [actionOutput, pickMap, gripperOutput],
  });
  // language-only training twin: the color head reached from `langInput`
  // alone (no vision node in the graph), so a gradient step touches only the
  // language weights. Shares layers with `model`, compiled below with its own
  // optimizer for the Loading-phase warm-up.
  const lang = tf.model({
    inputs: [langInput],
    outputs: [colorOutput],
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
  model.getLayer("pick_grid").setWeights([gridInit]);
  gridInit.dispose();

  // temper the attention at init: scale the query kernel by 1/√C so the
  // initial cell scores are small and the softmax starts near-uniform —
  // a peaked random map at batch 0 would gradient-starve the losing cells.
  // Only the INIT is scaled; the kernel itself stays fully trainable.
  // (getWeights returns the layer's LIVE variable tensors — same rule as
  // snapshotPolicy: never dispose them; only the derived temp is ours.)
  {
    const q = model.getLayer("pick_query");
    const [qKernel, qBias] = q.getWeights();
    const qScaled = tf.tidy(() => qKernel.mul(1 / Math.sqrt(C)));
    q.setWeights([qScaled, qBias]); // copies the values into the variables
    qScaled.dispose();
  }

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
  // the attention supervision: which grid cell holds the commanded block
  // (trainer.core builds the label from the same layout the IK label comes
  // from — the commanded block wherever it renders). The map is already a
  // softmax, so plain categorical CE.
  const weightedMapLoss = (yTrue: tfType.Tensor, yPred: tfType.Tensor) =>
    tf.tidy(() =>
      tf.metrics.categoricalCrossentropy(yTrue, yPred).mul(MAP_LOSS_WEIGHT)
    );
  // gripper: plain binary cross-entropy on the sigmoid head, scaled down like
  // the other aux heads so it nudges (not dominates) the shared trunk. The
  // action Huber stays output 0 so trainStep's convergence read (loss index 1)
  // is unchanged.
  const weightedGripLoss = (yTrue: tfType.Tensor, yPred: tfType.Tensor) =>
    tf.tidy(() =>
      tf.metrics.binaryCrossentropy(yTrue, yPred).mul(GRIPPER_LOSS_WEIGHT)
    );
  model.compile({
    optimizer: tf.train.adam(LEARNING_RATE),
    loss: [actionLoss, weightedColorLoss, weightedMapLoss, weightedGripLoss],
  });
  // language warm-up twin: plain (unweighted) cross-entropy on the color head —
  // in isolation there's no action loss to balance against, so the config
  // loss-weight that exists only to scale the head down relative to the
  // action loss isn't needed here. Its own Adam so warm-up momentum doesn't
  // touch the main optimizer's state.
  lang.compile({
    optimizer: tf.train.adam(LEARNING_RATE),
    loss: "categoricalCrossentropy",
  });

  return { model, viz, lang };
}
