// VLA network definition. TensorFlow.js is passed in (and only ever loaded
// via dynamic import in trainer.ts) so this module stays SSR-safe and the
// ~1MB tfjs bundle is fetched lazily on "Start Training".
//
// Vision: a real 3-layer CNN over the 32x32 scene (conv → pool → conv →
// pool → strided conv), not a single patch projection — the policy must
// tell the two blocks in a scene apart by color AND regress both joint
// angles.
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
// The attention scorer is LINEAR, so trainer.ts's attentionWeights() can
// recompute the exact same per-token weights CPU-side (from the scorer's two
// small weight tensors + the frozen embedding table) for the live per-token
// bars — no extra sub-model or forward pass required.

import type * as tfType from "@tensorflow/tfjs";
import { MAX_SEQ_LEN, VOCAB_SIZE, COLORS } from "./examples";
import { EMBED_DIM } from "./vocab.gen";

export type TF = typeof tfType;

export const IMG_SIZE = 32;

/** Adam learning rate. Bumped from the GRU-era 0.004 now that the language
    branch is a much smaller bag-of-embeddings — the network has fewer
    parameters and a simpler loss surface, so it tolerates (and benefits
    from) a faster step size for quicker convergence. */
export const LEARNING_RATE = 0.007;
/** Weight of the auxiliary color-classification loss vs. the action loss. */
export const COLOR_LOSS_WEIGHT = 0.3;
/** Huber transition point for the action loss. The two IK target clusters
    (commanded block on the left vs. right) sit ~4.3 rad apart, so under plain
    MSE a single wrong-side pick costs ~9.3 — the loss ends up dominated by
    the rare (~1%) misclassification tail rather than regression precision,
    which both floors the loss near ~0.09 and makes its gradient thrash on
    outliers. Huber is quadratic below DELTA (keeps precise regression on the
    correct-side jitter, whose spread is ~0.1 rad) and LINEAR above it, capping
    a wrong-side pick at ~2.5 instead of ~9.3. That drops the same-accuracy
    floor to ~0.025 and smooths the descent. 0.6 keeps correct-side samples
    comfortably in the quadratic zone while catching wrong-side picks early. */
export const ACTION_HUBER_DELTA = 0.6;

export interface VLAModels {
  /** Main policy: [vision, tokens] → [action Δθ (2), color softmax]. */
  model: tfType.LayersModel;
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
  // Branch A: vision CNN
  const visionInput = tf.input({
    shape: [IMG_SIZE, IMG_SIZE, 3],
    name: "vision_pixels",
  });
  let v = tf.layers
    .conv2d({
      filters: 8,
      kernelSize: 3,
      padding: "same",
      activation: "relu",
      name: "conv1",
    })
    .apply(visionInput);
  v = tf.layers.maxPooling2d({ poolSize: 2 }).apply(v);
  v = tf.layers
    .conv2d({
      filters: 16,
      kernelSize: 3,
      padding: "same",
      activation: "relu",
      name: "conv2",
    })
    .apply(v);
  v = tf.layers.maxPooling2d({ poolSize: 2 }).apply(v);
  v = tf.layers
    .conv2d({
      filters: 24,
      kernelSize: 3,
      strides: 2,
      activation: "relu",
      name: "conv3",
    })
    .apply(v);
  const flatVision = tf.layers.flatten().apply(v); // 4x4x24 = 384

  // Branch B: language — attention-pooled bag-of-embeddings (no recurrence)
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

  // fusion + heads
  const fused = tf.layers
    .concatenate()
    .apply([flatVision, langPooled] as tfType.SymbolicTensor[]);
  const dense1 = tf.layers
    .dense({ units: 64, activation: "relu" })
    .apply(fused);
  const actionOutput = tf.layers
    .dense({ units: 2, activation: "linear", name: "action" })
    .apply(dense1) as tfType.SymbolicTensor;
  // aux head reads ONLY the language vector — usable as a pure text decoder
  const colorOutput = tf.layers
    .dense({ units: COLORS.length, activation: "softmax", name: "color" })
    .apply(langPooled) as tfType.SymbolicTensor;

  const model = tf.model({
    inputs: [visionInput, langInput],
    outputs: [actionOutput, colorOutput],
  });

  // load the pretrained GloVe vectors into the (frozen) embedding table.
  // setWeights copies the values into the layer's variable, so the temp
  // tensor is disposed right after.
  const embedInit = tf.tensor2d(embedMatrix, [VOCAB_SIZE, EMBED_DIM]);
  model.getLayer("text_embedding").setWeights([embedInit]);
  embedInit.dispose();
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
  model.compile({
    optimizer: tf.train.adam(LEARNING_RATE),
    loss: [actionLoss, weightedColorLoss],
  });

  return { model };
}
