// VLA network definition. TensorFlow.js is passed in (and only ever loaded
// via dynamic import in trainer.ts) so this module stays SSR-safe and the
// ~1MB tfjs bundle is fetched lazily on "Start Training".
//
// Vision: a real 3-layer CNN over the 32x32 scene (conv → pool → conv →
// pool → strided conv), not a single patch projection — the policy must
// tell the two blocks in a scene apart by color AND regress both joint
// angles.
// Language: a bag-of-embeddings — embed each token, then mean-pool across
// the sequence (robust to WHERE the color word appears — free user text at
// inference; word order carries no signal for this task, so a recurrent
// reader isn't needed and costs meaningfully more to run/compile). An
// auxiliary color-classification head on the pooled language vector shapes
// it to be color-decodable and powers the live "decoded target" readout.
// Because the head is LINEAR and pooling is a mean, the decoded color's
// logit is an exact average of each token's own dot-product contribution —
// trainer.ts's tokenContributions() reads that off directly for the live
// per-token bars, no extra sub-model or forward pass required.

import type * as tfType from "@tensorflow/tfjs";
import { MAX_SEQ_LEN, VOCAB_SIZE, COLORS } from "./examples";

export type TF = typeof tfType;

export const IMG_SIZE = 32;

/** Adam learning rate. Bumped from the GRU-era 0.004 now that the language
    branch is a much smaller bag-of-embeddings — the network has fewer
    parameters and a simpler loss surface, so it tolerates (and benefits
    from) a faster step size for quicker convergence. */
export const LEARNING_RATE = 0.007;
/** Weight of the auxiliary color-classification loss vs. the action MSE. */
export const COLOR_LOSS_WEIGHT = 0.3;

export interface VLAModels {
  /** Main policy: [vision, tokens] → [action Δθ (2), color softmax]. */
  model: tfType.LayersModel;
}

export function buildVLAModel(tf: TF): VLAModels {
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

  // Branch B: language — bag-of-embeddings, no recurrence
  const langInput = tf.input({
    shape: [MAX_SEQ_LEN],
    name: "language_tokens",
    dtype: "int32",
  });
  const embedded = tf.layers
    .embedding({
      inputDim: VOCAB_SIZE,
      outputDim: 24,
      name: "text_embedding",
    })
    .apply(langInput); // [12, 24]
  const langPooled = tf.layers
    .globalAveragePooling1d({ name: "lang_vector" })
    .apply(embedded); // 24

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
  // tfjs-layers doesn't implement compile({lossWeights}) — scale the aux
  // color loss inside a custom per-output loss function instead (and the
  // loss array must then be all-functions, so MSE is a function too)
  const actionLoss = (yTrue: tfType.Tensor, yPred: tfType.Tensor) =>
    tf.metrics.MSE(yTrue, yPred);
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
