// VLA network definition + tokenizer. TensorFlow.js is passed in (and only
// ever loaded via dynamic import in trainer.ts) so this module stays
// SSR-safe and the ~1MB tfjs bundle is fetched lazily on "Start Training".

import type * as tfType from "@tensorflow/tfjs";

export type TF = typeof tfType;

export const VOCAB: Record<string, number> = {
  "<pad>": 0,
  pick: 1,
  up: 2,
  the: 3,
  red: 4,
  black: 5,
  block: 6,
  reach: 7,
  grasp: 8,
  let: 9,
  robot: 10,
  for: 11,
};
export const MAX_SEQ_LEN = 8;

/** Pure tokenizer: lowercase, strip punctuation, pad/truncate to MAX_SEQ_LEN. */
export function tokenize(sentence: string): number[] {
  const words = sentence
    .toLowerCase()
    .replace(/[^a-z ]/g, "")
    .split(" ")
    .filter(Boolean);
  const tokens = new Array<number>(MAX_SEQ_LEN).fill(0);
  for (let i = 0; i < Math.min(words.length, MAX_SEQ_LEN); i++) {
    tokens[i] = VOCAB[words[i]] || 0;
  }
  return tokens;
}

/**
 * Multi-input VLA graph: 16x16 RGB pixels are chunked into 4x4 spatial
 * patches (conv stride = kernel, the ViT-style patch embedding), the token
 * sequence goes through a learned embedding; both flatten into a fused
 * bottleneck that regresses two joint-velocity outputs. Adam at a high LR +
 * MSE so behavioral cloning visibly converges within ~10s in the browser.
 */
export function buildVLAModel(tf: TF): tfType.LayersModel {
  // Branch A: vision (explicit 4x4 spatial patching). 16 filters, not the
  // minimal 4: the policy must regress BOTH joint angles across mirrored
  // elbow configurations from 16x16 pixels — with 4 filters it gives up on
  // vision entirely and plateaus at the language-only loss (~1.5), which
  // makes the closed-loop rollout overshoot into the joint limits.
  const visionInput = tf.input({ shape: [16, 16, 3], name: "vision_pixels" });
  const patchConv = tf.layers
    .conv2d({
      filters: 16,
      kernelSize: 4,
      strides: 4,
      activation: "relu",
      name: "patch_embeddings",
    })
    .apply(visionInput);
  const flattenVision = tf.layers.flatten().apply(patchConv);

  // Branch B: language (sequence embedding)
  const langInput = tf.input({
    shape: [MAX_SEQ_LEN],
    name: "language_tokens",
    dtype: "int32",
  });
  const textEmbedding = tf.layers
    .embedding({
      inputDim: Object.keys(VOCAB).length,
      outputDim: 8,
      name: "text_embedding_layer",
    })
    .apply(langInput);
  const flattenLang = tf.layers.flatten().apply(textEmbedding);

  // Fusion bottleneck + regression action head
  const fusedFeatures = tf.layers
    .concatenate()
    .apply([flattenVision, flattenLang] as tfType.SymbolicTensor[]);
  const dense1 = tf.layers
    .dense({ units: 64, activation: "relu" })
    .apply(fusedFeatures);
  const actionOutput = tf.layers
    .dense({ units: 2, activation: "linear", name: "joint_velocities" })
    .apply(dense1) as tfType.SymbolicTensor;

  const model = tf.model({
    inputs: [visionInput, langInput],
    outputs: actionOutput,
  });

  model.compile({
    optimizer: tf.train.adam(0.008),
    loss: "meanSquaredError",
  });

  return model;
}
