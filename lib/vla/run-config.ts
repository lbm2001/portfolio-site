// The USER-facing run configuration of the VLA hero — the knobs the ⚙ menu in
// the training bar exposes before training starts, as opposed to CONFIG
// (lib/vla/config.ts), which is the developer knob sheet. A RunConfig picks
// WHICH task family the demo trains (palette size, scene density, task set);
// CONFIG tunes HOW it trains.
//
// The active RunConfig is plain module state, and — like examples.ts's
// registerFullVocab — it must be installed on BOTH threads: the main thread
// (Hero's demo-cycle layout/sentence sampling) and the trainer worker (batch
// synthesis) each hold their own copy of this module. Hero calls setRunConfig
// before trainer.start(); the proxy ships the config inside the {t:"start"}
// message and the worker installs it before building the model.
//
// It deliberately does NOT change any model shape: the color head stays
// 8-wide and the task head 3-wide regardless of the selection — numColors and
// tasks only restrict what the samplers draw, so every RunConfig trains the
// same architecture and the calibrated CONFIG numbers stay comparable.

import { CONFIG } from "./config";

/** Task order is the task head's class order — never reorder. */
export const TASKS = ["lift", "stack"] as const;
export type TaskKind = (typeof TASKS)[number];

export interface RunConfig {
  /** Palette size — scenes draw colors from the FIRST N entries of COLORS. */
  numColors: 2 | 4 | 8;
  /** Scene density cap — a scene holds 2..min(maxBlocks, numColors) blocks
      (colors are unique per scene, so the palette also caps the count). */
  maxBlocks: 2 | 3 | 4;
  /** Enabled task set (≥1). Training samples/demo cycles draw uniformly. */
  tasks: TaskKind[];
}

/** Today's landing-page behavior — what trains when the menu is untouched. */
export const DEFAULT_RUN_CONFIG: RunConfig = {
  numColors: 8,
  maxBlocks: 4,
  tasks: ["lift"],
};

let current: RunConfig = DEFAULT_RUN_CONFIG;

/** Install the active run config on THIS thread (defensive copy). */
export function setRunConfig(rc: RunConfig) {
  current = { ...rc, tasks: [...rc.tasks] };
}

export function runConfig(): RunConfig {
  return current;
}

/** DUMMY training-time estimate for the ⚙ menu, from the placeholder factor
    table in CONFIG.eta — to be replaced with gauged numbers per config. */
export function estimateTrainingSeconds(rc: RunConfig): number {
  const e = CONFIG.eta;
  const taskCost = rc.tasks.reduce((s, t) => s + e.taskCost[t], 0);
  return Math.round(
    e.baseSeconds * taskCost * e.colorFactor[rc.numColors] * e.blockFactor[rc.maxBlocks]
  );
}
