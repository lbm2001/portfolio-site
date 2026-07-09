// The USER-facing run configuration of the VLA hero — the knobs the ⚙ menu in
// the training bar exposes before training starts, as opposed to CONFIG
// (lib/vla/config.ts), which is the developer knob sheet. A RunConfig picks
// the scene difficulty the demo trains on (palette size, scene density);
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
// 8-wide regardless of the selection — numColors only restricts what the
// samplers draw, so every RunConfig trains the same architecture and the
// calibrated CONFIG numbers stay comparable.

import { CONFIG } from "./config";

export interface RunConfig {
  /** Palette size — scenes draw colors from the FIRST N entries of COLORS. */
  numColors: 2 | 4 | 8;
  /** Scene density cap — a scene holds 2..min(maxBlocks, numColors) blocks
      (colors are unique per scene, so the palette also caps the count). */
  maxBlocks: 2 | 3 | 4;
}

/** Today's landing-page behavior — what trains when the menu is untouched. */
export const DEFAULT_RUN_CONFIG: RunConfig = {
  numColors: 8,
  maxBlocks: 4,
};

let current: RunConfig = DEFAULT_RUN_CONFIG;

/** Install the active run config on THIS thread (defensive copy). */
export function setRunConfig(rc: RunConfig) {
  current = { ...rc };
}

export function runConfig(): RunConfig {
  return current;
}

/** Training-time estimate for the ⚙ menu, from the factor table in
    CONFIG.eta (gauged 2026-07 against the carry-flag pick-up arch — see
    the eta comment there for the measurement basis). */
export function estimateTrainingSeconds(rc: RunConfig): number {
  const e = CONFIG.eta;
  return Math.round(
    e.baseSeconds * e.colorFactor[rc.numColors] * e.blockFactor[rc.maxBlocks]
  );
}
