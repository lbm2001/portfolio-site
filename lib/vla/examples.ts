// The training-example space: every phrasing template x target color, with
// tokens precomputed. The trainer samples uniformly from these for every
// batch element; the hero displays one example at a time (prompt textarea,
// language chips, demonstration trajectory) and swaps it every demo cycle —
// thousands of the others train invisibly in between.

import { tokenize } from "./model";

export type BlockColor = "red" | "black";

export interface Example {
  color: BlockColor;
  text: string;
  words: string[];
  tokens: number[];
}

// first template x red is the SSR-default example shown before hydration
const TEMPLATES = [
  "pick up the {c} block",
  "grasp the {c} block",
  "reach for the {c} block",
  "let the robot grasp the {c} block",
];

export const EXAMPLES: Example[] = (["red", "black"] as const).flatMap((color) =>
  TEMPLATES.map((tpl) => {
    const text = tpl.replace("{c}", color);
    return { color, text, words: text.split(" "), tokens: tokenize(text) };
  })
);

export function randomExample(): Example {
  return EXAMPLES[Math.floor(Math.random() * EXAMPLES.length)];
}

// word classes for the live attention read-out: as the loss converges the
// verbs decay and the object words ("red"/"black", "block") ramp up
const VERBS = new Set(["pick", "grasp", "reach"]);
const COLORS = new Set(["red", "black"]);

export function attentionWeight(word: string, progress: number): number {
  if (COLORS.has(word)) return 0.08 + 0.8 * progress;
  if (word === "block") return 0.08 + 0.88 * progress;
  if (VERBS.has(word)) return 0.22 - 0.14 * progress;
  if (word === "up" || word === "for") return 0.1 - 0.05 * progress;
  return 0.07 - 0.03 * progress; // the / let / robot
}
