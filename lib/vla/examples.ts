// Language + scene-layout space for the VLA task.
//
// Four named colors (chosen for maximum hue contrast at small silhouette
// sizes), each with synonyms; sentences are generated from a slot grammar
// (filler? verb article color-word noun please?) so hundreds of surface
// forms collapse onto 4 intents. The word inventory lives in grammar.json
// (single source of truth shared with scripts/gen-embeddings-data.mjs).
// Every scene places exactly TWO blocks, one per side, with colors drawn
// from the four without replacement — so vision only ever has to tell two
// colors apart at once, not localize one among a crowd, which keeps
// training fast.
//
// Token ids index the pretrained GloVe table (vocab.gen.ts / public/vla/):
// 0 is <pad>, 1 is <unk>, then the ~20k-word GloVe vocab. Only the grammar
// words' ids (CORE_VOCAB) are bundled — tokenize() uses them synchronously
// (SSR default sentence, demo cycling) and upgrades to the full lazy-fetched
// list once lib/vla/embeddings.ts registers it, so free user text like
// "gold" resolves to a real pretrained vector, not <unk>. The trainer still
// randomly drops ~10% of non-color training tokens to <unk> so the encoder
// learns to shrug off genuinely unknown words.

import grammar from "./grammar.json";
import { CORE_VOCAB } from "./vocab.gen";

export { VOCAB_SIZE } from "./vocab.gen";

export const MAX_SEQ_LEN = 12;
export const PAD = 0;
export const UNK = 1;

export interface ColorDef {
  name: string;
  hex: string;
  synonyms: string[];
}

export const COLORS: ColorDef[] = grammar.colors;

const VERBS = grammar.verbs;
const ARTICLES = grammar.articles;
const NOUNS = grammar.nouns;
const FILLERS = grammar.fillers;

/** Token ids of every color synonym — exempt from training word-dropout. */
export const COLOR_TOKEN_IDS = new Set<number>(
  COLORS.flatMap((c) => c.synonyms.map((s) => CORE_VOCAB[s]))
);

// full GloVe word list, registered by loadEmbeddings() once fetched
let fullVocab: Map<string, number> | null = null;

/** Install the complete vocab (word at index i → token id i+2). */
export function registerFullVocab(words: string[]) {
  fullVocab = new Map(words.map((w, i) => [w, i + 2]));
}

/** Lowercase, strip punctuation, map OOV words to <unk>, pad to MAX_SEQ_LEN. */
export function tokenize(sentence: string): number[] {
  const words = sentence
    .toLowerCase()
    .replace(/[^a-z ]/g, "")
    .split(" ")
    .filter(Boolean);
  const tokens = new Array<number>(MAX_SEQ_LEN).fill(PAD);
  for (let i = 0; i < Math.min(words.length, MAX_SEQ_LEN); i++) {
    tokens[i] = fullVocab?.get(words[i]) ?? CORE_VOCAB[words[i]] ?? UNK;
  }
  return tokens;
}

export interface Sentence {
  color: number; // index into COLORS
  text: string;
  words: string[];
  tokens: number[];
}

const pick = <T,>(a: readonly T[]): T => a[Math.floor(Math.random() * a.length)];

export function sampleSentence(color: number): Sentence {
  const parts: string[] = [];
  if (Math.random() < 0.25) parts.push(pick(FILLERS));
  parts.push(...pick(VERBS));
  parts.push(pick(ARTICLES));
  parts.push(pick(COLORS[color].synonyms));
  parts.push(pick(NOUNS));
  if (Math.random() < 0.2) parts.push("please");
  const text = parts.join(" ");
  return { color, text, words: parts, tokens: tokenize(text) };
}

/** Deterministic default (safe for SSR/hydration — no randomness). */
export const DEFAULT_SENTENCE: Sentence = {
  color: 0,
  text: "pick up the red block",
  words: ["pick", "up", "the", "red", "block"],
  tokens: tokenize("pick up the red block"),
};

// ---- scene layouts ----

export interface BlockPos {
  x: number; // block center, workspace units
  color: number; // index into COLORS
}
export type Layout = BlockPos[];

/**
 * TWO blocks per scene — one on each side of the arm (like the original
 * black/red setup) — with the two colors drawn from all 8 without
 * replacement, and a small positional jitter per side.
 *
 * The jitter band is ±0.02 (was ±0.06). At the 32x32 model input a block is
 * only ~2.7px wide, so a ±0.06 band (~3px of travel) shifts the block by less
 * than its own width — unresolvable after the downsample — yet the correct
 * elbow angle swings up to ~0.5 rad across it. The policy therefore couldn't
 * see where in the band the block sat, learned the per-side MEAN target, and
 * landed off-center (the "reach is off / noisy" symptom + most of the loss
 * floor). ±0.02 keeps scenes visibly non-identical while shrinking the
 * unresolvable target spread ~9x, so the mean the policy can learn lands on
 * the block. (If wide scatter is ever wanted back, raise the input resolution
 * instead so the position becomes resolvable — see the 32->48 option.)
 */
const LAYOUT_JITTER = 0.04; // full width; ±0.02 per side
export function randomLayout(): Layout {
  const c1 = Math.floor(Math.random() * COLORS.length);
  let c2 = Math.floor(Math.random() * (COLORS.length - 1));
  if (c2 >= c1) c2++;
  return [
    { color: c1, x: 0.16 + (Math.random() - 0.5) * LAYOUT_JITTER },
    { color: c2, x: 0.84 + (Math.random() - 0.5) * LAYOUT_JITTER },
  ];
}

/** Deterministic default layout (SSR-safe): black left, red right. */
export const DEFAULT_LAYOUT: Layout = [
  { color: 1, x: 0.16 },
  { color: 0, x: 0.84 },
];

export function blockOfColor(layout: Layout, color: number): BlockPos {
  return layout.find((b) => b.color === color) ?? layout[0];
}

export function hasColor(layout: Layout, color: number): boolean {
  return layout.some((b) => b.color === color);
}

/** A random color that IS present in the given layout. */
export function presentColor(layout: Layout): number {
  return layout[Math.floor(Math.random() * layout.length)].color;
}
