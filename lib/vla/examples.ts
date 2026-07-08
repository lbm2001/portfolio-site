// Language + scene-layout space for the VLA task.
//
// Eight named colors (chosen for maximum hue contrast at small silhouette
// sizes), each with synonyms; sentences are generated from a slot grammar
// (filler? verb article color-word noun please?) so hundreds of surface
// forms collapse onto 8 intents. The word inventory lives in grammar.json
// (single source of truth shared with scripts/gen-embeddings-data.mjs).
// Every scene places exactly TWO blocks, one per side, with colors drawn
// from the eight without replacement — so vision only ever has to tell two
// colors apart at once, not localize one among a crowd, which keeps
// training fast; the wider palette makes the language-grounding (which of
// eight words → which of two blocks) the harder part.
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
import { CONFIG } from "./config";
import { BLOCK, BLOCK_MAX, BLOCK_MIN } from "./geometry";
import { CORE_VOCAB } from "./vocab.gen";

export { VOCAB_SIZE } from "./vocab.gen";

export const MAX_SEQ_LEN = CONFIG.task.maxSeqLen;
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
  if (Math.random() < CONFIG.task.fillerProb) parts.push(pick(FILLERS));
  parts.push(...pick(VERBS));
  parts.push(pick(ARTICLES));
  parts.push(pick(COLORS[color].synonyms));
  parts.push(pick(NOUNS));
  if (Math.random() < CONFIG.task.pleaseProb) parts.push("please");
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
  size: number; // block side length, workspace units (grasp height = size/2)
}
export type Layout = BlockPos[];

/**
 * TWO blocks per scene — one on each side of the arm — with each block placed
 * at a RANDOM position across its full cleanly-reachable side of the floor
 * (was a ±0.02 pinprick around fixed 0.16/0.84). The colors are drawn from the
 * palette without replacement, so vision still only has to tell two colors
 * apart.
 *
 * Reachable-floor analysis (arm base (0.5,0.2), links 0.32+0.26, grasp at
 * y=0.06): the annulus outer radius 0.58 covers the whole floor out to
 * |x-0.5| ≈ 0.56, and the inner radius 0.06 never bites (the base sits 0.14
 * above the floor). BUT blocks within |x-0.5| < 0.167 of the base need an
 * elbow bend past the sampled THETA2_RANGE — a near-singular fold that also
 * renders the forearm through the upper arm — so the genuinely grasp-able
 * floor is the two side BANDS below, not one span through the centre. Each
 * band's edges stay comfortably inside the joint ranges (verified: |θ2| ≤ 2.39
 * at the inner edges) and fully in-frame.
 *
 * Wide placement is only usable because the model input is now 64px (was 32,
 * see IMG_SIZE): a block renders ~8px and a band spans ~12px, so the ~0.9-rad
 * elbow swing across a band is resolvable to the target precision. At 32px the
 * position was unresolvable (~3px block), the policy learned the per-band MEAN
 * and landed off-centre — which is exactly why placement used to be pinned to
 * ±0.02. Raising the input resolution (the documented fix) is what lets the
 * scatter widen back out.
 *
 * Each block also randomizes its SIDE LENGTH in [BLOCK_MIN, BLOCK_MAX]. Size
 * is not just cosmetic: the grasp target is the block CENTRE (y = size/2), so a
 * bigger block is grasped higher and the policy has to read the size out of the
 * 64px image to get the reach height right. It also shifts the near-singular
 * dead zone — the LARGEST block (grasped at y=0.08) needs |x−0.5| ≳ 0.181 to
 * keep the elbow inside THETA2_RANGE — so the band inner edges (0.31/0.69) are
 * set for that worst case; smaller blocks clear it with room to spare.
 */
const PLACE_L: [number, number] = CONFIG.task.placeLeft; // left band [lo, hi]
const PLACE_R: [number, number] = CONFIG.task.placeRight; // right band [lo, hi]
const inBand = ([lo, hi]: [number, number]) => lo + Math.random() * (hi - lo);
const randSize = () => BLOCK_MIN + Math.random() * (BLOCK_MAX - BLOCK_MIN);
export function randomLayout(): Layout {
  const c1 = Math.floor(Math.random() * COLORS.length);
  let c2 = Math.floor(Math.random() * (COLORS.length - 1));
  if (c2 >= c1) c2++;
  // c1/c2 are already a uniform random pair, so which color lands on which
  // side is random too — c1 left, c2 right.
  return [
    { color: c1, x: inBand(PLACE_L), size: randSize() },
    { color: c2, x: inBand(PLACE_R), size: randSize() },
  ];
}

/** Deterministic default layout (SSR-safe): black left, red right, with two
    fixed sizes so the pre-training scene already shows the size variety. */
export const DEFAULT_LAYOUT: Layout = [
  { color: 1, x: 0.16, size: 0.09 },
  { color: 0, x: 0.84, size: BLOCK },
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
