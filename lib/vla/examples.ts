// Language + scene-layout space for the VLA task.
//
// Four named colors (chosen for maximum hue contrast at small silhouette
// sizes), each with synonyms; sentences are generated from a slot grammar
// (filler? verb article color-word noun please?) so hundreds of surface
// forms collapse onto 4 intents. Every scene places exactly TWO blocks, one
// per side, with colors drawn from the four without replacement — so vision
// only ever has to tell two colors apart at once, not localize one among a
// crowd, which keeps training fast.
//
// Token id 0 is <pad>, id 1 is <unk>: out-of-vocabulary words (typos, free
// text) map to <unk>, and the trainer also randomly drops ~10% of non-color
// training tokens to <unk> so the encoder learns to key on the color word
// and shrug off words it doesn't know.

export const MAX_SEQ_LEN = 12;
export const PAD = 0;
export const UNK = 1;

export interface ColorDef {
  name: string;
  hex: string;
  synonyms: string[];
}

export const COLORS: ColorDef[] = [
  { name: "red", hex: "#e12d1a", synonyms: ["red", "crimson", "scarlet"] },
  { name: "black", hex: "#1c1c1c", synonyms: ["black"] },
  { name: "blue", hex: "#2a6fdb", synonyms: ["blue", "azure", "navy"] },
  { name: "yellow", hex: "#d9a800", synonyms: ["yellow", "golden"] },
];

const VERBS = [
  ["pick", "up"],
  ["grab"],
  ["grasp"],
  ["take"],
  ["lift"],
  ["fetch"],
  ["get"],
  ["reach", "for"],
];
const ARTICLES = ["the", "a"];
const NOUNS = ["block", "box", "cube", "object"];
const FILLERS = ["please", "robot", "now"];

// stable vocab: pad, unk, then every grammar word in insertion order
export const VOCAB: Record<string, number> = { "<pad>": PAD, "<unk>": UNK };
{
  let id = 2;
  const add = (w: string) => {
    if (!(w in VOCAB)) VOCAB[w] = id++;
  };
  VERBS.flat().forEach(add);
  ARTICLES.forEach(add);
  NOUNS.forEach(add);
  FILLERS.forEach(add);
  COLORS.forEach((c) => c.synonyms.forEach(add));
}
export const VOCAB_SIZE = Object.keys(VOCAB).length;

/** Token ids of every color synonym — exempt from training word-dropout. */
export const COLOR_TOKEN_IDS = new Set<number>(
  COLORS.flatMap((c) => c.synonyms.map((s) => VOCAB[s]))
);

/** Lowercase, strip punctuation, map OOV words to <unk>, pad to MAX_SEQ_LEN. */
export function tokenize(sentence: string): number[] {
  const words = sentence
    .toLowerCase()
    .replace(/[^a-z ]/g, "")
    .split(" ")
    .filter(Boolean);
  const tokens = new Array<number>(MAX_SEQ_LEN).fill(PAD);
  for (let i = 0; i < Math.min(words.length, MAX_SEQ_LEN); i++) {
    tokens[i] = VOCAB[words[i]] ?? UNK;
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
 * replacement, and some positional jitter per side.
 */
export function randomLayout(): Layout {
  const c1 = Math.floor(Math.random() * COLORS.length);
  let c2 = Math.floor(Math.random() * (COLORS.length - 1));
  if (c2 >= c1) c2++;
  return [
    { color: c1, x: 0.16 + (Math.random() - 0.5) * 0.12 },
    { color: c2, x: 0.84 + (Math.random() - 0.5) * 0.12 },
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
