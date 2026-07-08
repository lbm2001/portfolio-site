// Language + scene-layout space for the VLA tasks.
//
// Eight named colors (chosen for maximum hue contrast at small silhouette
// sizes), each with synonyms; sentences are generated from a slot grammar
// with per-task verb sets (filler? verb article color-word noun please? —
// stack adds "prep article ref-color noun" for the reference block) so
// hundreds of surface forms collapse onto (task, color[, refColor]) intents.
// The word inventory lives in grammar.json (single source of truth shared
// with scripts/gen-embeddings-data.mjs). Which tasks/colors/densities are
// actually SAMPLED is the user's ⚙ run config (lib/vla/run-config.ts):
// scenes place 2..min(maxBlocks, numColors) blocks (≤2 per side band),
// colors drawn from the first numColors palette entries without replacement
// — every block is a unique color and the job is to localize the named
// one(s) among up to four.
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
import { runConfig, type TaskKind } from "./run-config";
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

const TASK_VERBS: Record<TaskKind, string[][]> = {
  lift: grammar.tasks.lift.verbs,
  stack: grammar.tasks.stack.verbs,
};
const STACK_PREPS = grammar.tasks.stack.preps;
const ARTICLES = grammar.articles;
const NOUNS = grammar.nouns;
const FILLERS = grammar.fillers;

/** Token ids of every color synonym — exempt from training word-dropout. */
export const COLOR_TOKEN_IDS = new Set<number>(
  COLORS.flatMap((c) => c.synonyms.map((s) => CORE_VOCAB[s]))
);

/** Token ids that CARRY LABEL INFORMATION — the color synonyms plus every
    task verb/preposition word. Exempt from training word-dropout: dropping
    the verb to <unk> while the task label still says "stack" would be
    label noise, exactly like dropping the color word (the original reason
    this exemption exists). Articles/nouns/fillers remain droppable, which is
    all the robustness the dropout was ever buying. */
export const LABEL_TOKEN_IDS = new Set<number>([
  ...COLOR_TOKEN_IDS,
  ...Object.values(TASK_VERBS)
    .flat(2)
    .map((w) => CORE_VOCAB[w]),
  ...STACK_PREPS.flat().map((w) => CORE_VOCAB[w]),
]);

// full GloVe word list, registered by loadEmbeddings() once fetched
let fullVocab: Map<string, number> | null = null;

/** Install the complete vocab (word at index i → token id i+2). */
export function registerFullVocab(words: string[]) {
  fullVocab = new Map(words.map((w, i) => [w, i + 2]));
}

// ---- typo tolerance for user-typed words ----
//
// A misspelling ("puple") is in neither the full GloVe list nor CORE_VOCAB, so
// it tokenizes to <unk> — a ZERO row in the embedding table (embeddings.ts), so
// the language track goes silent, not wrong. To recover the intent we fuzzy-map
// an OOV word to the nearest CORE_VOCAB entry (the color synonyms + grammar
// words). We deliberately DON'T fuzz over the full 20k GloVe list: the tight
// dictionary is both cheaper and safer ("puple" -> "purple", never "pupil").
//
// Distance is Optimal String Alignment (Levenshtein + adjacent transposition at
// cost 1), since transpositions ("purpel", "gerner") are the commonest typo.
// This only ever runs on the `?? UNK` branch below — i.e. never on the training
// hot path, whose sentences are built from in-vocab grammar words.

const MAX_EDITS = 2;

/** OSA edit distance, early-exiting once every cell in a row exceeds `max`. */
function editDist(a: string, b: string, max: number): number {
  if (Math.abs(a.length - b.length) > max) return max + 1;
  let prevPrev: number[] = [];
  let prev = Array.from({ length: b.length + 1 }, (_, j) => j);
  for (let i = 1; i <= a.length; i++) {
    const cur = [i];
    let rowMin = i;
    for (let j = 1; j <= b.length; j++) {
      const cost = a[i - 1] === b[j - 1] ? 0 : 1;
      let v = Math.min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost);
      if (i > 1 && j > 1 && a[i - 1] === b[j - 2] && a[i - 2] === b[j - 1])
        v = Math.min(v, prevPrev[j - 2] + 1); // adjacent transposition
      cur[j] = v;
      if (v < rowMin) rowMin = v;
    }
    if (rowMin > max) return max + 1; // no cell can still beat the budget
    prevPrev = prev;
    prev = cur;
  }
  return prev[b.length];
}

/** Token id of the nearest CORE_VOCAB word within the edit budget, or undefined.
    Budget scales down for short words so 2-3 letter garbage can't match a color
    (color words are >=3 chars, e.g. "reb" -> "red" still resolves at budget 1). */
function correctWord(word: string): number | undefined {
  const budget = Math.min(MAX_EDITS, word.length <= 3 ? 1 : MAX_EDITS);
  let best: string | undefined;
  let bestD = budget + 1;
  for (const cand in CORE_VOCAB) {
    const d = editDist(word, cand, bestD - 1); // only care if it strictly beats best
    if (d < bestD) {
      bestD = d;
      best = cand;
    }
  }
  return best === undefined ? undefined : CORE_VOCAB[best];
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
    const w = words[i];
    tokens[i] = fullVocab?.get(w) ?? CORE_VOCAB[w] ?? correctWord(w) ?? UNK;
  }
  return tokens;
}

export interface Sentence {
  task: TaskKind;
  color: number; // index into COLORS — the block the arm acts ON
  /** stack only: the block the commanded one goes ON TOP OF (else null). */
  refColor: number | null;
  text: string;
  words: string[];
  tokens: number[];
}

const pick = <T,>(a: readonly T[]): T => a[Math.floor(Math.random() * a.length)];

/** A random surface form for (task, color[, refColor]): filler? verb? article?
    color noun? [prep article? refColor noun?] please?. Verb/article/noun are
    randomly DROPPED (see the dropXxxProb comments in config.ts) so the
    position-aware language slots also train on compressed forms ("black on
    red", "grab red") — without this they overfit the full grammar's fixed
    color-word positions and terse user commands decode wrongly. Preps and
    color words never drop: they carry the task/color labels. */
export function sampleSentence(
  task: TaskKind,
  color: number,
  refColor: number | null = null
): Sentence {
  const keep = (dropProb: number) => Math.random() >= dropProb;
  const parts: string[] = [];
  if (Math.random() < CONFIG.task.fillerProb) parts.push(pick(FILLERS));
  if (keep(CONFIG.task.dropVerbProb)) parts.push(...pick(TASK_VERBS[task]));
  if (keep(CONFIG.task.dropArticleProb)) parts.push(pick(ARTICLES));
  parts.push(pick(COLORS[color].synonyms));
  if (keep(CONFIG.task.dropNounProb)) parts.push(pick(NOUNS));
  if (task === "stack" && refColor !== null) {
    parts.push(...pick(STACK_PREPS));
    if (keep(CONFIG.task.dropArticleProb)) parts.push(pick(ARTICLES));
    parts.push(pick(COLORS[refColor].synonyms));
    if (keep(CONFIG.task.dropNounProb)) parts.push(pick(NOUNS));
  }
  if (Math.random() < CONFIG.task.pleaseProb) parts.push("please");
  const text = parts.join(" ");
  return {
    task,
    color,
    refColor: task === "stack" ? refColor : null,
    text,
    words: parts,
    tokens: tokenize(text),
  };
}

/** A random command EXECUTABLE in the given layout, for a task drawn from
    the active run config: the acted-on color is present, and stack names a
    second, distinct present color to stack onto (layouts always hold ≥2
    blocks, so a pair always exists). `forceTask` pins the task instead of
    drawing it — the trainer's per-bucket probe batches use it. */
export function sampleCommand(layout: Layout, forceTask?: TaskKind): Sentence {
  const tasks = runConfig().tasks;
  const task = forceTask ?? tasks[Math.floor(Math.random() * tasks.length)];
  if (task === "stack") {
    const [color, ref] = presentColorPair(layout);
    return sampleSentence(task, color, ref);
  }
  return sampleSentence(task, presentColor(layout));
}

/** Deterministic default (safe for SSR/hydration — no randomness). */
export const DEFAULT_SENTENCE: Sentence = {
  task: "lift",
  color: 0,
  refColor: null,
  text: "pick up the red block",
  words: ["pick", "up", "the", "red", "block"],
  tokens: tokenize("pick up the red block"),
};

// ---- scene layouts ----

export interface BlockPos {
  x: number; // block center, workspace units
  color: number; // index into COLORS
  size: number; // block side length, workspace units (grasp height = size/2)
  /** Rest height of the block's BOTTOM (0 / absent = on the floor; >0 = on
      top of another block, set when a stack episode releases it there).
      Training scenes always start flat — stacked rests only arise from
      rollout/demo releases. */
  y?: number;
}
export type Layout = BlockPos[];

/**
 * 2..min(maxBlocks, numColors) blocks per scene (both from the active ⚙ run
 * config), at most 2 per side band, each placed at a RANDOM position across
 * its cleanly-reachable side of the floor. Colors are drawn from the first
 * numColors palette entries without replacement, so every block is a distinct
 * color and the named target is unambiguous — but vision has to pick it out
 * among up to four.
 *
 * Two blocks sharing one ~0.20-wide band would occlude each other (a block is
 * up to 0.16 wide, boosted ×silBlockScale in the model's-eye view). placeSide
 * reject-samples same-side positions/sizes until their silhouettes clear by
 * minBlockGap; if no attempt fits within the attempt budget (both drawn large),
 * it falls back to a SINGLE block rather than force an occluding overlap — so
 * two-per-side naturally biases toward smaller blocks, and huge pairs simply
 * don't happen. The unfilled color slot is just left unused (colors stay
 * unique among the blocks actually placed).
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
const MIN_GAP = CONFIG.task.minBlockGap;

const inBand = ([lo, hi]: [number, number]) => lo + Math.random() * (hi - lo);
const randSize = () => BLOCK_MIN + Math.random() * (BLOCK_MAX - BLOCK_MIN);

/** Half-width a block occupies in the MODEL'S-EYE view (silBlockScale boost) —
    the wider of the two views, so clearing this clears the display too. */
const silHalf = (size: number) => (size * CONFIG.render.silBlockScale) / 2;

/** k distinct color indices from the palette's first `poolSize` entries,
    uniformly without replacement (partial shuffle). */
function pickColors(k: number, poolSize: number): number[] {
  const pool = Array.from({ length: poolSize }, (_, i) => i);
  for (let i = 0; i < k; i++) {
    const j = i + Math.floor(Math.random() * (pool.length - i));
    [pool[i], pool[j]] = [pool[j], pool[i]];
  }
  return pool.slice(0, k);
}

/** Lay `colors` (1 or 2) into one band. For two we CONSTRUCT a non-overlapping
    placement rather than reject-sample: draw both sizes, require their center
    separation to clear both silhouettes plus MIN_GAP, then place the left block
    anywhere its clearance still fits before the band end and the right block
    anywhere past that clearance — so both keep full positional freedom and the
    pair always fits. Only a pair whose required clearance exceeds the whole
    band (both drawn large — the band is just ~0.20 wide) can't share it; that
    pair drops to a single block (see randomLayout's doc comment). */
function placeSide(band: [number, number], colors: number[]): BlockPos[] {
  const [lo, hi] = band;
  if (colors.length === 1)
    return [{ color: colors[0], x: inBand(band), size: randSize() }];
  const sizeL = randSize();
  const sizeR = randSize();
  const sep = silHalf(sizeL) + silHalf(sizeR) + MIN_GAP;
  if (sep > hi - lo)
    return [{ color: colors[0], x: inBand(band), size: sizeL }];
  const xL = lo + Math.random() * (hi - lo - sep); // left block, clearance still fits
  const xR = xL + sep + Math.random() * (hi - (xL + sep)); // right block, past the clearance
  return [
    { color: colors[0], x: xL, size: sizeL },
    { color: colors[1], x: xR, size: sizeR },
  ];
}

export function randomLayout(): Layout {
  const { numColors, maxBlocks } = runConfig();
  // colors are unique per scene, so the palette also caps the block count
  const cap = Math.min(maxBlocks, numColors);
  const n = 2 + Math.floor(Math.random() * (cap - 1)); // uniform 2..cap
  // split across the two bands, ≤2 per band: 2 → 1+1, 4 → 2+2, 3 → coin flip
  const nL = n === 2 ? 1 : n === 4 ? 2 : Math.random() < 0.5 ? 1 : 2;
  const colors = pickColors(n, numColors); // unique across the whole scene
  return [
    ...placeSide(PLACE_L, colors.slice(0, nL)),
    ...placeSide(PLACE_R, colors.slice(nL)),
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

/** A random color that IS present in the given layout. */
export function presentColor(layout: Layout): number {
  return layout[Math.floor(Math.random() * layout.length)].color;
}

/** Two DISTINCT present colors — stack's (acted-on, stacked-onto) pair.
    Layouts always hold ≥2 blocks, so a pair always exists. */
export function presentColorPair(layout: Layout): [number, number] {
  const i = Math.floor(Math.random() * layout.length);
  let j = Math.floor(Math.random() * (layout.length - 1));
  if (j >= i) j++;
  return [layout[i].color, layout[j].color];
}
