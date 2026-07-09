// Generates the pretrained word-embedding assets for the VLA hero's language
// encoder (lib/vla/embeddings.ts loads them at "Start Training"):
//
//   public/vla/embeddings-50d.bin  int8-quantized GloVe vectors, one 50-dim
//                                  row per vocab word (~1MB, lazy-fetched)
//   public/vla/vocab.txt           newline-joined words; token id = line
//                                  index + 2 (ids 0/1 are <pad>/<unk>)
//   lib/vla/vocab.gen.ts           VOCAB_SIZE / EMBED_DIM / dequantization
//                                  scales + CORE_VOCAB (the grammar words'
//                                  ids, bundled so tokenize() works
//                                  synchronously before the full list loads)
//
// Source: GloVe 6B 50d (wiki+gigaword) via the gensim-data release mirror —
// word2vec text format, FREQUENCY-ORDERED, so "the top N_WORDS words" is just
// the first N accepted lines. The stream is gunzipped + read line-by-line and
// aborted once the vocab is full (~10MB transferred of the 66MB file, nothing
// cached on disk). Rerun only when the grammar (lib/vla/grammar.json), vocab
// size, or quantization changes; the artifacts are committed.
//
//   node scripts/gen-embeddings-data.mjs
//
// Quantization: per-dimension symmetric int8 (scale[d] = maxAbs[d]/127),
// dequantized at load time in the browser. GloVe values sit in roughly
// [-3, 3]; the ~0.01 rounding error is far below what this task can resolve.

import { createWriteStream } from "node:fs";
import { mkdir, writeFile } from "node:fs/promises";
import { createInterface } from "node:readline";
import { Readable } from "node:stream";
import { createGunzip } from "node:zlib";
import { fileURLToPath } from "node:url";
import path from "node:path";

import grammar from "../lib/vla/grammar.json" with { type: "json" };

const ROOT = path.join(path.dirname(fileURLToPath(import.meta.url)), "..");

/** Vocabulary size (real words; <pad>/<unk> are added on top at runtime).
    Bigger = more user-typed words resolve to a pretrained vector instead of
    <unk> at try-it time (training is unaffected either way — it only ever
    sees the grammar words). Costs ~50 int8 bytes/word in the lazy .bin. */
const N_WORDS = 20000;
const EMBED_DIM = 50;
const SOURCE_URL =
  "https://github.com/piskvorky/gensim-data/releases/download/glove-wiki-gigaword-50/glove-wiki-gigaword-50.gz";

// every word the slot grammar can emit MUST get a row (they're all common
// English, so they land in the top N anyway — this is a fail-loud guarantee)
const grammarWords = new Set(
  [
    ...Object.values(grammar.tasks).flatMap((t) => [
      ...t.verbs.flat(),
      ...(t.preps ?? []).flat(),
    ]),
    ...grammar.articles,
    ...grammar.nouns,
    ...grammar.fillers,
    ...grammar.colors.flatMap((c) => c.synonyms),
  ].map((w) => w.toLowerCase())
);

console.log(`fetching top ${N_WORDS} GloVe ${EMBED_DIM}d vectors…`);
const res = await fetch(SOURCE_URL);
if (!res.ok) throw new Error(`fetch failed: ${res.status} ${res.statusText}`);

const words = [];
const vectors = []; // Float64Array rows, acceptance order = token order
const seen = new Set();

const lines = createInterface({
  input: Readable.fromWeb(res.body).pipe(createGunzip()),
  crlfDelay: Infinity,
});
let header = true;
for await (const line of lines) {
  if (header) {
    header = false; // "400000 50"
    continue;
  }
  const sp = line.indexOf(" ");
  const word = line.slice(0, sp);
  // the tokenizer strips everything but [a-z ], so rows for punctuation,
  // numbers etc. would be unreachable — skip them
  if (!/^[a-z]+$/.test(word) || seen.has(word)) continue;
  // past the size cap, keep scanning ONLY for missing grammar words
  if (words.length >= N_WORDS && !grammarWords.has(word)) continue;
  seen.add(word);
  words.push(word);
  vectors.push(Float64Array.from(line.slice(sp + 1).split(" "), Number));
  if (
    words.length >= N_WORDS &&
    [...grammarWords].every((w) => seen.has(w))
  )
    break;
}
lines.close();
res.body.cancel?.().catch(() => {});

const missing = [...grammarWords].filter((w) => !seen.has(w));
if (missing.length) throw new Error(`grammar words not in GloVe: ${missing}`);
console.log(`accepted ${words.length} words`);

// per-dimension symmetric int8 quantization
const scales = new Array(EMBED_DIM).fill(0);
for (const v of vectors)
  for (let d = 0; d < EMBED_DIM; d++)
    scales[d] = Math.max(scales[d], Math.abs(v[d]));
for (let d = 0; d < EMBED_DIM; d++) scales[d] /= 127;

const bin = new Int8Array(words.length * EMBED_DIM);
for (let i = 0; i < vectors.length; i++)
  for (let d = 0; d < EMBED_DIM; d++)
    bin[i * EMBED_DIM + d] = Math.round(vectors[i][d] / scales[d]);

await mkdir(path.join(ROOT, "public/vla"), { recursive: true });
await writeFile(path.join(ROOT, "public/vla/embeddings-50d.bin"), bin);
await writeFile(path.join(ROOT, "public/vla/vocab.txt"), words.join("\n"));

// CORE_VOCAB: the grammar words' final token ids, bundled into the page so
// tokenize() works synchronously (SSR default sentence, demo cycling) before
// the full 20k-word list has been lazy-fetched
const core = Object.fromEntries(
  words
    .map((w, i) => [w, i + 2])
    .filter(([w]) => grammarWords.has(w))
    .sort((a, b) => a[1] - b[1])
);
const gen = `// GENERATED by scripts/gen-embeddings-data.mjs — do not edit.
// Metadata for the pretrained GloVe embedding assets in public/vla/ (see
// lib/vla/embeddings.ts). Token ids: 0=<pad>, 1=<unk>, then vocab.txt line
// index + 2; the .bin holds one int8 row per word, dequantized per dimension
// with EMBED_SCALES.

export const EMBED_DIM = ${EMBED_DIM};
/** <pad> + <unk> + ${words.length} GloVe words. */
export const VOCAB_SIZE = ${words.length + 2};

/** Per-dimension int8 dequantization scales (value = int8 * scale[d]). */
export const EMBED_SCALES: number[] = [
${scales.map((s) => `  ${s.toPrecision(7)},`).join("\n")}
];

/** Grammar words' token ids — the sync-available slice of the full vocab. */
export const CORE_VOCAB: Record<string, number> = ${JSON.stringify(core, null, 2)};
`;
await writeFile(path.join(ROOT, "lib/vla/vocab.gen.ts"), gen);

console.log(
  `wrote public/vla/embeddings-50d.bin (${(bin.length / 1024).toFixed(0)}KB), ` +
    `public/vla/vocab.txt, lib/vla/vocab.gen.ts (${Object.keys(core).length} core words)`
);
