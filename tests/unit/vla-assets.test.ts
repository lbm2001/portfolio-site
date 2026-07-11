import { existsSync, readFileSync, statSync } from "node:fs";
import { createRequire } from "node:module";
import { join } from "node:path";
import { describe, expect, it } from "vitest";

import {
  VLA_ASSET_BASE,
  VLA_REPLAY_MANIFEST,
  VLA_RUNTIME_ASSETS,
  VLA_VERSION,
} from "../../lib/vla-assets";

// The seam that neither repo can guard alone.
//
// mini-vla's own suite proves the package fetches from whatever `assetBase` it
// is handed. It cannot prove THIS app hands it the right one. If the copy
// script writes public/vla/0.4.0/ while Hero asks for /vla/0.4.0-beta/, every
// test in mini-vla passes and the site 404s on the visitor's first click —
// silently, because the hero only fetches embeddings lazily on "Start
// Training", so the page looks perfectly healthy until someone presses it.
//
// This runs after the asset-copy step in the build, so a mismatch fails the
// build instead. No browser, no GPU, milliseconds.

const require = createRequire(import.meta.url);
const pkgVersion: string = require("mini-vla/package.json").version;

const root = join(__dirname, "../..");
const assetDir = join(root, "public", VLA_ASSET_BASE);

describe("mini-vla asset path", () => {
  it("derives the version from the package, not a hard-coded string", () => {
    // mini-vla's release script refuses to tag when package.json's version
    // disagrees with the tag, so this field is the source of truth. Anyone who
    // hard-codes a version in lib/vla-assets.ts fails here on the next bump.
    expect(VLA_VERSION).toBe(pkgVersion);
    expect(VLA_ASSET_BASE).toBe(`/vla/${pkgVersion}`);
  });

  it("is root-relative with no trailing slash", () => {
    // loadEmbeddings joins `${assetBase}/vocab.txt`; a trailing slash or a
    // relative base yields a URL that 404s only at runtime.
    expect(VLA_ASSET_BASE.startsWith("/")).toBe(true);
    expect(VLA_ASSET_BASE.endsWith("/")).toBe(false);
  });

  it("has the copy script's output directory where Hero will look for it", () => {
    expect(
      existsSync(assetDir),
      `${assetDir} is missing — run \`node scripts/copy-vla-assets.mjs\``,
    ).toBe(true);
    expect(statSync(assetDir).isDirectory()).toBe(true);
  });

  it.each(VLA_RUNTIME_ASSETS)("serves %s from that directory", (file) => {
    const path = join(assetDir, file);
    expect(existsSync(path), `${path} is missing`).toBe(true);
    // A zero-byte file would be fetched happily and then fail validation deep
    // inside loadEmbeddings, where the error is far from the cause.
    expect(statSync(path).size).toBeGreaterThan(0);
  });
});

// The replay fallback's assets — the SAME reachability seam as the embeddings
// above, one layer deeper. When the package swaps to the CPU-backend replay
// (the iOS/iPadOS path, where the real WebGL run can't get going), it fetches
// replay/manifest.json from assetBase and then each checkpoint bin the manifest
// names. These are copied automatically by copy-vla-assets.mjs's recursive copy,
// but nothing else asserts they arrived: a missing or zeroed checkpoint would
// look perfectly healthy until an actual iPad visitor triggers the swap and the
// replay 404s. This block fails the build instead. Parse the checkpoint list
// from the manifest — `npm run gen:replay` can change the count — rather than
// hard-coding the current six.
describe("mini-vla replay assets", () => {
  const manifestPath = join(assetDir, VLA_REPLAY_MANIFEST);

  it("serves a non-empty replay manifest where the replay will look for it", () => {
    expect(existsSync(manifestPath), `${manifestPath} is missing`).toBe(true);
    expect(statSync(manifestPath).size).toBeGreaterThan(0);
  });

  it("serves every checkpoint the manifest names, each non-empty", () => {
    const manifest = JSON.parse(readFileSync(manifestPath, "utf8")) as {
      checkpoints: { file: string }[];
    };
    // A manifest that parses but lists no checkpoints is as broken as a missing
    // one — the replay would load nothing and never produce a rollout.
    expect(Array.isArray(manifest.checkpoints)).toBe(true);
    expect(manifest.checkpoints.length).toBeGreaterThan(0);
    const replayDir = join(assetDir, "replay");
    for (const { file } of manifest.checkpoints) {
      const path = join(replayDir, file);
      expect(existsSync(path), `${path} is missing`).toBe(true);
      expect(statSync(path).size, `${path} is empty`).toBeGreaterThan(0);
    }
  });
});
