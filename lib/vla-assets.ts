// Where the hero fetches mini-vla's runtime embedding assets from.
//
// Since mini-vla v0.4.0 the assets are validated against constants compiled
// into the package's JS, so a build must only ever be served the assets that
// shipped with it. The package version is therefore part of the URL, and it is
// read from the package's own manifest rather than written down anywhere — a
// hard-coded copy here (or in scripts/copy-vla-assets.mjs) would drift on the
// next bump and 404 on the visitor's first click.
//
// scripts/copy-vla-assets.mjs copies into `public${VLA_ASSET_BASE}/`.
// tests/unit/vla-assets.test.ts asserts that directory exists and holds the two
// runtime files, and that both sides still derive from this one import.
//
// Only embeddings-50d.bin and vocab.txt are fetched at runtime; grammar.json is
// a bundle-time import and assetBase does not apply to it.
// Node's native ESM loader (which e2e's Playwright/wrangler run under, unlike
// Next's bundler) requires this attribute on a bare JSON import — without it,
// tests/e2e/routes.spec.ts's import chain throws "needs an import attribute
// of type: json" before a single test runs. It also only provides a DEFAULT
// export (no per-property named exports the way a bundler's JSON loader
// allows), hence destructuring off `pkg` below rather than a named import.
import pkg from "mini-vla/package.json" with { type: "json" };

export const VLA_VERSION: string = pkg.version;

/** Passed to `new VLATrainer({ assetBase })`. Root-relative, no trailing slash. */
export const VLA_ASSET_BASE = `/vla/${VLA_VERSION}`;

/** The two files loadEmbeddings() fetches at runtime. */
export const VLA_RUNTIME_ASSETS = ["embeddings-50d.bin", "vocab.txt"] as const;

/**
 * The replay fallback's manifest, fetched at runtime from `assetBase` exactly
 * like the embeddings when the package swaps to the CPU-backend replay (the
 * iOS/iPadOS path). It lists the checkpoint bins the replay loads; both live
 * under `public${VLA_ASSET_BASE}/replay/` via the recursive asset copy.
 * Relative to VLA_ASSET_BASE, no leading slash. tests/unit/vla-assets.test.ts
 * asserts this file and every checkpoint it names are present and non-empty.
 */
export const VLA_REPLAY_MANIFEST = "replay/manifest.json";
