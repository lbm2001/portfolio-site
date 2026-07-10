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
import { version } from "mini-vla/package.json";

export const VLA_VERSION: string = version;

/** Passed to `new VLATrainer({ assetBase })`. Root-relative, no trailing slash. */
export const VLA_ASSET_BASE = `/vla/${VLA_VERSION}`;

/** The two files loadEmbeddings() fetches at runtime. */
export const VLA_RUNTIME_ASSETS = ["embeddings-50d.bin", "vocab.txt"] as const;
