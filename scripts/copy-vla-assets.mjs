// Copies the mini-vla package's embedding assets into public/vla/<version>/ so
// the hero's loadEmbeddings() can fetch them at "Start Training". Runs on
// predev + prebuild, and inside open-next.config.ts's buildCommand.
//
// The directory carries the package version because these assets are versioned
// WITH the JS that reads them: since v0.4.0 loadEmbeddings() validates the
// fetched bytes against constants compiled into the bundle and rejects a
// mismatch. Under a flat /vla/ a redeploy replaced one release's embeddings
// under another release's cached code. Versioning the path also makes the
// directory safe to cache immutably (see public/_headers).
//
// public/vla/ is gitignored — the mini-vla package is the single source of
// truth. `version` comes from its package.json (exported on purpose); the same
// import drives Hero.tsx's assetBase, and tests/unit/vla-assets.test.ts asserts
// the two agree, because nothing else does.
import { cp, mkdir } from "node:fs/promises";
import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";
import path from "node:path";

const require = createRequire(import.meta.url);
const { version } = require("mini-vla/package.json");

const root = path.join(path.dirname(fileURLToPath(import.meta.url)), "..");
const src = path.join(root, "node_modules/mini-vla/assets");
const dst = path.join(root, "public/vla", version);

await mkdir(dst, { recursive: true });
await cp(src, dst, { recursive: true, dereference: true });
console.log(`[copy-vla-assets] copied ${src} -> ${dst}`);
