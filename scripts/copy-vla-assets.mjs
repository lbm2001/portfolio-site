// Copies the mini-vla package's embedding assets into public/vla/ so the hero's
// loadEmbeddings() (default assetBase "/vla") can fetch them at "Start
// Training". Runs on predev + prebuild. public/vla/ is gitignored — the mini-vla
// package is the single source of truth for these assets.
import { cp, mkdir } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import path from "node:path";

const root = path.join(path.dirname(fileURLToPath(import.meta.url)), "..");
const src = path.join(root, "node_modules/mini-vla/assets");
const dst = path.join(root, "public/vla");

await mkdir(dst, { recursive: true });
await cp(src, dst, { recursive: true, dereference: true });
console.log(`[copy-vla-assets] copied ${src} -> ${dst}`);
