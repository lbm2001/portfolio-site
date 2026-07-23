// Shared by gen-blog-data.mjs, gen-projects-data.mjs, and gen-resume-source.mjs
// — previously copy-pasted verbatim in all three (one comment even said
// "Mirrors gen-projects-data.mjs"), so an edit to the lookup order or dotfile
// list in one would silently not reach the others.
import fs from "node:fs";
import path from "node:path";

/** Read GITHUB_TOKEN from the environment, or a gitignored .dev.vars /
 *  .env.local / .env under `root` (the build scripts run before Next loads
 *  env, so they read it themselves). Returns null if none is set. */
export function loadToken(root) {
  if (process.env.GITHUB_TOKEN) return process.env.GITHUB_TOKEN.trim();
  for (const f of [".dev.vars", ".env.local", ".env"]) {
    try {
      const txt = fs.readFileSync(path.join(root, f), "utf8");
      const m = txt.match(/^\s*GITHUB_TOKEN\s*=\s*(.+)\s*$/m);
      if (m) return m[1].replace(/^["']|["']$/g, "").trim();
    } catch {}
  }
  return null;
}
