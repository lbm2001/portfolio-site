#!/usr/bin/env node
// Pulls the résumé source (resume.tex + resume.pdf) from a PRIVATE GitHub repo at
// BUILD time and writes them into public/, so the committed copies never have to
// hold the real résumé. Runs BEFORE gen-resume-data.mjs (see build-resume.sh),
// which then parses the freshly-pulled public/resume.tex into lib/resume-data.json.
//
// Same private-repo fetch pattern as gen-projects-data.mjs: read a token from the
// env or a gitignored dotfile and hit the GitHub Contents API (which serves raw
// bytes of private files). The source repo/paths live in resume.source.json:
//   { repo, dir, tex, pdf }  ->  <repo>/<dir>/<tex> and <repo>/<dir>/<pdf>
//
// ROBUST BY DESIGN: this NEVER fails the build. With no token, or if the repo /
// files can't be fetched, the committed public/resume.tex and public/resume.pdf
// are kept, so the build always has a last-known résumé and works offline.
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

// --- token: env, or a gitignored .dev.vars / .env.local (build scripts run
// before Next loads env, so we read it ourselves). Mirrors gen-projects-data.mjs.
function loadToken() {
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
const TOKEN = loadToken();

// Fetch a repo file's raw bytes via the Contents API (works for private repos).
async function ghFile(repo, filePath) {
  const url = `https://api.github.com/repos/${repo}/contents/${filePath}`;
  const res = await fetch(url, {
    headers: {
      Accept: "application/vnd.github.raw",
      "User-Agent": "portfolio-build",
      ...(TOKEN ? { Authorization: `Bearer ${TOKEN}` } : {}),
    },
  });
  if (!res.ok) throw new Error(`${res.status} ${res.statusText} for ${repo}/${filePath}`);
  return Buffer.from(await res.arrayBuffer());
}

async function main() {
  const cfg = JSON.parse(fs.readFileSync(path.join(root, "resume.source.json"), "utf8"));
  const dir = cfg.dir ? `${cfg.dir}/` : "";

  if (!TOKEN) {
    console.warn("gen-resume-source: no GITHUB_TOKEN found — keeping committed public/resume.*");
    return;
  }

  // Each file is best-effort and independent: a failure keeps the committed copy.
  for (const key of ["tex", "pdf"]) {
    const name = cfg[key];
    if (!name) continue;
    const out = path.join(root, "public", name);
    try {
      const bytes = await ghFile(cfg.repo, `${dir}${name}`);
      fs.writeFileSync(out, bytes);
      console.log(`gen-resume-source: fetched public/${name} <- ${cfg.repo}/${dir}${name}`);
    } catch (e) {
      console.warn(`gen-resume-source: ${name} failed (${e.message}) — keeping committed public/${name}`);
    }
  }
}

main().catch((e) => {
  // Even an unexpected error must not break the build — keep last-known résumé.
  console.warn(`gen-resume-source: unexpected error (${e.message}) — keeping committed public/resume.*`);
});
