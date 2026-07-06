#!/usr/bin/env node
// Pulls the résumé (resume.tex + resume.pdf) from the PRIVATE source repo in
// config/resume.source.json at BUILD time and writes them into public/. That repo is the
// SINGLE SOURCE OF TRUTH — this project keeps NO committed copy of the résumé —
// so this step is REQUIRED: if the token is missing or the fetch fails, it errors
// and fails the build rather than proceeding without a résumé.
//
// Runs BEFORE gen-resume-data.mjs (see build-resume.sh), which parses the
// freshly-pulled public/resume.tex into lib/resume-data.json.
//
// Same private-repo fetch as gen-projects-data.mjs: read a token from the env or a
// gitignored dotfile and hit the GitHub Contents API (serves raw bytes of private
// files). The source repo/paths live in config/resume.source.json:
//   { repo, dir, tex, pdf }  ->  <repo>/<dir>/<tex> and <repo>/<dir>/<pdf>
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
      Authorization: `Bearer ${TOKEN}`,
    },
  });
  if (!res.ok) throw new Error(`${res.status} ${res.statusText} for ${repo}/${filePath}`);
  return Buffer.from(await res.arrayBuffer());
}

async function main() {
  const cfg = JSON.parse(fs.readFileSync(path.join(root, "config", "resume.source.json"), "utf8"));
  const dir = cfg.dir ? `${cfg.dir}/` : "";

  if (!TOKEN) {
    throw new Error(
      `no GITHUB_TOKEN found — it is REQUIRED to fetch the résumé from ${cfg.repo} ` +
        `(this repo keeps no committed copy). Set GITHUB_TOKEN in the environment or in .dev.vars.`,
    );
  }

  // Both files are required: a failure fails the build (there is no fallback).
  for (const key of ["tex", "pdf"]) {
    const name = cfg[key];
    if (!name) continue;
    const bytes = await ghFile(cfg.repo, `${dir}${name}`);
    fs.writeFileSync(path.join(root, "public", name), bytes);
    console.log(`gen-resume-source: fetched public/${name} <- ${cfg.repo}/${dir}${name}`);
  }
}

main().catch((e) => {
  console.error(`gen-resume-source: ${e.message}`);
  process.exit(1);
});
