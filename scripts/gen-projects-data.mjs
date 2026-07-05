#!/usr/bin/env node
// Pulls each project's data straight from its GitHub repo at BUILD time and
// writes lib/projects-data.json, which lib/content.ts imports as a static module
// (same pattern as gen-resume-data.mjs / lib/resume-data.json). Doing this at build time
// keeps ALL fetching out of the request path: Cloudflare Workers (OpenNext) has
// no runtime filesystem or reliable outbound fetch during render, so the site
// serves a fully static, pre-fetched snapshot.
//
// Everything a project exposes to the site lives in ONE self-contained
// `portfolio/` directory in its repo (dir + file are set in projects.sources.json):
//   portfolio/README.md      frontmatter + Markdown body (GitHub renders it when
//                            you open the folder)
//   portfolio/assets/*.png   images the README references relatively
//   portfolio/paper.pdf      an optional paper linked from the frontmatter
//
// Per repo it reads:
//   - GitHub repo metadata:  description -> blurb (fallback), topics -> tags
//                            (fallback), html_url -> the "Code" link.
//   - portfolio/README.md:   frontmatter (title/venue/period/blurb/tags/links/
//                            aiAssisted) + a Markdown body.
//   - referenced assets:     images and paper links that are RELATIVE to the
//                            README are resolved inside portfolio/, downloaded
//                            into public/projects/<slug>/, and the body's paths
//                            are rewritten to /projects/...
//
// ROBUST BY DESIGN: this NEVER fails the build. If the token is missing or a
// repo/file can't be fetched, the previous entry from the committed
// lib/projects-data.json is reused, so the site always builds with last-known
// data. That also makes the whole thing testable offline.
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { parseProjectMd } from "../lib/project-md.ts";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const OUT = path.join(root, "lib", "projects-data.json");

// --- token: read from env, or a gitignored .dev.vars / .env.local (the build
// scripts run before Next loads env, so we read it ourselves).
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

function ghHeaders() {
  const h = { Accept: "application/vnd.github+json", "User-Agent": "portfolio-build" };
  if (TOKEN) h.Authorization = `Bearer ${TOKEN}`;
  return h;
}

async function ghJson(url) {
  const res = await fetch(url, { headers: ghHeaders() });
  if (!res.ok) throw new Error(`${res.status} ${res.statusText} for ${url}`);
  return res.json();
}

// Fetch a repo file's raw bytes via the Contents API (works for private repos).
async function ghFile(repo, filePath) {
  const url = `https://api.github.com/repos/${repo}/contents/${filePath}`;
  const res = await fetch(url, {
    headers: { ...ghHeaders(), Accept: "application/vnd.github.raw" },
  });
  if (!res.ok) throw new Error(`${res.status} for ${repo}/${filePath}`);
  return Buffer.from(await res.arrayBuffer());
}

function readExisting() {
  try {
    return JSON.parse(fs.readFileSync(OUT, "utf8"));
  } catch {
    return [];
  }
}

function topicToTag(t) {
  // GitHub topics are lowercase-hyphenated; make a readable-ish label.
  return t.replace(/-/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

async function buildOne(source, dir, file) {
  const { slug, repo } = source;
  // Resolve a path written RELATIVE to the README (which lives in `dir/`) to its
  // location in the repo, e.g. dir="portfolio" + "assets/x.png" -> "portfolio/assets/x.png".
  const inDir = (rel) => (dir ? `${dir}/${rel}` : rel);

  const meta = await ghJson(`https://api.github.com/repos/${repo}`);
  const mdText = (await ghFile(repo, inDir(file))).toString("utf8");
  const fmParsed = parseProjectMd(mdText);

  const publicDir = path.join(root, "public", "projects", slug);
  fs.mkdirSync(publicDir, { recursive: true });

  // Download every relatively-referenced image and rewrite its path.
  let body = fmParsed.body;
  const imgRe = /!\[[^\]]*\]\((?!https?:|\/)([^)\s]+)/g;
  const assets = new Set();
  let m;
  while ((m = imgRe.exec(body))) assets.add(m[1]);
  for (const rel of assets) {
    const base = path.basename(rel);
    try {
      const bytes = await ghFile(repo, inDir(rel));
      fs.writeFileSync(path.join(publicDir, base), bytes);
      body = body.split(`](${rel}`).join(`](/projects/${slug}/${base}`);
    } catch (e) {
      console.warn(`  ! asset ${rel} for ${slug}: ${e.message}`);
    }
  }

  // Links: Code (repo) is always first, then the frontmatter links. If a link's
  // href is relative (e.g. a paper PDF inside portfolio/), download it too.
  const links = [{ label: "Code", href: meta.html_url }];
  for (const lk of fmParsed.links || []) {
    if (!/^https?:|^\//.test(lk.href)) {
      const base = path.basename(lk.href);
      try {
        const bytes = await ghFile(repo, inDir(lk.href));
        fs.writeFileSync(path.join(publicDir, base), bytes);
        links.push({ label: lk.label, href: `/projects/${slug}/${base}` });
        continue;
      } catch (e) {
        console.warn(`  ! link asset ${lk.href} for ${slug}: ${e.message}`);
      }
    }
    links.push(lk);
  }

  return {
    slug,
    title: fmParsed.title || meta.name,
    venue: fmParsed.venue || "",
    ...(fmParsed.period ? { period: fmParsed.period } : {}),
    blurb: fmParsed.blurb || meta.description || "",
    tags: fmParsed.tags || (meta.topics || []).map(topicToTag),
    links,
    ...(fmParsed.aiAssisted ? { aiAssisted: true } : {}),
    body,
  };
}

async function main() {
  const config = JSON.parse(
    fs.readFileSync(path.join(root, "projects.sources.json"), "utf8"),
  );
  const existing = readExisting();
  const byslug = Object.fromEntries(existing.map((p) => [p.slug, p]));

  if (!TOKEN) {
    console.warn(
      "gen-projects-data: no GITHUB_TOKEN found — keeping committed lib/projects-data.json",
    );
    return;
  }

  const out = [];
  for (const source of config.projects) {
    try {
      out.push(await buildOne(source, config.dir, config.file));
      console.log(`gen-projects-data: fetched ${source.slug} <- ${source.repo}`);
    } catch (e) {
      console.warn(
        `gen-projects-data: ${source.slug} failed (${e.message}) — keeping existing entry`,
      );
      if (byslug[source.slug]) out.push(byslug[source.slug]);
    }
  }

  if (out.length === 0) {
    console.warn("gen-projects-data: nothing fetched — leaving file untouched");
    return;
  }
  fs.writeFileSync(OUT, JSON.stringify(out, null, 2) + "\n");
  console.log(`gen-projects-data: wrote lib/projects-data.json (${out.length} projects)`);
}

main().catch((e) => {
  // Even an unexpected error must not break the build — keep last-known data.
  console.warn(`gen-projects-data: unexpected error (${e.message}) — keeping committed data`);
});
