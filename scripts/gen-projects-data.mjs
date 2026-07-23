#!/usr/bin/env node
// Pulls each project's data at BUILD time and writes lib/projects-data.json,
// which lib/content.ts imports as a static module (same pattern as
// gen-resume-data.mjs / lib/resume-data.json). Doing this at build time keeps
// ALL fetching out of the request path: Cloudflare Workers (OpenNext) has no
// runtime filesystem or reliable outbound fetch during render, so the site
// serves a fully static, pre-fetched snapshot.
//
// Content (README + assets + an optional paper) lives in ONE PRIVATE repo,
// config.contentRepo (currently lukasmueller-dev/portfolio-project-content), one
// directory per project:
//   <contentDir>/README.md      frontmatter + Markdown body
//   <contentDir>/assets/*.png   images the README references relatively
//   <contentDir>/paper.pdf      an optional paper linked from the frontmatter
// Each project's own repo (config/projects.sources.json's `repo`) supplies
// only public GitHub metadata (description -> blurb fallback, topics -> tags
// fallback, html_url -> the "Code" link) and, unless overridden, the derived
// slug/contentDir (both default to the repo name, e.g. lukasmueller-dev/mujopy ->
// "mujopy"; set `slug`/`contentDir` on a project entry to override).
//
// `repo` is OPTIONAL: for a project with no code to show, omit it and set
// `slug`/`contentDir` explicitly instead (nothing to derive them from). The
// GitHub metadata fetch is skipped entirely, so title/blurb/tags come only
// from the README frontmatter and there's no auto "Code" link.
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

// Sort key for ordering projects newest-first by the START of their working
// period. A period of "Ongoing" (active work) sorts ABOVE every dated project;
// otherwise we key on the first month + year in the string, so a range like
// "Jan–Jun 2026" keys on its start (Jan 2026) and a single "Oct 2025" on itself.
// Larger key => listed earlier; a finite sentinel avoids NaN when entries tie.
const ONGOING = Number.MAX_SAFE_INTEGER;
function periodKey(s) {
  if (!s) return 0;
  if (/ongoing/i.test(s)) return ONGOING;
  const year = s.match(/\d{4}/)?.[0];
  const month = s.match(/[A-Za-z]{3,}/)?.[0]; // first month token (start of a range)
  const t = Date.parse(`1 ${month || "Jan"} ${year || ""}`.trim());
  return Number.isNaN(t) ? 0 : t;
}

// Single source of truth for a source's slug — main() previously
// re-derived this independently (for log/catch messages) from buildOne()'s
// copy (for actual use); an edit to one without the other could silently
// make logged output and actual output disagree.
function deriveSlug(source) {
  return source.slug ?? source.repo?.split("/")[1];
}

async function buildOne(source, contentRepo, file) {
  const { repo } = source;
  const repoName = repo?.split("/")[1];
  const slug = deriveSlug(source);
  const contentDir = source.contentDir ?? repoName;
  if (!slug || !contentDir) {
    throw new Error(`no "repo" set — "slug" and "contentDir" must both be given explicitly`);
  }
  // Resolve a path written RELATIVE to the README (which lives at the root of
  // <contentDir> in contentRepo) to its location there, e.g. contentDir="mujopy"
  // + "assets/x.png" -> "mujopy/assets/x.png".
  const inContentDir = (rel) => `${contentDir}/${rel}`;

  // README first: it's the primary content (frontmatter + body); `meta` is
  // only a FALLBACK for title/blurb/tags when frontmatter omits them (see
  // header comment). Fetch it independently so a flaky/rate-limited metadata
  // call can never discard an otherwise-successful README fetch — it used to
  // throw before the README was even requested, so a good content edit could
  // silently not ship because of an unrelated metadata hiccup.
  let meta = null;
  if (repo) {
    try {
      meta = await ghJson(`https://api.github.com/repos/${repo}`);
    } catch (e) {
      console.warn(`  ! metadata for ${repo}: ${e.message} — falling back to frontmatter-only title/blurb/tags, no "Code" link`);
    }
  }
  const mdText = (await ghFile(contentRepo, inContentDir(file))).toString("utf8");
  const fmParsed = parseProjectMd(mdText);

  const publicDir = path.join(root, "public", "projects", slug);
  // Create the per-slug dir lazily, on the first asset that downloads, so
  // asset-less projects leave no empty directory behind.
  const writeAsset = (base, bytes) => {
    fs.mkdirSync(publicDir, { recursive: true });
    fs.writeFileSync(path.join(publicDir, base), bytes);
  };

  // Download every relatively-referenced image and rewrite its path.
  let body = fmParsed.body;
  const imgRe = /!\[[^\]]*\]\((?!https?:|\/)([^)\s]+)/g;
  const assets = new Set();
  let m;
  while ((m = imgRe.exec(body))) assets.add(m[1]);
  for (const rel of assets) {
    const base = path.basename(rel);
    try {
      const bytes = await ghFile(contentRepo, inContentDir(rel));
      writeAsset(base, bytes);
      body = body.split(`](${rel}`).join(`](/projects/${slug}/${base}`);
    } catch (e) {
      console.warn(`  ! asset ${rel} for ${slug}: ${e.message}`);
    }
  }

  // Links: Code (repo) is first when there is a repo, then the frontmatter
  // links. If a link's href is relative (e.g. a paper PDF next to the
  // README), download it too.
  const links = meta ? [{ label: "Code", href: meta.html_url }] : [];
  for (const lk of fmParsed.links || []) {
    if (!/^https?:|^\//.test(lk.href)) {
      const base = path.basename(lk.href);
      try {
        const bytes = await ghFile(contentRepo, inContentDir(lk.href));
        writeAsset(base, bytes);
        links.push({ label: lk.label, href: `/projects/${slug}/${base}` });
        continue;
      } catch (e) {
        console.warn(`  ! link asset ${lk.href} for ${slug}: ${e.message}`);
      }
    }
    links.push(lk);
  }

  // Project.title (lib/content.ts) is a required string, but frontmatter can
  // omit `title` and `meta` is null for a repo-less project (source has no
  // "repo") — falling back to `undefined` would render as the literal string
  // "undefined" on the page (JSON.stringify drops the key, and the `as
  // Project[]` cast at the read side gives no runtime check). Fall back to
  // the slug instead, so a missing title is visibly a slug, not a crash or a
  // literal "undefined".
  if (!fmParsed.title && !meta?.name) {
    console.warn(`  ! ${slug}: no title in frontmatter and no repo metadata — using slug as title`);
  }
  return {
    slug,
    title: fmParsed.title || meta?.name || slug,
    venue: fmParsed.venue || "",
    ...(fmParsed.period ? { period: fmParsed.period } : {}),
    blurb: fmParsed.blurb || meta?.description || "",
    tags: fmParsed.tags || (meta?.topics || []).map(topicToTag),
    links,
    ...(fmParsed.aiAssisted ? { aiAssisted: true } : {}),
    body,
  };
}

async function main() {
  const config = JSON.parse(
    fs.readFileSync(path.join(root, "config", "projects.sources.json"), "utf8"),
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
    const slug = deriveSlug(source);
    try {
      out.push(await buildOne(source, config.contentRepo, config.file));
      console.log(`gen-projects-data: fetched ${slug} <- ${source.repo ?? `${config.contentRepo}/${source.contentDir ?? slug}`}`);
    } catch (e) {
      console.warn(
        `gen-projects-data: ${slug} failed (${e.message}) — keeping existing entry`,
      );
      if (byslug[slug]) out.push(byslug[slug]);
    }
  }

  if (out.length === 0) {
    console.warn("gen-projects-data: nothing fetched — leaving file untouched");
    return;
  }
  // Newest-first by period start; "Ongoing" projects pinned to the top.
  out.sort((a, b) => periodKey(b.period) - periodKey(a.period));
  fs.writeFileSync(OUT, JSON.stringify(out, null, 2) + "\n");
  console.log(`gen-projects-data: wrote lib/projects-data.json (${out.length} projects)`);
}

main().catch((e) => {
  // Even an unexpected error must not break the build — keep last-known data.
  console.warn(`gen-projects-data: unexpected error (${e.message}) — keeping committed data`);
});
