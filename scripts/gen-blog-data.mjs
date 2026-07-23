#!/usr/bin/env node
// Pulls blog posts straight from a single GitHub repo at BUILD time and writes
// lib/posts-data.json, which lib/content.ts imports as a static module (same
// pattern as gen-projects-data.mjs / gen-resume-data.mjs). Doing this at build
// time keeps ALL fetching out of the request path: Cloudflare Workers (OpenNext)
// has no runtime filesystem or reliable outbound fetch during render, so the site
// serves a fully static, pre-fetched snapshot.
//
// Unlike projects (one repo each, listed explicitly), the blog is ONE repo with
// MANY posts. config/blog.sources.json gives { repo, dir, file }; this script LISTS the
// `dir/` directory and treats every subfolder as a post — so you publish a post
// by committing a new `posts/<slug>/index.md` to the blog repo, with no edit here.
//
// Per post folder `dir/<slug>/` it reads:
//   index.md            frontmatter (title/date/cat/excerpt/aiAssisted) + a
//                       Markdown body (see lib/post-md.ts)
//   *.png, *.jpg, ...   images the body references relatively; downloaded into
//                       public/blog/<slug>/ and the body's paths rewritten to /blog/...
//
// ROBUST BY DESIGN: this NEVER fails the build. If the token is missing or the
// repo can't be fetched, the previous committed lib/posts-data.json is reused, so
// the site always builds with last-known data. An EMPTY posts dir is a valid
// result, though: the listing succeeding with zero post folders writes [] (the
// blog page then shows "Writing Coming Soon"). Only an ERROR keeps existing data.
// Exception: a listing that comes back empty when lib/posts-data.json already
// has posts is treated as suspicious, not a legitimate empty-blog transition —
// see the guard in main(). ghList() maps both "dir/repo doesn't exist" and "the
// token can't see this private repo" to the same 404-derived [], so a narrowed
// or revoked GITHUB_TOKEN would otherwise silently wipe every committed post.
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { parsePostMd } from "../lib/post-md.ts";
import { loadToken } from "./lib/github-token.mjs";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const OUT = path.join(root, "lib", "posts-data.json");

const TOKEN = loadToken(root);

function ghHeaders() {
  const h = { Accept: "application/vnd.github+json", "User-Agent": "portfolio-build" };
  if (TOKEN) h.Authorization = `Bearer ${TOKEN}`;
  return h;
}

// List a repo directory. Returns [] for a missing dir (404) — an empty blog is a
// valid state, not an error.
async function ghList(repo, dirPath) {
  const url = `https://api.github.com/repos/${repo}/contents/${dirPath}`;
  const res = await fetch(url, { headers: ghHeaders() });
  if (res.status === 404) return [];
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

// Sort key from a "Mon YYYY" (or any Date-parseable) date string; newest first.
function dateKey(s) {
  const t = Date.parse(`1 ${s}`);
  return Number.isNaN(t) ? 0 : t;
}

async function buildOne(repo, dir, file, slug) {
  const inDir = (rel) => `${dir}/${slug}/${rel}`;
  const mdText = (await ghFile(repo, inDir(file))).toString("utf8");
  const fm = parsePostMd(mdText);

  const publicDir = path.join(root, "public", "blog", slug);

  // Download every relatively-referenced image and rewrite its path.
  let body = fm.body;
  const imgRe = /!\[[^\]]*\]\((?!https?:|\/)([^)\s]+)/g;
  const assets = new Set();
  let m;
  while ((m = imgRe.exec(body))) assets.add(m[1]);
  if (assets.size) fs.mkdirSync(publicDir, { recursive: true });
  for (const rel of assets) {
    const base = path.basename(rel);
    try {
      const bytes = await ghFile(repo, inDir(rel));
      fs.writeFileSync(path.join(publicDir, base), bytes);
      body = body.split(`](${rel}`).join(`](/blog/${slug}/${base}`);
    } catch (e) {
      console.warn(`  ! asset ${rel} for ${slug}: ${e.message}`);
    }
  }

  return {
    slug,
    date: fm.date || "",
    cat: fm.cat || "",
    title: fm.title || slug,
    excerpt: fm.excerpt || "",
    ...(fm.aiAssisted ? { aiAssisted: true } : {}),
    ...(body ? { body } : {}),
  };
}

async function main() {
  const cfg = JSON.parse(fs.readFileSync(path.join(root, "config", "blog.sources.json"), "utf8"));
  const { repo, dir, file } = cfg;

  if (!TOKEN) {
    console.warn(
      "gen-blog-data: no GITHUB_TOKEN found — keeping committed lib/posts-data.json",
    );
    return;
  }

  let entries;
  try {
    entries = await ghList(repo, dir);
  } catch (e) {
    console.warn(
      `gen-blog-data: could not list ${repo}/${dir} (${e.message}) — keeping committed lib/posts-data.json`,
    );
    return;
  }

  const slugs = entries.filter((e) => e.type === "dir").map((e) => e.name);
  const existing = readExisting();
  const byslug = Object.fromEntries(existing.map((p) => [p.slug, p]));

  if (slugs.length === 0 && existing.length > 0) {
    console.warn(
      `gen-blog-data: listing ${repo}/${dir} returned zero post folders, but lib/posts-data.json already has ${existing.length} — likely a misconfigured dir or an unauthorized token (both surface as 404), not a newly-emptied blog; leaving the file untouched`,
    );
    return;
  }

  const out = [];
  for (const slug of slugs) {
    try {
      out.push(await buildOne(repo, dir, file, slug));
      console.log(`gen-blog-data: fetched ${slug} <- ${repo}/${dir}/${slug}`);
    } catch (e) {
      console.warn(
        `gen-blog-data: ${slug} failed (${e.message}) — keeping existing entry`,
      );
      if (byslug[slug]) out.push(byslug[slug]);
    }
  }

  out.sort((a, b) => dateKey(b.date) - dateKey(a.date));
  fs.writeFileSync(OUT, JSON.stringify(out, null, 2) + "\n");
  console.log(`gen-blog-data: wrote lib/posts-data.json (${out.length} posts)`);
}

main().catch((e) => {
  // Even an unexpected error must not break the build — keep last-known data.
  console.warn(`gen-blog-data: unexpected error (${e.message}) — keeping committed data`);
});
