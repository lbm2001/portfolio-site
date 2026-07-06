#!/usr/bin/env bash
# Build-time resume artifacts. Runs in every build env (local + Cloudflare).
set -euo pipefail
cd "$(dirname "$0")/.."

# 0) REQUIRED: pull the résumé (public/resume.tex + public/resume.pdf) from the
#    private source repo in config/resume.source.json. That repo is the single source of
#    truth — this project keeps no committed copy — so a missing GITHUB_TOKEN or a
#    failed fetch fails the build here (rather than later, cryptically, at parse).
node --experimental-strip-types scripts/gen-resume-source.mjs

# 1) Always regenerate the parsed résumé data the /resume page imports. Node is
#    present in every build env; this keeps LaTeX parsing at build time and out of
#    the request path (Cloudflare Workers has no runtime filesystem, so reading the
#    .tex during render throws "Internal Server Error").
node --experimental-strip-types scripts/gen-resume-data.mjs

# 1b) Refresh project data from each project's GitHub repo (see
#     config/projects.sources.json). Best-effort: with no GITHUB_TOKEN, or if a repo is
#     unreachable, it keeps the committed lib/projects-data.json so the build
#     never fails and always has last-known data.
node --experimental-strip-types scripts/gen-projects-data.mjs

# 1c) Refresh blog posts from the single blog repo (see config/blog.sources.json).
#     Best-effort like projects: with no GITHUB_TOKEN, or if the repo is
#     unreachable, it keeps the committed lib/posts-data.json. An EMPTY blog repo
#     is a valid result and writes [] (the /blog page shows "Writing Coming Soon").
node --experimental-strip-types scripts/gen-blog-data.mjs

# 2) Optional: recompile public/resume.tex -> public/resume.pdf when latexmk is
#    available (refreshes the PDF from the just-fetched .tex). Cloudflare's build
#    image has no latexmk, so the public/resume.pdf fetched in step 0 is served
#    there as-is. A scratch build dir keeps LaTeX byproducts (.aux, .log, ...) out
#    of public/.
if ! command -v latexmk >/dev/null 2>&1; then
  echo "note: latexmk not found, serving the resume.pdf fetched from the source repo" >&2
  exit 0
fi

build_dir="$(mktemp -d)"
trap 'rm -rf "$build_dir"' EXIT

latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir="$build_dir" public/resume.tex
cp "$build_dir/resume.pdf" public/resume.pdf
