#!/usr/bin/env bash
# Build-time resume artifacts. Runs in every build env (local + Cloudflare).
set -euo pipefail
cd "$(dirname "$0")/.."

# 1) Always regenerate the parsed CV data the /resume page imports. Node is
#    present in every build env; this keeps LaTeX parsing at build time and out of
#    the request path (Cloudflare Workers has no runtime filesystem, so reading the
#    .tex during render throws "Internal Server Error").
node --experimental-strip-types scripts/gen-cv-data.mjs

# 2) Best-effort: compile public/resume.tex -> public/resume.pdf when latexmk is
#    available. Cloudflare's build image has no latexmk, so the committed
#    public/resume.pdf is served there instead. A scratch build dir keeps LaTeX
#    byproducts (.aux, .log, .fls, ...) out of public/.
if ! command -v latexmk >/dev/null 2>&1; then
  echo "warning: latexmk not found, skipping resume PDF build (using committed public/resume.pdf)" >&2
  exit 0
fi

build_dir="$(mktemp -d)"
trap 'rm -rf "$build_dir"' EXIT

latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir="$build_dir" public/resume.tex
cp "$build_dir/resume.pdf" public/resume.pdf
