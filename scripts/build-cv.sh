#!/usr/bin/env bash
# Compiles public/resume.tex -> public/resume.pdf using a scratch build dir so
# LaTeX byproducts (.aux, .log, .fls, ...) never land in public/.
set -euo pipefail
cd "$(dirname "$0")/.."

if ! command -v latexmk >/dev/null 2>&1; then
  echo "warning: latexmk not found, skipping resume PDF build (public/resume.pdf will not be updated)" >&2
  exit 0
fi

build_dir="$(mktemp -d)"
trap 'rm -rf "$build_dir"' EXIT

latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir="$build_dir" public/resume.tex
cp "$build_dir/resume.pdf" public/resume.pdf
