#!/usr/bin/env bash
# Compiles public/cv.tex -> public/cv.pdf using a scratch build dir so LaTeX
# byproducts (.aux, .log, .fls, ...) never land in public/.
set -euo pipefail
cd "$(dirname "$0")/.."

if ! command -v latexmk >/dev/null 2>&1; then
  echo "warning: latexmk not found, skipping CV PDF build (public/cv.pdf will not be updated)" >&2
  exit 0
fi

build_dir="$(mktemp -d)"
trap 'rm -rf "$build_dir"' EXIT

latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir="$build_dir" public/cv.tex
cp "$build_dir/cv.pdf" public/cv.pdf
