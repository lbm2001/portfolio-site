# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
nvm use && npm install          # Node >= 22.6 required (uses --experimental-strip-types)
cp .dev.vars.example .dev.vars  # then fill in GITHUB_TOKEN (optional for dev, see below)

npm run dev                     # copies VLA assets, refreshes projects+blog, starts next dev
npm run typecheck               # tsc --noEmit
npm run lint                    # eslint .
npm test                        # vitest run (pretest stages VLA assets first)
npm run test:watch              # vitest watch mode
npx vitest run tests/unit/richtext.test.tsx   # single test file
npx next build                  # builds from committed data, no token needed — what CI runs
npm run build                   # full content fetch (needs GITHUB_TOKEN) + next build

npm run e2e:build && npm run e2e       # Playwright against the real Workers bundle (wrangler dev)
npm run e2e:full                       # adds the slow train-to-convergence hero specs (VLA_FULL=1)
npm run preview                        # build + serve the Workers bundle locally
npm run deploy                         # build + deploy to Cloudflare Workers
```

`.githooks/pre-push` blocks direct pushes to `main`; after cloning, run
`git config core.hooksPath .githooks` once (per clone/worktree).

## Architecture

**Content is fetched at build time, never at request time.** Cloudflare
Workers has no runtime filesystem, so `scripts/gen-*.mjs` (run via `predev`/
`prebuild`) pull from private GitHub repos listed in `config/*.sources.json`
into `lib/*-data.json`. Those JSON files are **committed**, so `next dev`,
`tsc`, and CI all run offline from last-known content — a missing
`GITHUB_TOKEN` degrades projects/blog to the committed snapshot but hard-fails
the résumé fetch (no fallback exists for it). `lib/content.ts` is the single
import point for this data (`projects`, `posts`, `profile`, `nav`) — pages
never read the generator scripts or `config/` directly.

**The Hero VLA demo is the centerpiece and lives almost entirely in
`components/Hero.tsx`** (~2500 lines): a TensorFlow.js behavior-cloning
policy that trains in a Web Worker against an analytical-IK expert, then
takes typed commands. The model, geometry, scene rendering, and rollout
engine are all imported from the external `mini-vla` package (pinned in
`package.json` as a GitHub dependency, bumped by `.github/workflows/
bump-mini-vla.yml`); this repo only renders the surrounding encoder/status
panels and wires up the worker lifecycle. `components/hero/guidance.tsx`
holds the static narration/guidance layer extracted out of `Hero.tsx`.

**`lib/vla-assets.ts` derives the runtime asset path from `mini-vla`'s own
package version** (`VLA_ASSET_BASE = /vla/${pkg.version}`), rather than
hardcoding it, because assets are validated against constants compiled into
that exact package version — a stale hardcoded path 404s silently on the next
bump. `scripts/copy-vla-assets.mjs` copies `node_modules/mini-vla/assets/`
into `public/vla/<version>/` on `predev`/`prebuild`/`pretest`, since the
directory is gitignored.

**CI runs two independent lanes** (`.github/workflows/ci.yml`): `check`
(typecheck/lint/unit/`next build`, all from committed data, no secrets
needed — this is what a fork's PR can run) and `e2e` (full OpenNext Workers
bundle + Playwright against `wrangler dev`, i.e. the actual deploy runtime).
Both intentionally run `npx next build` / `npm run e2e:build` rather than
`npm run build`, since the latter requires `GITHUB_TOKEN`. Neither lane sets
`VLA_FULL`, so the slow train-to-convergence specs
(`tests/e2e/hero-full.spec.ts`) never run on a normal push or PR by design —
`.github/workflows/nightly-e2e-full.yml` runs them against `main` on a daily
cron instead, and `.github/workflows/bump-mini-vla.yml` runs them on every
mini-vla version bump, so a regression still surfaces within a day rather
than never.

**Cache-busting across deploys**: every deploy produces a fresh
content-hashed asset manifest, so page HTML is served
`Cache-Control: max-age=0, must-revalidate` (`next.config.mjs`). A tab left
open across a deploy holds stale chunk URLs in memory; when it lazily
requests one (e.g. constructing the trainer worker on "Start Training"), the
404 is caught and surfaces as a **Reload** prompt (`loadFailed` in
`components/Hero.tsx`) rather than a silent failure.

## Layout

```
app/          App Router pages: home, about, projects, blog, resume
components/   Hero (the VLA demo), Nav, Footer, page sections
lib/          Content types + committed data JSON, markdown/LaTeX parsing
scripts/      Build-time content fetch + parse scripts
config/       Which repos/paths content is pulled from
```
