---
name: verify
description: How to build, launch and drive this portfolio site (esp. the live VLA training hero) to verify changes end-to-end.
---

# Verifying this repo

## Test harness (prefer this over ad-hoc scripts)
- `npm test` — Vitest units: lib/ parsers (post-md, project-md, resume,
  richtext) + committed lib/*-data.json invariants.
- `npm run e2e:build && npm run e2e` — Playwright against the REAL worker
  (`opennextjs-cloudflare build` → `wrangler dev`/workerd on :8787): route
  sweep from the committed JSONs, /vla asset reachability, cache-header
  regression guard (caching.spec.ts — the stale-chunk outage), hero alive-gate
  (training starts + batches advance) on desktop (1440×900) and mobile
  (390×844) viewports. This is the "deploys cleanly on Cloudflare" gate; CI
  runs it on every PR (.github/workflows/ci.yml, e2e job).
- `npm run e2e:full` — opt-in slow specs (VLA_FULL=1): train to "Ready" and
  drive the try-it command box. ~6 min per viewport (headless SwiftShader does
  ~2 batches/s; mini-vla converges ~370-560 batches, `maxBatches: 800` is the
  hard fallback). Runs serially (workers: 1). **Never drive a second browser
  while it runs** — sharing SwiftShader cuts the batch rate ~5x and freezes the
  counter for minutes; waitForConverged() will fail it as a 180s stall.
- `npm run smoke:live` — same suite against https://lukasmueller.dev; run
  after every `npm run deploy`.
- Selectors/waits live in tests/e2e/helpers.ts — reuse them in scratch
  scripts instead of re-deriving.

## Build / typecheck
- `npx tsc --noEmit` — strict, catches most integration breaks.
- `npm run build` — Next.js 16 (Turbopack). `prebuild` builds the resume PDF.
- `opennextjs-cloudflare build` now uses a buildCommand override in
  open-next.config.ts (copy-vla-assets + `npx next build`, NO résumé fetch) —
  deploy/preview scripts fetch the résumé explicitly beforehand.

## Launch
- `npm run dev` — but the user often already has `next dev` running on
  **localhost:3000** (a second instance refuses to start and exits 1).
  `curl -s localhost:3000` first; drive the existing server if it's up —
  it hot-reloads the working tree, so it serves your edits.

## Drive the VLA hero (the main runtime surface)
- Headless browser: `playwright-core` is in node_modules (no `playwright`
  package). Import it by absolute path in a scratch script:
  `import { chromium } from "<repo>/node_modules/playwright-core/index.mjs"`.
  Launch with `executablePath` pointing at the ms-playwright Chromium cache
  (`~/Library/Caches/ms-playwright/chromium-*/…`) and args
  `--use-angle=swiftshader --enable-unsafe-swiftshader` — tfjs-webgl works on
  SwiftShader, just slower (~4-6 batches/s vs ~10 on real GPU).
- **The viewport picks the task**, so both tiers need driving:
  - ≥1100px (e.g. 1440×900): the desktop ring; trains 8 colors / ≤4 blocks.
  - <1100px (e.g. 390×844): the pipeline is `display:none` until you click
    `.hero-demo-btn` ("Mini VLA Demo"), which stacks it. Trains 4 colors /
    ≤3 blocks — a smaller task, so it converges sooner than desktop.
- Key selectors: `.vla-bar .vla-btn` (Start Training / Pause),
  `.vla-status-text` (Idle/Loading/Training/Ready), `.vla-status-sub`
  (idle: the task profile + ETA; live: examples/batches), `.hero-demo-btn` /
  `.vla-close` (mobile open/close), `.vla-prompt` (demo command),
  `.vla-decoded` (language readout), `.vla-action-vals`
  (shoulder/elbow/gripper), `.vla-try-input` / `.vla-try-btn` /
  `.vla-try-shuffle` (converged try-it), `.vla-try-chip` (mobile-only preset
  commands, one per trained color), `.vla-try-note` (fires when a command
  names a color the run never trained, or one absent from the scene).
- Training ends at `converge.maxBatches` (CONFIG) at the latest — on
  SwiftShader budget ~2-5 min until status "Ready". Wait with
  `waitForFunction` on `.vla-status-text === "Ready"`, generous timeout.
- Capture `page.on("pageerror")` and console errors — worker failures
  surface there, not in the DOM.

## Gotchas
- Training runs in a module Web Worker; a stale chunk after edits means the
  page needs a reload, not a retrain.
- tfjs GPU→CPU fallback (trainer.core) makes batches ~10x slower but keeps
  the flow alive — a "slow but moving" run may be on the cpu backend.
