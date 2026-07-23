# Project Status — portfolio-site

> Long-lived, one per repo. The durable *current* picture of the project:
> what it is for, how it is built, what was decided, what is open. A
> snapshot, not a log — git history is the log, so finished work is removed
> rather than archived here.

_Last updated: 2026-07-23 · server (srv1841294)_

## Goal

Lukas's personal portfolio site (projects, blog, résumé), deployed to
Cloudflare Workers. Its centerpiece is a live, in-browser TensorFlow.js
behavior-cloning demo (`components/Hero.tsx`) built on the `mini-vla`
package — the site doubles as a working demonstration of the author's
robot-learning work, not just a static résumé page.

## Architecture

See `CLAUDE.md` for the full picture (build-time content fetch, the two CI
lanes, cache-busting/stale-tab reload, the Hero VLA demo's watchdog/replay
design) — that file is kept accurate and is the canonical reference; this
section stays a pointer rather than a second copy that can drift from it.

## Key decisions

- 2026-07-23 (round 1, PR #57): mini-vla version/tag mismatches are checked
  but not build-blocking — a mismatch can be harmless (verified for
  v0.7.1/0.7.0 by diffing asset trees); see `bump-mini-vla.yml`.
- 2026-07-23 (round 1, PR #57): `.env` added to `.gitignore` — was a
  supported token source but not ignored, a live-PAT-commit risk.
- 2026-07-23 (round 1, PR #57): `app/vla-debug/` (a leftover debug route)
  removed after confirming its root cause (mini-vla asset-path drift) was
  already fixed elsewhere.
- 2026-07-23 (round 1, PR #57): proposed a new codebase-review dimension,
  "silent partial-failure in build-time generation loops" — lives in the
  separate `agentic-dev-toolkit` repo, not actioned here.
- 2026-07-23 (round 2, this branch): `pauseTraining()` now clears the
  train-stall watchdog on every pause — a short pause/resume near the
  stall deadline could previously false-trip it. See
  `docs/review-round-2-findings.md` #1.
- 2026-07-23 (round 2, this branch): `gen-blog-data.mjs` now refuses to
  overwrite a non-empty `lib/posts-data.json` with an empty listing result
  (previously indistinguishable from a genuinely empty blog). See
  `docs/review-round-2-findings.md` #2.
- 2026-07-23 (round 2, this branch): the `bump-mini-vla.yml`
  version-consistency check stays warn-only, not build-blocking, by design
  (same reasoning as the round-1 decision above) — its PR-body wording was
  fixed to stop implying otherwise. See `docs/review-round-2-findings.md` #3
  and its "Open questions" below.

## Roadmap

Planned work lives in `PROJECT_ROADMAP.md`, one designed item per task —
this file keeps only the pointer.

## Open questions

_Unresolved things that block or shape the work._

- Should `bump-mini-vla.yml`'s version-consistency check be hardened from
  warn-only to build-blocking (with a manual-override escape hatch for a
  verified-harmless mismatch), or is warn-only the right permanent design?
  See `docs/review-round-2-findings.md` #3.
- `nightly-e2e-full.yml` has no real failure-notification path beyond
  GitHub's default scheduled-workflow email — worth adding (e.g.
  issue-on-failure), but needs an owner call on the `issues: write`
  permission grant and dedup logic. See `docs/review-round-2-findings.md` #13.
