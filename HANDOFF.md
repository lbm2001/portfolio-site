# Handoff — portfolio-site / fresh-review-1

> The baton. Present tense only: where the work stands *now* and what the
> next session should do. Overwrite it each session — never append. History
> lives in git and rationale in commit bodies, not here. Before the task
> ends, promote anything durable (project status, repo instructions, commit
> body) and delete this file from the branch: a finished task hands
> nothing off, and merged, a leftover baton strays onto the default branch.

- **Repo:** portfolio-site
- **Branch:** `fresh-review-1`
- **Worktree:** /root/git/worktrees/portfolio-site/fresh-review-1
- **Last updated:** 2026-07-23 17:35 UTC · server (srv1841294)

## State

Discovery + merge done. Baseline gate (typecheck/lint/test/`next build`) was
green — no fixes needed there. 7 blind sub-reviewers ran in parallel over the
scope below, then the `codebase-health` scan-only leg ran strictly after, then
everything was merged into one ranked list and de-duplicated against
`PROJECT_STATUS.md`/`PROJECT_ROADMAP.md` (both empty scaffolds — nothing to
skip) and a repo-wide TODO/FIXME grep (nothing existing matched). The full
ranked list (22 code findings + 1 process correction) was posted to the user
in chat this session — **it has no durable home yet**. Top items, so a cold
read doesn't need the chat transcript:

1. **Confirmed live bug**: `config/projects.sources.json:9` project
   `vla-rl-support` (added 2026-07-13) has zero entries in
   `lib/projects-data.json` — silently dropped by `gen-projects-data.mjs`'s
   per-project catch+warn, invisible on the live site for ~10 days.
2. **Confirmed live bug**: `package.json` pins `mini-vla#v0.7.1`, but the
   locked package's own `package.json` version is still `0.7.0` — defeats the
   version-derived asset-path guarantee `lib/vla-assets.ts` depends on.
3. **Confirmed security gap**: `.env` (a supported `GITHUB_TOKEN` source) is
   not covered by `.gitignore` (`.env*.local` only).
4. CI's `e2e` job never sets `VLA_FULL`, so `hero-full.spec.ts` — the only
   spec that waits for training convergence — is skipped by default; the
   site's centerpiece is untested by default CI.
5. Script-injection shape in `bump-mini-vla.yml` (`${{ }}` interpolated into
   shell `run:` before validation runs), with write permissions.
6. `bump-mini-vla.yml` never runs e2e before a version-bump PR merges — the
   exact gap that let #2 land.

Full list (all 23 items, with file:line and failure scenarios) is in this
session's chat transcript only right now — promote it before this task ends
(see Next action).

**Process correction to this file's own standing rules**: the "CI here is
opt-in per PR — apply a label" assumption below is **false** for this repo.
Verified: `.github/workflows/ci.yml` triggers unconditionally on
`pull_request`/`push`, no label anywhere. No label is needed on the review PR.

**New review dimension this round surfaced** (not yet in the
codebase-review skill's catalog): *silent partial-failure in build-time
generation loops* — whether a per-item failure in a commit-time
data-generation loop (content fetch, asset copy, config parsing) fails the
build, warns-and-skips, or silently produces wrong/missing output with no
signal. 4 of this round's findings are instances of it.

Scope — the review-dimensions generic core, applied to this repo:

- Adversarial static read against this repo's own documented pitfalls: read
  `CLAUDE.md` closely first (content-fetch-at-build-time model, the
  `lib/vla-assets.ts` version-derived asset path, the two independent CI
  lanes, the cache-busting/stale-chunk reload path in `components/Hero.tsx`)
  and hunt for at least one instance where the code doesn't actually hold
  the guarantee the doc claims.
- Docs-vs-code drift: README, `CLAUDE.md`, inline comments, `config/*.sources.json`
  examples — do they still match what the code does today?
- Test blind spots and harness isolation: guards without tests, e2e specs
  that only run under `VLA_FULL=1`, assertions that could pass vacuously,
  anything in the Playwright/Workers e2e path that only exercises one
  platform.
- Config and permission safety: `.dev.vars.example` vs actual required env,
  `config/*.sources.json` reach, anything that assumes `GITHUB_TOKEN` is
  present without the documented degrade-gracefully path actually holding.
- CI correctness and guard coverage: do `.github/workflows/ci.yml`'s two
  lanes (`check`, `e2e`) run what `CLAUDE.md` claims; does `bump-mini-vla.yml`
  correctly gate on the version the asset path depends on.
- Cross-file consistency: anywhere a constant, path, or contract is
  duplicated between `lib/content.ts`, `lib/vla-assets.ts`, the generator
  scripts, and the components that consume them, and kept in sync only by
  convention or comment.
- Derived-but-unused values and guard patterns with blind spots, per the
  catalog's general description — this repo has no toolkit-specific
  anchors (`bin/vibe`, `install.sh`, `docs/skill-quality.md`), so treat the
  generic core as the full scope. Do not import the agentic-dev-toolkit's
  own review anatomy into this repo.

This repo is not the toolkit — extend scope from `CLAUDE.md` and this
repo's own docs, not from the toolkit's conventions.

Round-1 extras (no prior round to build on):

- After the rebase, dispatch parallel sub-reviewers over independent
  dimensions — never before: a stale tree wastes the whole fan-out.
- Discover blind, then de-duplicate: complete the full discovery pass
  before reading `PROJECT_STATUS.md` / `PROJECT_ROADMAP.md` or grepping for
  existing TODOs. A re-found item already tracked there is replication —
  report it separately, never as a new finding.
- Run a `codebase-health` pass as its own leg, strictly after the blind
  discovery pass finishes — reading a tool-driven scan first anchors the
  reviewer to what the tool already knows. Scan only: stop at that skill's
  report, take no approval step, open none of its `health/*` branches or
  PRs. Fold whatever survives into the ranked list, de-duplicated against
  the discovery pass, and note which language reference it used (or that it
  had none).

## Next action

Discovery/merge (former steps 1-3) is done — do not re-run the fan-out.
Waiting on the owner for how to proceed with the results:

1. Decide which findings get fixed now vs. filed for later, and whether the
   findings list becomes its own committed report + PR (the standing rule
   below about labeling "the review PR" assumes one gets opened) or stays a
   chat-only artifact.
2. Whatever is decided, promote the findings list out of chat into a durable
   home before this task ends — a committed report, `PROJECT_ROADMAP.md`
   items, or PR descriptions — per this file's own hygiene rule. Right now
   it lives only in this session's transcript, which does not survive.
3. Add the new dimension named in **State** above to the codebase-review
   skill's dimension catalog.

## Blockers

None — waiting on owner direction (see Next action), not stuck.

Standing rules for this round:

- One commit per concern, with the failure it prevents in the commit body,
  so any single fix can be reverted alone.
- Get owner go-ahead before any high-blast-radius fix (installer/uninstall
  safety, destructive git paths, permission or settings changes) — this
  repo has none of those by nature, but a finding could still touch CI or
  the pre-push hook, which counts.
- CI here is opt-in per PR — apply whatever label this repo's workflow
  requires to the review PR, or the run reads as passing when it never ran.
- End the round by naming any new review dimension its findings imply, so
  it can be added to the codebase-review skill's dimension catalog instead
  of evaporating.

## Gotchas (unpromoted)

- This worktree/container had no Node.js installed at all (not even via nvm —
  `nvm`/`node`/`npm` were all missing from PATH, despite `.nvmrc` requiring
  22). Had to download a standalone Node v22.17.0 tarball and symlink
  `node`/`npm`/`npx` into `/usr/local/bin` before anything in CLAUDE.md's
  Commands section would run. Worth checking if this is specific to this
  container or a recurring provisioning gap — if it recurs, promote to repo
  or toolkit docs.
