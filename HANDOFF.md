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
- **Last updated:** 2026-07-23 15:24 UTC · server (srv1841294)

## State

Not started. This is review round 1 for portfolio-site — no prior review
round exists, so there is no coverage to skip and no review-PR history to
de-duplicate against.

Mission: run one independent, blind review pass over the whole repo, then a
scan-only `codebase-health` leg, then merge the two into a single ranked
findings list.

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

1. `git rebase origin/main` (never the resume verb — it fast-forwards to
   this branch's own upstream and reports up to date while arbitrarily far
   behind the default branch).
2. Run the repo's own verification gate as the baseline: `npm run typecheck`,
   `npm run lint`, `npm test`, `npx next build`. A failure here is yours to
   fix or flag first — treat whatever this gate does *not* catch as the
   round's real target.
3. Fan out parallel sub-reviewers over the scope above, then run the
   `codebase-health` scan leg, then merge into one ranked findings list —
   `file:line` and a concrete failure scenario each, reasoned-but-unproven
   claims labelled as such.

## Blockers

None known yet.

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

None yet — this section is for surprises the round turns up that aren't
resolved by the time it hands off.
