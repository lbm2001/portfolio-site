# Handoff — portfolio-site / fresh-review-2

> The baton. Present tense only: where the work stands *now* and what the
> next session should do. Overwrite it each session — never append. History
> lives in git and rationale in commit bodies, not here. Before the task
> ends, promote anything durable (project status, repo instructions, commit
> body) and delete this file from the branch: a finished task hands
> nothing off, and merged, a leftover baton strays onto the default branch.

- **Repo:** portfolio-site
- **Branch:** `fresh-review-2`
- **Worktree:** /root/git/worktrees/portfolio-site/fresh-review-2
- **Last updated:** 2026-07-23 18:45 UTC · server (srv1841294)

## State

Not started. This is review round 2 for portfolio-site. Round 1 landed as
PR #57 (22 findings, all fixed/tested/investigated) and merged via squash
into main at commit `814cb7e` — no other commits have landed since, so this
round reviews the exact tree round 1 left behind, fixes and all.

Mission: a second independent, blind review round — same generic-core
dimension set as round 1 (adversarial static read, docs-vs-code drift, test
blind spots, config/permission safety, CI correctness, cross-file
consistency, derived-but-unused values, guard blind spots), fanned out over
parallel sub-reviewers, followed by a scan-only `codebase-health` leg, then
merged into one ranked findings list. **You may fix what you find as you
go this round**, the same way round 1 did — the standing rules below still
apply (one commit per concern, owner go-ahead before anything high-blast-
radius).

**This round's value depends on not re-finding round 1.** Do the full
discovery pass *first* — read `CLAUDE.md` and this repo's own docs, not
`docs/review-round-1-findings.md` or PR #57 — then de-duplicate against
that report. Re-finding something round 1 already fixed is a sign the pass
is working, not a finding; report it separately from anything new.

What round 1 covered thoroughly, so you can weight effort elsewhere:
`CLAUDE.md`'s documented guarantees (content-fetch-at-build-time,
`lib/vla-assets.ts`'s version-derived path, the two CI lanes, the
cache-busting/stale-chunk reload path), `scripts/gen-projects-data.mjs` and
`scripts/gen-resume-source.mjs`'s failure semantics, the Hero.tsx watchdogs,
`config/projects.sources.json` vs. the committed data, and security/CI
issues in `bump-mini-vla.yml`/`claude-review.yml`/`ci.yml`.

Where round 1 was thin, and where this round should push hardest:

- **`scripts/gen-blog-data.mjs` never got the same scrutiny** as its two
  siblings. Round 1's whole "silent partial-failure in build-time
  generation loops" finding (config vs. committed data going silently out
  of sync, a partial fetch failure discarding good data, a required field
  silently defaulting to `undefined`) was found in `gen-projects-data.mjs`
  and `gen-resume-source.mjs` — check whether the same shapes exist here
  too. This class of defect isn't in the codebase-review skill's
  dimension catalog yet (`references/review-dimensions.md` in
  agentic-dev-toolkit) even though round 1 proposed it — name it again if
  this round confirms it's a recurring class, so it actually gets added.
- **Round 1's own new surface is unreviewed by construction** — it was
  added and fixed in the same pass that found it, so nothing adversarial
  has looked at it yet: `.github/workflows/nightly-e2e-full.yml` (new),
  the `bump-mini-vla.yml` changes (version-check step, e2e gate, the `env:`
  fix for the injection), `claude-review.yml`'s `author_association` gate,
  `lib/build-id.ts` + `tests/unit/build-id.test.ts`, the `"@"` alias added
  to `vitest.config.ts`, `tests/e2e/hero-watchdog.spec.ts`'s scripted
  `Worker` double, and the `webkit-mobile` Playwright project. Read these
  with the same adversarial eye as everything else — a fix landed under
  time pressure across many concurrent findings is exactly where a new
  defect hides.
- **Three CI workflows now each install Playwright browsers with
  near-identical steps** (`ci.yml`, `bump-mini-vla.yml`,
  `nightly-e2e-full.yml`) — worth a cross-file-consistency look; a browser
  added to one and not the others would silently narrow that workflow's
  coverage the same way the pre-round-1 route-list duplication did.
- **`PROJECT_STATUS.md` and `PROJECT_ROADMAP.md` are still empty
  scaffolds** — round 1's decisions (the mini-vla version-mismatch guard,
  the `.env` gitignore fix, the `vla-debug` removal, the proposed new
  review dimension) were never promoted into `PROJECT_STATUS.md`'s "Key
  decisions" list. Not this round's job to backfill, but worth flagging to
  the owner if it's still true when this round ends.

This repo is not the toolkit — no conditional dimensions apply (no
`bin/vibe`, `install.sh`, `docs/skill-quality.md` anchors here); extend
scope from `CLAUDE.md` and this repo's own docs, never from the toolkit's
own review anatomy.

## Next action

1. `git rebase origin/main` (never `vibe resume` — it fast-forwards to this
   branch's own upstream and reports up to date while arbitrarily far
   behind the default branch; this branch was just cut from a
   freshly-fetched `main`, so this should be a no-op, but confirm it).
2. Run the repo's own verification gate as the baseline: `npm run
   typecheck`, `npm run lint`, `npm test`, `npx next build`. Baseline at
   staging time: all four green (round 1 left them that way). A failure
   here is yours to fix or flag first — treat whatever this gate does
   *not* catch as the round's real target.
3. Fan out parallel sub-reviewers over the scope above, then run the
   `codebase-health` scan leg strictly after (never before — it anchors
   the reviewer to what the tool already knows), then merge into one
   ranked findings list — `file:line` and a concrete failure scenario
   each, reasoned-but-unproven claims labelled as such. Fix what's
   reasonable to fix in the same round; report the rest.

## Blockers

None known. Node.js was missing entirely from the container round 1 ran
in (not even via nvm) and had to be installed by hand from a tarball before
anything in CLAUDE.md's Commands section would run — check early whether
this round's environment has the same gap, so it doesn't eat time
mid-round.

## Gotchas (unpromoted)

- **This repo's CI is not label-gated** — `ci.yml` triggers unconditionally
  on `pull_request`/`push`, confirmed by reading it directly. Round 1's own
  `HANDOFF.md` wrongly assumed a label was needed; don't repeat that
  assumption — no label is needed on this round's review PR either.
- **The staging tool (`review.sh`) can't detect a squash-merged round as
  done.** Round 1's PR was squash-merged (GitHub default), so the merged
  commit on `main` shares no history with `fresh-review-1` and its subject
  doesn't contain the branch name — `review.sh create`'s ancestor-and-
  regex-fallback check couldn't tell round 1 was finished and re-adopted
  the old branch instead of computing round 2. This branch was staged by
  hand (same `stage_worktree`/template-render mechanics, just with the
  round number set correctly) after confirming via `gh pr view 57` that it
  was genuinely merged. Not this round's job to fix, but worth naming as a
  toolkit-level gap: `review.sh`'s merge check should also try a `gh pr
  list --state merged` search scoped to the round's own head branch, not
  just branch ancestry.
