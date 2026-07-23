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
- **Last updated:** 2026-07-23 · server (srv1841294)

## State

Review round 2 is done. Fanned out 8 blind parallel sub-reviewers (same
generic-core dimension set as round 1) + a scan-only `codebase-health` leg,
merged into one ranked list, de-duplicated against
`docs/review-round-1-findings.md` (nothing overlapped — round 1's 22
findings all stayed fixed). Fixed and verified 11 of ~17 substantiated
findings; 6 reported with reasoning for why they weren't fixed this round.
Full detail: `docs/review-round-2-findings.md`. `PROJECT_STATUS.md` was
backfilled (was still an empty scaffold, including round 1's own
never-promoted decisions — flagged by round 1's own handoff).

Baseline gate was green going in and stayed green: `typecheck`, `lint`,
`test`, `next build` all pass, plus a full `next e2e:build && npm run e2e`
run — 59 passed, 5 skipped as designed, across desktop/mobile/webkit-mobile.
The two behavioral fixes were each verified by deliberately reverting and
confirming the new test fails, then restoring and confirming it passes,
against the real Workers bundle.

12 commits on this branch, `814cb7e..HEAD` (round 1's merge commit to
current). **Nothing pushed yet** — this branch only exists locally in this
worktree; `origin` still points at round 1's merged state.

## Next action

Ask the owner whether to push this branch and open a PR (mirroring round
1's PR #57) — push/PR creation wasn't pre-authorized for this round, so it
wasn't done automatically. If yes: `git push -u origin fresh-review-2`,
then `gh pr create` with a body summarizing `docs/review-round-2-findings.md`
(same shape as round 1's PR).

Two things worth the owner's attention regardless of whether this becomes a
PR now:
- **Open question in `PROJECT_STATUS.md`**: should
  `bump-mini-vla.yml`'s version-consistency check move from warn-only to
  build-blocking? Left as-is this round (matches round 1's own design
  intent), only the misleading PR-body wording was fixed.
- **Two reported-not-fixed items involve owner-level tradeoffs**, not
  mechanical fixes: `nightly-e2e-full.yml` has no real failure notification
  (needs an `issues: write` permission grant + dedup logic), and
  `hero-full.spec.ts`'s 35-min timeout is unverified on the `mobile`/
  `webkit-mobile` Playwright projects it also runs on (needs either
  narrowing coverage or actually re-measuring). See
  `docs/review-round-2-findings.md` #13–14 for full reasoning.

## Blockers

None. Node.js (v22.17.0, matches `.nvmrc`) was present this time — no
install-from-tarball needed, unlike round 1.

## Gotchas (unpromoted)

- **A 1-second margin before a 20-second virtualized watchdog deadline is
  too tight for real-browser e2e timing jitter.** Building
  `hero-watchdog.spec.ts`'s new pause/resume-race test, a `page.clock`
  fast-forward to 19s (1s before `TRAIN_STALL_MS`) was flaky — roughly 1-in-3
  runs; the stall fired a couple seconds "early" relative to nominal,
  apparently from compounding microtask/rAF timing in a real headless
  browser. Widened all the margins in that test to 8-10s+ and it now passes
  reliably (5/5, then 3/3 repeated). If a future watchdog test feels flaky
  near its nominal deadline, this is the likely cause — don't shave the
  margin thin to save a few seconds of test runtime.
- **A scripted worker-double's `postMessage` ack timing matters, not just
  its content.** The same new test initially used a *synchronous* ack for
  `{t: "resume"}` (for test determinism) — but that made the ack land
  *before* `resumeTraining()` had set its own `statusRef.current`, routing
  `onUpdate` through the transition branch (which unconditionally re-arms
  the watchdog) instead of the steady-state "Resume re-arm" branch actually
  under test — silently defeating the test's own purpose (it passed even
  with the bug reverted). Real workers only ever ack asynchronously; matched
  that with a microtask, same as the existing "start" ack, and the test
  then correctly failed/passed with the bug reverted/fixed.
