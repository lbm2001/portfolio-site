# Review round 2 — findings (2026-07-23)

Second independent, blind review of the whole repo, reviewing the exact tree
round 1 left behind (`814cb7e`, merged via PR #57): 8 parallel sub-reviewers
(adversarial static read, docs-vs-code drift, test blind spots,
config/permission safety, CI correctness, cross-file consistency,
derived-but-unused values, guard blind spots), followed by a scan-only
`codebase-health` leg, merged into one ranked list. Sub-reviewers were kept
blind to round 1's findings (`docs/review-round-1-findings.md`) so their
results could be checked for overlap afterward — none of round 2's findings
duplicate round 1's; every item below is new. Baseline gate (`typecheck`,
`lint`, `test`, `next build`) was green going in and stayed green throughout.

## Status: 11 of ~17 substantiated findings fixed and verified; 6 reported only

Every fix below was verified against the real committed data/scripts and,
for the two behavioral bug fixes, against the actual OpenNext/Workers bundle
via Playwright (including a deliberate revert-and-recheck to confirm the new
test fails without the fix and passes with it). See the commit history on
this branch (`814cb7e..HEAD`) for each change.

## Confirmed live bugs

1. **Pausing training just before the stall watchdog's deadline and resuming
   could falsely tear down a healthy run.** `components/Hero.tsx`'s
   `pauseTraining()` never cleared `trainWatchdogRef` — its deadline was
   computed from the last batch *before* the pause. A pause landing close
   enough to `TRAIN_STALL_MS` (20s), followed by a resume before the stale
   timer fired, left it pending through the resume; it then fired against
   the now-resumed `"training"` status, releasing the worker and surfacing
   "Training stalled — reload" on a run that never actually stalled. The
   existing "long pause already fired" re-arm path (`resumeTraining`'s
   entry-arm, guarded by `trainWatchdogRef.current === null`) only handled
   the case where the stale timer had already fired during the pause, not
   one still pending at resume time.
   **Fixed:** `pauseTraining()` now clears the watchdog unconditionally on
   pause, making the existing null-check re-arm correct for a short pause
   too. Added a regression test (`tests/e2e/hero-watchdog.spec.ts`, scripted
   worker double + `page.clock`) covering exactly this sequence, plus
   confirming the watchdog still fires on a genuinely stalled resume.
   Verified the test fails deterministically (3/3 runs) with the fix
   reverted and passes reliably (3/3) with it restored, against the real
   Workers bundle.
   **Decisions/risks:** this is the site's centerpiece demo, and this exact
   race had no prior test coverage and no live user report — flagging as
   reviewed-but-worth-a-second-look rather than fully closed.

2. **`gen-blog-data.mjs` could silently wipe every committed post.**
   `ghList()` maps both "the configured `dir` doesn't exist" and "the token
   can't see this private repo" (GitHub's Contents API returns 404, not 403,
   for an unauthorized private repo) to the same empty-listing result. Since
   blog slugs (unlike the static projects config) are *derived from a live
   directory listing*, either case was silently treated as "a legitimately
   empty blog" and would overwrite a non-empty `lib/posts-data.json` with
   `[]` — with no warning distinguishing a real content change from a
   misconfiguration or a narrowed token. `gen-projects-data.mjs` already
   guards its own equivalent case (round 1, finding #1/#15); this script had
   no analog. Found independently by three sub-reviewers (adversarial static
   read, test blind spots, guard blind spots) — the strongest three-way
   corroboration of the round.
   **Fixed:** added a guard for the specific "had posts, now has none"
   transition only (`slugs.length === 0 && existing.length > 0`), so a
   genuinely new/empty blog still writes `[]` as designed, while a
   regression from a full blog to zero posts is treated as suspicious and
   leaves the file untouched instead.

## High-impact gaps

3. **`bump-mini-vla.yml`'s version-consistency check can't actually block a
   merge, and the PR body implied otherwise.** The check (added in round 1
   to catch a repeat of finding #2) only ever emits a `::warning::`
   annotation and always exits 0 — yet the PR body listed it among "the
   gates below all passed here," alongside real pass/fail gates like
   typecheck and the e2e suite. A reviewer skimming green checks could merge
   past a real version mismatch without ever seeing the warning. Found
   independently by two sub-reviewers (adversarial static read, CI
   correctness).
   **Fixed the misleading wording**, not the underlying pass-fail semantics:
   round 1 deliberately made this check non-blocking, since a mismatch can
   be harmless (their own investigation proved exactly that for the
   v0.7.1/0.7.0 case) — hardening it to a hard failure is a real process
   decision (it would block a future legitimate-but-mismatched bump on a
   manual override), not a bug fix, so it's left as an open question below
   rather than changed unilaterally.
   **Also fixed:** the tag validator accepted anything shaped `v<digit>*`
   via a loose shell glob; tightened to a strict semver regex. Not currently
   exploitable (the value only reaches shell text through `env:`
   indirection), but a cheap defense-in-depth improvement while touching the
   file.

4. **`tests/unit/build-id.test.ts` never checked the JSON payload shape**,
   only the file path string. Verified directly: renaming `next.config.mjs`'s
   written key from `id` to `buildId` still passed the full suite. In
   production that would make `components/Hero.tsx`'s `const { id } = await
   r.json()` always read `undefined`, forcing a reload on every
   `visibilitychange` — a live reload-loop regression the test suite is
   supposed to guard against but didn't.
   **Fixed:** added an assertion pinning the payload key; confirmed it fails
   on the renamed-key case and passes once reverted.
   **Related, separately fixed:** `components/Hero.tsx`'s own runtime guard
   around this fetch had the same gap one level down — a malformed-but-200
   response (parses fine, but `id` isn't a string) would compare
   `undefined !== buildId` as true and force a reload, instead of being
   treated the same as the network/parse failures already handled next to
   it. Added a `typeof id === "string"` check before the comparison.

## Cross-file consistency / doc drift

5. **Three independent hand-written lists of the same top-level page
   routes**, nothing tying them together: `next.config.mjs`'s `pageRoutes`
   (drives the must-revalidate `Cache-Control` rule central to the
   stale-chunk reload path), `lib/content.ts`'s `nav`, and
   `tests/e2e/helpers.ts`'s `sitePageRoutes()` (drives `routes.spec.ts` /
   `caching.spec.ts`'s coverage — those two had already drifted apart once
   before being unified into `sitePageRoutes()`, per that function's own
   comment). A route added to `nav` or `sitePageRoutes()` but forgotten in
   `next.config.mjs` would silently lose the revalidate rule, and
   `caching.spec.ts` wouldn't catch it either, since it sweeps
   `sitePageRoutes()`'s list, not `next.config.mjs`'s.
   **Fixed:** added `tests/unit/route-list.test.ts`, pinning all three
   (same text-based approach as `build-id.test.ts`, for the same reason —
   neither `next.config.mjs` nor a Playwright-importing helper file can be
   safely imported into a vitest unit test). Verified it fails when a route
   is removed from `next.config.mjs` and passes once restored.

6. **`lib/build-id.ts`'s own docstring and `public/_headers`'s comment both
   wrongly claimed `next.config.mjs` imports the shared `BUILD_ID_PATH`
   constant.** It doesn't — `next.config.mjs`'s own adjacent comment
   explains it can't (Next's config loader can't type-strip `.ts`) and keeps
   a separate literal, exactly like `public/_headers`. Only
   `components/Hero.tsx` actually imports the module.
   **Fixed:** corrected both comments.

7. **`tests/e2e/helpers.ts` had a stale value in a comment**: said the
   loading watchdog "gives up on 'loading' after 10s," but
   `LOADING_WATCHDOG_MS` (`components/Hero.tsx`) has been 15s since round
   1's replay-fallback work — 10s is actually `IDLE_TEARDOWN_GRACE_MS`, a
   different watchdog entirely.
   **Fixed.**

8. **`tests/unit/content-data.test.ts` reimplemented `gen-projects-data.mjs`'s
   slug-derivation formula inline** instead of importing it — a change to
   the real derivation could silently leave the test checking the old
   formula. Couldn't import `gen-projects-data.mjs` directly (it runs
   `main()`, including real fetches, as a side effect of being loaded).
   **Fixed:** extracted `deriveSlug()` into `scripts/lib/slug.mjs`, a
   side-effect-free module both the script and the test import.

## Duplication (codebase-health leg)

9. **`loadToken()` was byte-identical across all three generator scripts**
   (`gen-blog-data.mjs`, `gen-projects-data.mjs`, `gen-resume-source.mjs`) —
   one comment even said "Mirrors gen-projects-data.mjs."
   **Fixed:** extracted to `scripts/lib/github-token.mjs`. Verified all
   three scripts still run correctly (no-token path warns-and-exits-0 for
   projects/blog, exits 1 for the required resume fetch).

10. **`lib/post-md.ts` and `lib/project-md.ts` each hand-rolled the same
    "key: value" line regex, blank/comment-skip check, and `aiAssisted`
    boolean parse.**
    **Fixed:** moved into `lib/frontmatter.ts` (already the shared home for
    `splitFrontmatter`/`stripQuotes`) as `parseKvLine()`/`parseAiAssisted()`;
    each parser keeps its own key handling and `project-md.ts`'s
    index-based `tags`/`links` lookahead untouched. Verified against both
    parsers' existing unit tests.

## Lower severity

11. **`gen-projects-data.mjs` computed `readExisting()`/`byslug` (a file
    read + parse) unconditionally, then discarded them whenever
    `GITHUB_TOKEN` was missing** — the very next branch returns without
    touching either. `gen-blog-data.mjs` gets this right (computes them
    *after* its own no-token check). Cheap, not correctness-affecting, but a
    genuine dead derivation, and the asymmetry between two near-identical
    scripts suggested this was simply missed when the pattern was copied.
    **Fixed:** moved the two lines below the `!TOKEN` check.

12. **`lib/content.ts` exported `NavLink`, which nothing outside the file
    imports by name** — only the `nav` value (whose type is inferred at its
    own declaration) is consumed elsewhere.
    **Fixed:** stopped exporting it.

## Reasoned, not fixed — reported for the owner

13. **`nightly-e2e-full.yml`'s only failure signal is a Playwright HTML
    artifact upload**, visible only to someone who manually opens that
    specific Actions run. No issue-creation or notification step exists in
    any of the four workflows. This undercuts the workflow's own header
    comment (and CLAUDE.md's claim) that a regression "surfaces within a day
    rather than never" — the actual backstop is GitHub's default
    scheduled-workflow-failure email, sent only to whoever last edited the
    file, which is easy to miss or mute. Not fixed: adding a real
    notification (e.g., issue-on-failure) means granting `issues: write` to
    a workflow currently scoped to `contents: read` only, plus dedup logic
    to avoid spamming a new issue every night — a permissions/design call
    better made by the owner than assumed here.

14. **`hero-full.spec.ts`'s 35-minute timeout was measured only on
    desktop/Chromium+SwiftShader**, but its `testMatch` pattern also matches
    the `mobile` and `webkit-mobile` Playwright projects, and neither
    `npm run e2e:full` nor the workflows that call it pass a `--project`
    filter. `mobile` trains a different task (`MOBILE_RUN_CONFIG`) and
    `webkit-mobile` runs on real WebKit rather than SwiftShader — neither's
    convergence rate against this budget has been measured. Not fixed:
    doing so needs either removing real coverage (restricting to desktop
    only) or actually running the slow convergence suite on both other
    engines to re-measure — both are product/cost decisions, not a
    same-round mechanical fix.

15. **`lib/vla-assets.ts`'s `VLA_RUNTIME_ASSETS`/`VLA_REPLAY_MANIFEST`
    filename constants are a second, independent hardcoding of the same
    strings `mini-vla` itself hardcodes internally** (in
    `embeddings.ts`/`trainer.replay.ts`) — unlike `VLA_VERSION`, which
    genuinely derives from `mini-vla/package.json`, these have no shared
    source; they just happen to agree today. A `mini-vla` rename would
    silently 404 at first "Start Training" click. Not fixed: no clean single
    source of truth exists without `mini-vla` itself exporting these names,
    which is outside this repo's scope (same boundary as the pre-existing
    mini-vla version-drift check).

16. **`components/Hero.tsx`'s `DEAD_LOSS_LIMIT` steady-state collapse path
    (8 dead batches in a row without a status transition) has no test
    coverage** — `hero-watchdog.spec.ts`'s "collapse" mode jumps straight to
    a status *transition*, never exercising the steady-state branch that
    increments the dead-loss counter. Not fixed this round: scripting a
    worker double that posts 8 same-status dead-loss batches is
    straightforward but adds meaningful test runtime for a lower-probability
    regression path; noted for a future pass.

17. **`claude-review.yml` gates on who *wrote the @claude comment*, not who
    authored the PR/diff content the bot then reads** — a trusted
    collaborator commenting on an untrusted fork PR still hands the action
    attacker-controlled diff text (a prompt-injection surface). This is the
    standard trade-off for this class of review bot, already implicitly
    accepted by the repo's own design, not a new hole — noted for
    completeness, not a same-round fix.

## Verification

All fixes verified against the baseline gate (`typecheck`, `lint`, `test`,
`next build`, all green throughout) plus a full `next e2e:build && npm run
e2e` pass — 59 passed, 5 skipped as designed (the two `VLA_FULL`-gated
convergence specs × 3 projects, the mobile-only demo-close test on
desktop/webkit, and the tokenless resume-PDF test) — across all three
Playwright projects (desktop, mobile, webkit-mobile). The two behavioral
fixes (Hero.tsx watchdog race, build-id payload guard) were additionally
verified by deliberately reverting each and confirming its new test fails,
then restoring and confirming it passes.
