# Review round 1 — findings (2026-07-23)

Independent, blind review of the whole repo: 7 parallel sub-reviewers, one
per dimension, followed by a `codebase-health` scan-only leg, merged into one
ranked list. `PROJECT_STATUS.md`/`PROJECT_ROADMAP.md` were empty scaffolds
and a repo-wide TODO/FIXME grep found nothing pre-existing, so every item
below is new. Baseline gate (`typecheck`, `lint`, `test`, `next build`) was
green going in.

## Status: all 22 findings + the process correction addressed

Every finding below was fixed, tested, or (for #7 and #14) investigated and
found not to need a code change — see the commit history on this branch for
the verification each fix went through (regenerating real content with a
live token, running the actual Playwright suite against a real
OpenNext/wrangler bundle, etc.). One follow-up remains outside this repo's
scope: adding *silent partial-failure in build-time generation loops* (see
bottom of this doc) to the codebase-review skill's dimension catalog, which
lives in the separate agentic-dev-toolkit repo.

## Confirmed live bugs

1. **`vla-rl-support` project silently missing from the site.**
   `config/projects.sources.json:9` added the entry on 2026-07-13, but
   `lib/projects-data.json` has zero entries for it (verified:
   `grep -c vla-rl-support lib/projects-data.json` → `0`). Cause:
   `scripts/gen-projects-data.mjs`'s per-project `try/catch` logs a
   `console.warn` and moves on when a project has never successfully built —
   since it's never previously succeeded, there's no stale entry to keep, so
   nothing is written. CI never runs this script (`next build` reads only the
   committed JSON), so there is no signal anywhere that a configured project
   isn't shipping.
   **Fixed:** regenerated `lib/projects-data.json` with a real token
   (picked up `vla-rl-support` and also `portfolio-site`, which turned out
   to have the same problem); added a unit test asserting every configured
   slug has a data entry, verified it catches the regression.

2. **`mini-vla` asset path is serving under the wrong version.**
   `package.json` pins `mini-vla@github:lukasmueller-dev/mini-vla#v0.7.1`, but
   the locked package's own `package.json` reports `version: "0.7.0"`
   (verified: `node -e "console.log(require('mini-vla/package.json').version)"`
   → `0.7.0`). `lib/vla-assets.ts` and `scripts/copy-vla-assets.mjs` both
   derive `VLA_ASSET_BASE` from that `version` field, so v0.7.1's actual
   asset payload is served/cached under `/vla/0.7.0/` — the exact
   silent-drift failure mode the versioned-path scheme was built to prevent.
   `tests/unit/vla-assets.test.ts`'s own comment claims mini-vla's release
   script "refuses to tag when package.json's version disagrees with the
   tag" — this already-merged bump (commit `264dc96`) disproves that.
   **Fixed:** verified the current mismatch is harmless (the `assets/` tree
   hash is identical between the v0.7.0 and v0.7.1 tags — confirmed via the
   GitHub API), and added a version-consistency check to
   `bump-mini-vla.yml` so a future mismatch warns loudly at bump time
   instead of shipping silently. The root cause (mini-vla's own release
   process not actually bumping `package.json`'s version) lives in a
   separate repo, out of this PR's scope.

3. **`.env` is a supported secret source but not gitignored.**
   `scripts/gen-resume-source.mjs`, `scripts/gen-projects-data.mjs`, and
   `scripts/gen-blog-data.mjs` all check `.dev.vars`, `.env.local`, **and
   `.env`** for `GITHUB_TOKEN=`. `.gitignore` only ignores `.env*.local`
   (verified: `git check-ignore -v .env` exits 1 — not ignored;
   `.env.local` exits 0 — ignored). A developer following the common
   Next.js convention of using `.env` instead of `.dev.vars`, followed by an
   ordinary `git add -A`, would commit a live PAT with read access to
   private content repos.
   **Fixed:** added `.env` to `.gitignore`; verified with
   `git check-ignore -v .env`.

## High-impact gaps

4. **CI never exercises the hero's actual training-to-convergence path.**
   `tests/e2e/hero-full.spec.ts:21` — the only spec that waits for
   convergence and asserts command-decoding accuracy — opens with
   `test.skip(!process.env.VLA_FULL, ...)`. `.github/workflows/ci.yml`'s
   `e2e` job runs plain `npm run e2e`, never setting `VLA_FULL`. The site's
   centerpiece demo is untested by default CI; only `npm run e2e:full`
   (manual) or a real deploy would catch a regression here.
   **Fixed:** added `.github/workflows/nightly-e2e-full.yml`, a daily
   cron running `e2e:full` against `main` — keeps regular push/PR CI fast
   while surfacing a regression within 24h instead of never.

5. **Script-injection shape in `bump-mini-vla.yml`.**
   Lines 46 and 62 interpolate `${{ github.event.client_payload.tag ||
   github.event.inputs.tag }}` directly into shell `run:` text. A tag value
   like `v1'; <cmd>; '` breaks out of the quotes before the `case "$TAG" in
   v[0-9]*)` glob check ever runs (and that glob check wouldn't catch
   embedded `;`/`` ` ``/`$()` anyway). The job holds `contents: write` +
   `pull-requests: write`. Reachable via `repository_dispatch`
   (`client_payload.tag`, gated behind a PAT in the sibling `mini-vla` repo)
   or `workflow_dispatch` (requires a collaborator) — not by an anonymous
   PR, but a real injection primitive sitting one secret away from a trust
   boundary. Fix: move the value through `env:` and reference `"$TAG"`
   inside the script instead of interpolating the expression into shell text.
   **Fixed** as described; verified locally that an injection payload
   which still passes the glob check no longer executes as a command once
   it's a real shell variable instead of interpolated text.

6. **`bump-mini-vla.yml` never runs e2e before merging a version bump.**
   Its own pre-merge check is `typecheck` → `npm test` → `npx next build`
   (lines 68-75) — no Playwright, no real-browser exercise of the
   trainer/rollout contract `mini-vla` actually drives. The unit test that
   does run (`vla-assets.test.ts`) only checks that asset files exist and
   that `VLA_VERSION`/`VLA_ASSET_BASE` agree with each other — not that they
   agree with the *tag* being bumped to, which is exactly how finding #2
   happened. Because the bot-authored PR uses `GITHUB_TOKEN`, it doesn't
   trigger `ci.yml`'s own `pull_request` job either, so this really is the
   PR's only gate.
   **Fixed:** added lint + a real `e2e:build`/`e2e:full` run (Playwright
   against `wrangler dev`, including the slow convergence specs) to
   `bump-mini-vla.yml` before it opens the bump PR.

## Reasoned, not fully proven

7. **`replayFallback: true` may make the documented Reload path unreachable.**
   `components/Hero.tsx:1506-1518` always constructs `VLATrainer({
   replayFallback: true, ... })`. Tracing `node_modules/mini-vla/js/src/
   trainer.ts:278-284`, `handleWorkerFailure()` checks `replayFallback`
   first and calls `triggerReplay()` instead of ever setting `errorReason =
   "worker"` — the only branch that reaches the Reload button
   (`Hero.tsx:2206`). If true, a stale-tab worker-chunk 404 (the exact
   scenario CLAUDE.md documents as surfacing "Reload") would instead
   silently substitute the CPU-backend canned replay animation, with no
   error surfaced at all. Traced from dependency source, not reproduced in
   a live browser.
   **Investigated, not a bug:** `tests/e2e/hero-error.spec.ts` already tests
   and documents exactly this as deliberate, shipped behavior — a dead
   worker chunk IS meant to be rescued by the replay, not show Reload. The
   actual gap was that CLAUDE.md/README still described the old (pre-replay-
   fallback) behavior; fixed as part of #9 below.

## Cross-file consistency / doc drift

8. **`public/build-id.json`'s shape and path are duplicated three times**
   with no shared constant, type, or test: `next.config.mjs:6-9` (writer,
   `{ id: BUILD_ID }`), `components/Hero.tsx:1691-1699` (hardcoded
   `"/build-id.json"`, destructures `{ id }`, wrapped in a `try/catch` that
   swallows failures), and `public/_headers` (a separate hardcoded cache
   rule for the same path). Renaming the key or moving the file while
   touching only one of these three silently breaks stale-tab detection or
   the cache rule, with no error anywhere.
   **Fixed:** added `lib/build-id.ts` as the shared constant/type;
   `Hero.tsx` imports it directly, `next.config.mjs` can't (it's loaded by
   Next's own config loader, which can't type-strip `.ts` — confirmed by
   actually trying it), so it keeps a literal with a test
   (`tests/unit/build-id.test.ts`) guarding it and `public/_headers`
   against drift.

9. **`CLAUDE.md` and `README.md` both reference a `loadFailed` symbol in
   `Hero.tsx` that doesn't exist** (verified: repo-wide grep for
   `loadFailed` returns nothing outside the two doc lines). The actual
   mechanism is `errorReason` (`Hero.tsx:1439`, checked at `2206`) and a
   separate `hostFailure` branch (`2183`). Stale rename — update both docs
   to name `errorReason`/`hostFailure`.
   **Fixed:** rewrote both doc passages to describe the actual (and
   correctly tested) replay-fallback behavior, not just the symbol name.

10. **Three files call the project content file `portfolio.md`; it's
    actually `README.md`.** `lib/project-md.ts:1`, `lib/richtext.tsx:8`, and
    `lib/content.ts:75` all say `portfolio.md`, but
    `config/projects.sources.json:3` sets `"file": "README.md"` and
    `scripts/gen-projects-data.mjs:9-14` documents reading
    `<contentDir>/README.md`. Looks like a stale name from before the
    content file was renamed.
    **Fixed:** all three corrected to `README.md`.

## Test blind spots

11. **Watchdog `HostFailure` paths have zero test coverage.** None of
    `train-stalled`, `train-collapsed`, or `load-stuck`
    (`components/Hero.tsx:1308-1394`) appear anywhere under `tests/` except
    as an escape hatch in `tests/e2e/helpers.ts:55,69` (a `test.skip` for a
    *different* stall condition, not an assertion this watchdog fires
    correctly). A regression to `TRAIN_STALL_MS` or `releaseWorkerToIdle()`
    (e.g. a leaked WebGL context) ships undetected.
    **Fixed:** added `tests/e2e/hero-watchdog.spec.ts`, covering all three
    watchdogs via `page.clock` fast-forwarding plus a scripted
    `window.Worker` double that speaks mini-vla's real wire protocol (for
    the two training-loop watchdogs) and real hung `page.route` fetches
    (for load-stuck). Verified 3x across desktop+mobile with zero flakes,
    and each test independently confirmed to fail for the right reason by
    temporarily breaking the corresponding production code and reverting.

12. **`lib/posts-data.json` is currently `[]`**, so the `...posts.map(...)`
    spreads in `tests/e2e/routes.spec.ts:26` and `caching.spec.ts:34`
    generate zero test cases for `/blog/[slug]`. A broken blog-post page
    would ship undetected until a post is actually published — the loop
    silently produces no assertions instead of failing.
    **Fixed:** added `tests/unit/blog-slug-page.test.tsx`, testing the page's
    `generateStaticParams`/`generateMetadata`/render/notFound logic against
    a synthetic fixture, independent of real content. Also added a `"@"`
    path alias to `vitest.config.ts` (mirroring `tsconfig.json`'s) so tests
    can import `app/`/component modules directly.

13. **WebKit/iOS is never actually run.** `playwright.config.ts:45-61`
    defines no WebKit engine, and `ci.yml` installs only `--with-deps
    chromium`. The CPU-backend replay fallback is documented (`Hero.tsx`
    comments) as "the iOS/iPadOS path," but it's only exercised via a
    forced worker-chunk abort on Chromium/SwiftShader
    (`tests/e2e/hero-error.spec.ts:52-86`) — never a genuine WebKit run
    where it would trigger naturally.
    **Fixed:** added a `webkit-mobile` Playwright project (real WebKit via
    the `iPhone 13` device preset) matching `hero*.spec.ts`; installed
    webkit alongside chromium in every workflow that runs e2e. Verified
    locally against the real bundle: `hero.spec.ts` and
    `hero-error.spec.ts` (including the replay-fallback rescue) both pass
    on webkit-mobile.

14. **`tests/unit/resume.test.ts:84-108`**'s `describe.runIf(existsSync(texPath))`
    skips the "understands the live template" test whenever
    `public/resume.tex` isn't staged — which is exactly CI's `check` lane
    (tokenless). Real-template drift is only caught at deploy time.
    **Investigated, not a bug:** this is already correctly designed and
    documented — the comment directly above it explains the tradeoff, and
    a compensating fixture-based suite covers the tokenless case. "Fixing"
    it would mean adding a secret to CI's deliberately-tokenless `check`
    lane, which contradicts a design pillar CLAUDE.md documents elsewhere.
    No change made.

## Silent partial-failure in build-time generation loops

(See "New review dimension" below — findings #1, 15, 16, 17 are all
instances of this pattern.)

15. **`gen-projects-data.mjs`'s `buildOne()` discards a good README fetch
    because of an unrelated, earlier metadata failure.** Repo metadata
    (`meta`) is fetched *before* the README body; if that call throws (rate
    limit, renamed/deleted repo, transient 5xx), the function throws before
    ever reading the README, so `main()` falls back to the stale committed
    entry — even though metadata is documented (file header, lines 21-24)
    as only a fallback for title/blurb/tags. A README edit pushed while
    GitHub's metadata endpoint is rate-limited silently doesn't ship.
    **Fixed:** fetch metadata in its own try/catch, falling back to `null`
    (with a warning) on failure, so it can never block the README fetch.

16. **`Project.title` is typed as required `string` but the generator can
    emit `undefined`.** `scripts/gen-projects-data.mjs:166`:
    `title: fmParsed.title || meta?.name`. Two config entries
    (`vla-rl-support`, `cavitation-experiment-database`) omit `repo`, so
    `meta` is `null`; if that project's frontmatter also omits `title:`,
    the field is `undefined` and `JSON.stringify` drops the key entirely —
    `lib/content.ts:76`'s `as Project[]` cast gives no runtime check.
    Symptom would be a page `<title>` of `"undefined · Lukas Müller"` and an
    empty `ProjectCard` title. Dormant today (both repo-less projects
    currently have frontmatter titles).
    **Fixed:** fall back to the slug when both frontmatter title and meta
    name are missing, with a warning. Verified the fallback chain logic
    directly and re-ran the generator (byte-identical output today).

17. **`gen-resume-source.mjs` contradicts its own "no fallback" comment.**
    The header comment and `build-resume.sh` both say both files are
    required and a failure fails the build with no fallback — but the loop
    (lines 62-68) does `if (!name) continue`, silently skipping a missing
    `tex`/`pdf` key in `config/resume.source.json` rather than erroring.
    Dormant today (both keys present in the config).
    **Fixed:** now throws (caught by the existing `main().catch`, which
    already exits 1). Verified both paths live: the happy path still
    fetches both real files correctly, and a config missing `"pdf"` now
    exits 1 with a clear message instead of continuing silently.

## Lower severity

18. **`claude-review.yml` triggers for any commenter's `"@claude"`**, with
    no `author_association` check — on a public repo, any account can
    trigger an LLM with `pull-requests: write` and repo-read access via a
    PR comment. Blast radius is capped (no write beyond PR comments,
    `contents: read` only), but it's a real confused-deputy /
    prompt-injection surface. Fix: gate on
    `github.event.comment.user.login` / `author_association` in
    `OWNER`/`MEMBER`/`COLLABORATOR`.
    **Fixed** as described.

19. **Duplicated route/slug-list boilerplate** between
    `tests/e2e/caching.spec.ts:3-28` and `tests/e2e/routes.spec.ts:5-19`
    (found by `jscpd`, 26 duplicated lines) — the same `projects`/`posts`
    JSON reads and route-list construction, copy-pasted. A route added to
    one list and not the other silently narrows that file's coverage.
    **Fixed:** extracted `sitePageRoutes()` into `tests/e2e/helpers.ts`.
    Verified all 37 tests across both files still pass against the real
    bundle; `jscpd` now finds zero clones.

20. **`ProjectLink` (`lib/content.ts:9-12`) and `ProjectMdLink`
    (`lib/project-md.ts:34-37`) are structurally identical types**
    (`{ label: string; href: string }`), defined independently. Adding a
    field to one without the other compiles fine and silently drops the
    field wherever the mismatched type is used.
    **Fixed:** `lib/project-md.ts` now imports `ProjectLink` as a
    type-only import from `lib/content.ts` instead of redefining it.

21. **`ci.yml` has no explicit `permissions:` block** (relies on the repo
    default — no exploit found, just least-privilege hygiene).
    **`app/vla-debug/` is a leftover debugging route** shipped to
    production per its own header comment ("delete this route once the
    cause is pinned").
    **Fixed:** added `contents: read` to `ci.yml`. Confirmed with the site
    owner that the debug route's cause was pinned (mini-vla's
    `replayFallback` is the shipped fix) and removed it — verified no
    other file referenced it, and the full build + e2e route sweep still
    pass without it.

22. **Slug-derivation logic is duplicated** between `main()`
    (`gen-projects-data.mjs:193`, for log/catch messages) and `buildOne()`
    (lines 105-106, for actual use) — both compute
    `source.slug ?? source.repo?.split("/")[1]` independently. Not yet
    divergent, but an edit to one without the other would make logged
    output and actual output silently disagree.
    **Fixed:** extracted `deriveSlug()`, used by both. Re-ran the generator
    live and got byte-identical output.

## Process correction (not a code finding)

This round's own `HANDOFF.md` assumed CI here is opt-in per PR via a label.
That's false: `.github/workflows/ci.yml` triggers unconditionally on
`pull_request`/`push`, no label anywhere in this repo. No label was needed
on this PR.

## New review dimension

**Silent partial-failure in build-time generation loops**: whether a
per-item failure in a commit-time data-generation loop (content fetch, asset
copy, config parsing) fails the build, warns-and-skips, or silently produces
wrong/missing output with no signal anywhere. Findings #1, #15, #16, #17 are
all instances of this and none of the existing dimensions in the
codebase-review skill's catalog names it directly — worth adding.
