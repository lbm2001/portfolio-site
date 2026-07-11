---
name: commit-push-pr
description: Autonomously stage, commit, push, and open a GitHub PR. Use when the user says "commit and push", "open a PR", "ship this", "create a pull request", or similar, after code changes are ready. Handles Conventional Commits messages, pushes the branch, and opens a PR via GitHub CLI with an auto-generated title and body. Do NOT use for merging PRs, force-pushing to shared branches, or committing secrets.
---

# Commit, Push & PR

Autonomous flow: stage → commit (Conventional Commits) → push → open PR via `gh`. Run end-to-end without asking for confirmation unless a **Stop condition** below is hit.

## Prerequisites (check once, fail fast)

```bash
git rev-parse --is-inside-work-tree   # must be a repo
gh auth status                        # gh must be authenticated
```

If `gh` is missing/unauthenticated, tell the user and stop.

## Flow

### 1. Assess

```bash
git status --porcelain
git branch --show-current
git diff --stat HEAD
```

- If there are no changes (staged or unstaged), stop and say so.
- Read the actual diff (`git diff HEAD`) to understand what changed — the commit message must reflect real changes, not a guess.

### 2. Branch

Never commit directly to `main`/`master`. If on a protected/default branch, create a topic branch:

```bash
git checkout -b <type>/<short-slug>
```

Derive `<type>` from the change (feat, fix, chore, docs, refactor…) and `<slug>` from the primary change (e.g. `fix/oauth-token-refresh`). If already on a topic branch, stay on it.

### 3. Commit

```bash
git add -A
git commit -m "<type>[optional scope]: <description>" -m "<body>"
```

Conventional Commits rules:
- Types: `feat`, `fix`, `chore`, `docs`, `style`, `refactor`, `perf`, `test`, `build`, `ci`.
- Subject ≤ 72 chars, imperative mood, no trailing period.
- Add `!` after type/scope for breaking changes and a `BREAKING CHANGE:` footer.
- Body (optional) explains *why*, wrapped at ~72 cols.
- If changes are logically distinct, make multiple commits by staging paths selectively (`git add <paths>`).

### 4. Push

```bash
git push -u origin HEAD
```

Only ever push the current topic branch. Never `--force` / `--force-with-lease` to a shared branch.

### 5. Open PR

```bash
gh pr create --fill --base <default-branch>
```

- `--fill` auto-generates title/body from the commits. Resolve `<default-branch>` via `gh repo view --json defaultBranchRef -q .defaultBranchRef.name`.
- If commits are messy or a single squashed narrative is clearer, replace `--fill` with explicit `--title`/`--body` synthesized from the diff (title = Conventional Commit summary; body = **Summary** / **Changes** / **Test plan** sections).
- If a PR already exists for the branch, skip creation and report its URL (`gh pr view --json url -q .url`).

Return the PR URL as the final output.

## Stop conditions (pause and ask the user)

- Diff contains apparent secrets/credentials (API keys, tokens, `.env` values, private keys).
- Working tree has merge-conflict markers or unresolved conflicts.
- Would require force-pushing a shared/protected branch.
- No remote named `origin`, or push is rejected (diverged history).
- Repo default branch is protected in a way that blocks PR creation.

## Notes

- Respect `.gitignore`; if large/binary artifacts are about to be staged, flag before `git add -A`.
- Don't amend or rewrite already-pushed commits.
- Keep output terse: report branch, commit subject(s), and the PR URL.