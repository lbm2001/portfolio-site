---
name: verify
description: Quick local check before pushing changes to this portfolio site — typecheck + unit tests. CI (.github/workflows/ci.yml) already covers lint, build, and the full e2e/hero suite on every push.
---

# Verifying this repo

Quick local gate before pushing — CI handles the rest (lint, `next build`,
worker e2e + hero alive-gate on desktop/mobile). No need to duplicate that
locally.

```
npx tsc --noEmit && npm test
```

- `npx tsc --noEmit` — strict typecheck, catches most integration breaks.
- `npm test` — Vitest units: lib/ parsers (post-md, project-md, resume,
  richtext) + committed lib/*-data.json invariants.

## If you touched the VLA hero (lib/vla, Hero.tsx) specifically
Run the real e2e suite locally instead of hand-driving a browser — it's the
same check CI runs:
```
npm run e2e:build && npm run e2e
```
Selectors/waits live in tests/e2e/helpers.ts if you need to write a new spec.

## After deploying
`npm run smoke:live` — same suite against https://lukasmueller.dev.
