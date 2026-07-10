# portfolio-site

My personal site — [lukasmueller.dev](https://lukasmueller.dev). Next.js (App Router)
on Cloudflare Workers via OpenNext.

The landing page runs a live Vision-Language-Action demo: a TensorFlow.js
behaviour-cloning policy that trains in the browser, in a Web Worker, against an
analytical-IK expert, and then takes typed commands. The model, rollout engine and
canvas renderers live in a separate package,
[mini-vla](https://github.com/lbm2001/mini-vla); this repo renders the encoder
panels around it.

## Requirements

- **Node ≥ 22.6** (see `.nvmrc`). The build scripts execute TypeScript directly
  through `--experimental-strip-types`, which lands in 22.6.
- A **GitHub token** for the build-time content fetch (below). Not needed for
  `next dev` or for a plain `next build`.

## Setup

```bash
nvm use            # or any Node >= 22.6
npm install
cp .dev.vars.example .dev.vars   # then fill in GITHUB_TOKEN
npm run dev
```

`.dev.vars` is gitignored. `GITHUB_TOKEN` should be a fine-grained personal access
token with read-only **Contents** + **Metadata**, scoped to the repos named in
`config/`. It is read only by the Node build scripts and never reaches the client.

Without a token, `npm run dev` still works — the project and blog fetches fall back
to the committed data (below) and log a warning.

## Content pipeline

Nothing is fetched at request time. Cloudflare Workers has no runtime filesystem, so
every external source is pulled at **build** time into a JSON module that the pages
import statically:

| Source (private repo)                    | Script                    | Output                   |
| ---------------------------------------- | ------------------------- | ------------------------ |
| `lbm2001/application-material` (résumé)  | `gen-resume-source.mjs` → `gen-resume-data.mjs` | `public/resume.{tex,pdf}`, `lib/resume-data.json` |
| `lbm2001/portfolio-project-content`      | `gen-projects-data.mjs`   | `lib/projects-data.json` |
| `lbm2001/blog`                           | `gen-blog-data.mjs`       | `lib/posts-data.json`    |
| `node_modules/mini-vla/assets`           | `copy-vla-assets.mjs`     | `public/vla/`            |

The three `lib/*-data.json` files are **committed** so that `next dev`, `tsc` and CI
build offline from last-known content. Which sources are read is configured in
`config/*.sources.json`.

The résumé fetch is **required** — it has no committed fallback, so a missing token
fails the build there rather than silently shipping a site with no CV. Projects and
blog are best-effort and keep the committed data on any failure.

## Scripts

| Command             | What it does                                                        |
| ------------------- | ------------------------------------------------------------------- |
| `npm run dev`       | Copies VLA assets, refreshes projects + blog, starts `next dev`.     |
| `npm run build`     | Full content fetch (needs a token), then `next build`.               |
| `npx next build`    | Builds from committed data, no token needed. What CI runs.           |
| `npm run typecheck` | `tsc --noEmit`.                                                      |
| `npm run lint`      | `eslint .`.                                                          |
| `npm run preview`   | Builds and serves the Workers bundle locally.                        |
| `npm run deploy`    | Builds and deploys to Cloudflare.                                    |

## Layout

```
app/          App Router pages: home, about, projects, blog, resume
components/   Hero (the VLA demo), Nav, Footer, and the page sections
lib/          Content types + the committed data JSON, markdown/LaTeX parsing
scripts/      Build-time fetch + parse scripts
config/       Which repos to pull content from
```

## Deploying

`npm run deploy` runs the content fetch, builds through OpenNext and pushes to
Cloudflare Workers. Every deploy ships a fresh asset manifest, so the previous
deploy's content-hashed `/_next/static/*` chunks stop existing. Page HTML is
therefore served with `Cache-Control: public, max-age=0, must-revalidate` (see
`next.config.mjs`) so a returning visitor never boots a page that points at chunks
which are gone.

A tab left open *across* a deploy is a separate case: it holds the old chunk URLs in
memory and only requests them lazily, when "Start Training" constructs the trainer
worker. That fetch 404s, the trainer reports failure by returning to idle, and the
hero swaps the button for a **Reload** (see `loadFailed` in `components/Hero.tsx`).
