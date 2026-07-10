// default open-next.config.ts file created by @opennextjs/cloudflare
import { defineCloudflareConfig } from "@opennextjs/cloudflare";
// import r2IncrementalCache from "@opennextjs/cloudflare/overrides/incremental-cache/r2-incremental-cache";

const config = defineCloudflareConfig({
	// For best results consider enabling R2 caching
	// See https://opennext.js.org/cloudflare/caching for more details
	// incrementalCache: r2IncrementalCache
});

// What `opennextjs-cloudflare build` runs to produce .next/. The default is
// `npm run build`, whose prebuild hook fetches the résumé from a private repo
// (scripts/build-resume.sh, REQUIRES a GITHUB_TOKEN). Deploy/preview npm
// scripts already run that fetch explicitly beforehand, so the hook only
// duplicated it — and it made the bundle unbuildable in tokenless CI (the e2e
// lane), which must build from the committed lib/*-data.json instead.
//
// The asset check sits BETWEEN the copy and the build on purpose. mini-vla's
// suite proves the package fetches from whatever `assetBase` it is handed; it
// cannot prove this app hands it the right one. A drifted path 404s only when
// a visitor first clicks "Start Training" — the page itself looks healthy — so
// fail the build here instead. No browser, milliseconds.
config.buildCommand =
  "node scripts/copy-vla-assets.mjs && npx vitest run tests/unit/vla-assets.test.ts && npx next build";

export default config;
