import { expect, test } from "@playwright/test";
import { openHero } from "./helpers";

// The error states, driven by failing the real network requests.
//
// Both are host-only concerns: mini-vla reports status "error" with a reason,
// but WHICH recovery the viewer is offered is Hero.tsx's call, and the
// asymmetry is the point:
//
//   "assets" -> Try again. start() refetches (loadEmbeddings un-caches a
//               rejected promise), so an in-page retry can genuinely succeed.
//   "worker" -> Reload. A fresh `new Worker(...)` resolves the same dead chunk
//               URL, so retrying in place is futile.
//
// Getting this backwards is invisible in unit tests and in mini-vla's suite,
// and it is exactly the stale-chunk outage: a button that looks alive and
// cannot possibly work.

const status = ".vla-status-text";
const primaryBtn = ".vla-bar .vla-btn";

test("failed embedding assets offer a retry that works", async ({ page, isMobile }) => {
  // Kill the runtime embedding fetch (/vla/<version>/{embeddings-50d.bin,vocab.txt}).
  await page.route("**/vla/**", (route) => route.abort());

  await openHero(page, isMobile);
  await page.locator(primaryBtn).first().click();

  await expect(page.locator(status)).toHaveText("Load failed", { timeout: 60_000 });
  const btn = page.locator(primaryBtn).first();
  await expect(btn).toHaveText("Try again");

  // The retry is not decorative: with the network restored it must actually
  // refetch and get training. This is the half a mocked test would miss.
  await page.unroute("**/vla/**");
  await btn.click();
  await expect(page.locator(status)).not.toHaveText("Load failed", { timeout: 60_000 });
  await expect(page.locator(status)).toHaveText(/Language Warmup|Training/, {
    timeout: 60_000,
  });
});

test("a dead worker chunk offers Reload, not a futile retry", async ({
  page,
  isMobile,
}) => {
  // Turbopack bundles mini-vla's module worker behind a loader chunk at
  // /_next/static/chunks/turbopack-worker-<hash>.js — NOT the trainer.worker.ts
  // source path. A redeploy deleting that chunk under an open tab is the real
  // stale-chunk outage. `worker.onerror` is how mini-vla notices.
  const isWorkerChunk = (url: URL) =>
    /\/_next\/static\/chunks\/.*worker.*\.js/.test(url.pathname);

  let aborted = 0;
  await page.route(isWorkerChunk, (route) => {
    aborted++;
    return route.abort();
  });

  await openHero(page, isMobile);
  await page.locator(primaryBtn).first().click();

  await expect(page.locator(status)).toHaveText("Load failed", { timeout: 60_000 });
  await expect(page.locator(primaryBtn).first()).toHaveText("Reload");

  // Without this the test would still pass if a bundler rename made the route
  // match nothing — a green check proving only that nothing was intercepted.
  expect(aborted, "worker chunk was never intercepted").toBeGreaterThan(0);
});
