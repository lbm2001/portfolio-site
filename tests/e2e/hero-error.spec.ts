import { expect, test } from "@playwright/test";
import { openHero } from "./helpers";

// The error states, driven by failing the real network requests.
//
// With replayFallback on (see Hero.tsx), the package intercepts every failure
// EXCEPT unreachable assets and transparently swaps in the CPU-backend replay,
// so the two cases now diverge:
//
//   "assets"  -> the one live terminal. If BOTH the real embeddings AND the
//                replay's own checkpoints are unreachable, nothing can run:
//                status "error"/"assets" -> "Load failed" + Try again. The
//                retry refetches and (via the replay) genuinely succeeds.
//   "worker"  -> no longer terminal. A dead worker chunk (the stale-chunk
//                outage) is just another reason to fall over: the replay —
//                which never touches the worker or WebGL — takes over and the
//                run reaches a real grasp behind the honest "replay" chip.
//
// The asymmetry is still the point, and still invisible to unit tests and to
// mini-vla's own suite: only a real fetch failure against the deployed bundle
// exercises which path the host lands on.

const status = ".vla-status-text";
const primaryBtn = ".vla-bar .vla-btn";

test("failed embedding assets offer a retry that works", async ({ page, isMobile }) => {
  // Kill every /vla/ fetch — the runtime embeddings AND the replay's manifest +
  // checkpoints. That double-abort is what keeps this a terminal "assets"
  // failure with replayFallback on: the real path errors, the package falls
  // over to the replay, and the replay's own assets are dead too, so it lands
  // right back on "Load failed" (the true last-ditch). Restoring the network
  // and pressing Try again then lets the replay load and train for real.
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

test("a dead worker chunk is rescued by the replay fallback", async ({
  page,
  isMobile,
}) => {
  // Turbopack bundles mini-vla's module worker behind a loader chunk at
  // /_next/static/chunks/turbopack-worker-<hash>.js — NOT the trainer.worker.ts
  // source path. A redeploy deleting that chunk under an open tab is the real
  // stale-chunk outage. `worker.onerror` is how mini-vla notices — and with
  // replayFallback on it treats that as just another reason to fall over,
  // swapping in the CPU-backend replay (which never touches the worker) instead
  // of stranding the viewer on a "Reload" terminal.
  const isWorkerChunk = (url: URL) =>
    /\/_next\/static\/chunks\/.*worker.*\.js/.test(url.pathname);

  let aborted = 0;
  await page.route(isWorkerChunk, (route) => {
    aborted++;
    return route.abort();
  });

  await openHero(page, isMobile);
  await page.locator(primaryBtn).first().click();

  // No "Load failed": the replay transparently takes over, flags itself with
  // the honest "replay" chip, and trains through to a real grasp. Matching
  // Training-or-Ready (rather than a single frame) is robust to the replay's
  // per-visit-randomized convergence time.
  await expect(page.locator(".vla-replay-chip")).toBeVisible({ timeout: 60_000 });
  await expect(page.locator(status)).toHaveText(/Training|Ready/, { timeout: 60_000 });
  await expect(page.locator(status)).not.toHaveText("Load failed");

  // Without this the test would still pass if a bundler rename made the route
  // match nothing — a green check proving only that nothing was intercepted.
  expect(aborted, "worker chunk was never intercepted").toBeGreaterThan(0);
});
