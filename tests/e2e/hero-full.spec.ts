import { expect, test } from "@playwright/test";
import {
  collectPageErrors,
  openHero,
  runTryCommand,
  startTraining,
  waitForConverged,
} from "./helpers";

// Full convergence: train to "Ready" and drive the try-it command box.
//
// Measured alone on headless SwiftShader (~2 batches/s): converges around
// 370-560 batches in ~6 min, and mini-vla's `maxBatches: 800` fallback
// guarantees "Ready" regardless. The generous timeout is headroom for a slow
// machine, not the expected cost. Don't drive another browser while this
// runs — sharing SwiftShader starves it badly (see waitForConverged).
//
// Opt-in: CI runs only the alive-gate (hero.spec.ts). Run with
// `npm run e2e:full`, or against production via `VLA_FULL=1 npm run smoke:live`.

test.skip(!process.env.VLA_FULL, "slow convergence spec — set VLA_FULL=1 to run");

test("trains to Ready and answers a typed command", async ({ page, isMobile }) => {
  test.setTimeout(35 * 60_000);
  const errors = collectPageErrors(page);

  await openHero(page, isMobile);
  await startTraining(page);
  await waitForConverged(page, 30 * 60_000);

  // One chip per trained color (rendered on both tiers, visible only on
  // mobile) — a chip's color is guaranteed trainable, so command from it.
  const chip = page.locator(".vla-try-chip").first();
  const color = (await chip.textContent())?.trim() ?? "";
  expect(color).not.toBe("");

  // The language head must decode the command back to the color we asked for.
  // Measured on a converged run: 8/8 colors decode correctly at 98-100%, so
  // this is a real assertion about the shipped pipeline, not a coin flip.
  const decoded = await runTryCommand(page, color, isMobile);
  expect(decoded.name).toBe(color);
  expect(decoded.prob).toBeGreaterThan(50);

  // A command that ran leaves no complaint behind.
  await expect(page.locator(".vla-try-note")).toHaveCount(0);

  expect(errors).toEqual([]);
});
