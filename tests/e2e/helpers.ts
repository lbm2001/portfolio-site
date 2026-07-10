import { expect, type Page } from "@playwright/test";

// Content images under /projects/ and /blog/ are fetched at build time with a
// GITHUB_TOKEN and are absent from tokenless CI builds — their 404s are
// expected there, but NOT against the live site (smoke:live must catch a
// deploy that lost them). Everything else (/_next/ chunks, /vla/ assets)
// stays strict everywhere: a 404 on those is the deploy-breaking failure
// mode this suite hunts.
const EXPECTED_MISSING = process.env.PLAYWRIGHT_BASE_URL
  ? /$^/ // matches nothing — live runs stay strict
  : /\/(projects|blog)\/[^?]+\.(png|jpe?g|gif|webp|avif|svg)$/i;

/** Collect uncaught exceptions and console.error output for the whole page
 *  lifetime — worker/tfjs failures surface here, not in the DOM. */
export function collectPageErrors(page: Page): string[] {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(`pageerror: ${e.message}`));
  page.on("console", (msg) => {
    if (msg.type() !== "error") return;
    if (EXPECTED_MISSING.test(msg.location().url)) return;
    errors.push(`console.error: ${msg.text()} (${msg.location().url})`);
  });
  return errors;
}

/** Land on the hero with the pipeline visible. On the mobile tier the pipeline
 *  is display:none until the "Mini VLA Demo" CTA stacks it. */
export async function openHero(page: Page, isMobile: boolean): Promise<void> {
  await page.goto("/");
  if (isMobile) {
    await page.locator(".hero-demo-btn").click();
  }
  await expect(page.locator(".vla-bar")).toBeVisible();
}

/** Click "Start Training" and wait until the run is actually training.
 *  Distinguishes a dead button / failed asset load from a slow warmup. */
export async function startTraining(page: Page): Promise<void> {
  const primaryBtn = page.locator(".vla-bar .vla-btn").first();
  await expect(primaryBtn).toHaveText("Start Training");
  await primaryBtn.click();

  const status = page.locator(".vla-status-text");
  // The button must react at all (stale-chunk regression: it silently did
  // nothing when the deployed HTML referenced a previous deploy's chunks).
  await expect(status).not.toHaveText("Idle", { timeout: 30_000 });

  // Language warmup fetches the embedding assets; then training begins.
  // NOTE: waitForFunction's signature is (fn, arg, options) — passing the
  // options object in the `arg` slot silently drops it and leaves the 30s
  // default in place. The `null` arg is what makes this timeout real.
  await page.waitForFunction(
    () => {
      const t = document.querySelector(".vla-status-text")?.textContent;
      return t === "Training" || t === "Load failed";
    },
    null,
    { timeout: 180_000 },
  );
  await expect(status).toHaveText("Training");
}

/** Current batch count from the HUD ("1,234 batches"), or 0 before it shows. */
export async function batchCount(page: Page): Promise<number> {
  const subs = page.locator(".vla-status-sub");
  if ((await subs.count()) < 2) return 0;
  const text = (await subs.nth(1).textContent()) ?? "";
  return Number(text.replace(/\D/g, "")) || 0;
}

/** Parsed ".vla-decoded" readout, or null while it reads "—". */
async function readDecoded(page: Page): Promise<{ name: string; prob: number } | null> {
  const text = (await page.locator(".vla-decoded").textContent()) ?? "";
  const m = /decoded target:\s*(\w+)\s+(\d+)%/.exec(text);
  return m ? { name: m[1], prob: Number(m[2]) } : null;
}

/**
 * Ask the converged policy for `color`, reshuffling the scene as needed.
 *
 * The scene only holds a few blocks, so a command naming a trained colour that
 * isn't currently on the table is REFUSED with "no <c> block in this scene —
 * ⟳ to reshuffle" rather than executed. Following that affordance is the whole
 * point: without it the test reads a stale answer, because Hero's `langViz`
 * rAF loop keeps re-decoding the still-active demo sentence and repaints the
 * previous colour over the blanked readout.
 *
 * Returns the decoded readout for the command that actually ran.
 */
export async function runTryCommand(
  page: Page,
  color: string,
  isMobile: boolean,
  attempts = 6,
): Promise<{ name: string; prob: number }> {
  for (let i = 0; i < attempts; i++) {
    if (isMobile) {
      await page.locator(".vla-try-chip", { hasText: color }).first().click();
    } else {
      await page.locator(".vla-try-input").fill(`pick up the ${color} block`);
      await page.locator(".vla-try-btn", { hasText: "Run" }).click();
    }

    // Give the worker round-trip a beat, then read the outcome.
    await page.waitForTimeout(1_200);

    // `.vla-try-note` only exists when the hero REFUSED the command. Check
    // existence with count() first: textContent() auto-waits for the element
    // to attach, and with Playwright's default actionTimeout of 0 that waits
    // forever on the success path — a hang, not a caught error.
    const noteEl = page.locator(".vla-try-note");
    const note = (await noteEl.count()) > 0 ? await noteEl.textContent() : null;
    if (note && /no .* block in this scene/.test(note)) {
      await page.locator(".vla-try-shuffle").click();
      await page.waitForTimeout(600);
      continue;
    }

    const decoded = await readDecoded(page);
    if (decoded) return decoded;
  }
  throw new Error(`"${color}" never ran after ${attempts} reshuffles`);
}

/**
 * Wait for training to reach "Ready".
 *
 * mini-vla CONFIG.converge: a healthy pick-up run crosses the loss threshold
 * somewhere around 370-560 batches, and `maxBatches: 800` converges it
 * regardless — so "Ready" is guaranteed by batch 800. Measured alone on
 * headless SwiftShader: ~2 batches/s, converging in ~6 min.
 *
 * That rate collapses under GPU contention (a second browser sharing
 * SwiftShader dropped it ~5x and froze the counter for minutes at a time),
 * which is why the convergence specs run with workers: 1 and want nothing
 * else driving a browser. Rather than sit behind one blind timeout, watch the
 * batch counter: a run that stops advancing is hung (dead worker, lost
 * backend) and should fail in minutes, not at the ceiling.
 */
export async function waitForConverged(page: Page, timeoutMs: number): Promise<void> {
  const STALL_MS = 180_000;
  const deadline = Date.now() + timeoutMs;
  let lastBatches = -1;
  let lastProgressAt = Date.now();

  while (Date.now() < deadline) {
    const status = await page.locator(".vla-status-text").textContent();
    if (status === "Ready") return;
    if (status === "Load failed") throw new Error("hero reported 'Load failed'");

    const batches = await batchCount(page);
    if (batches > lastBatches) {
      lastBatches = batches;
      lastProgressAt = Date.now();
    } else if (Date.now() - lastProgressAt > STALL_MS) {
      throw new Error(
        `training stalled at ${lastBatches} batches for ${STALL_MS / 1000}s (status: ${status})`,
      );
    }
    await page.waitForTimeout(2_000);
  }
  throw new Error(
    `did not reach "Ready" within ${Math.round(timeoutMs / 1000)}s (last: ${lastBatches} batches)`,
  );
}
