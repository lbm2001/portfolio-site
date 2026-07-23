import { expect, test } from "@playwright/test";
import { openHero } from "./helpers";

// The three host-side `HostFailure` watchdogs in Hero.tsx (onTrainStalled,
// onTrainCollapsed, onLoadStuck/onReplayLoadStuck) exist to catch failures
// that never surface as a package-level error: a wedged fetch that neither
// resolves nor rejects, or a WebGL context that dies mid-run without the
// worker ever posting anything. Nothing in the suite exercised any of them
// (see docs/review-round-1-findings.md #11) — a regression to any of the
// *_MS constants, or to releaseWorkerToIdle() itself, would ship undetected.
//
// Two techniques make these tractable without waiting real wall-clock
// minutes or relying on a genuinely dying GPU context:
//
//  - `page.clock` virtualizes the page's setTimeout/Date, so the host's (and
//    the package's own) watchdog timers can be fast-forwarded instead of
//    waited out for real.
//  - For the two training-loop watchdogs, the real trainer.worker.ts runs a
//    genuine tfjs training loop with no host-controllable hook to make it go
//    silent or emit a dead-physical loss on demand. Rather than fake that
//    (fragile, or requiring mini-vla exports it doesn't have), these tests
//    replace `window.Worker` for the page with a scripted stand-in that
//    speaks the exact wire protocol trainer.ts already depends on (see
//    node_modules/mini-vla/js/src/trainer.worker.ts's documented
//    `{t:"state", gen, status, errorReason, loss, smoothLoss, initialLoss,
//    batches}` message) — the real VLATrainer proxy, Hero's onUpdate, and the
//    watchdogs under test all run unmodified; only the noisy tfjs/WebGL
//    innards are swapped out. mini-vla is the only thing on this page that
//    constructs a Worker, so replacing the global is safe here.

const status = ".vla-status-text";
const primaryBtn = ".vla-bar .vla-btn";

test.describe("training-loop watchdogs (scripted worker double)", () => {
  /** Install a `window.Worker` stand-in for mini-vla's trainer.worker.ts.
   *  On "start" it always posts one real "training" batch (so the run
   *  genuinely leaves "loading" and disarms the load watchdog exactly like a
   *  healthy device would) — then, depending on `mode`:
   *   - "stall": posts nothing else, ever — the silence onTrainStalled exists
   *     to catch (a dead WebGL context that produces no further batches).
   *   - "collapse": immediately posts one more batch reporting a "converged"
   *     status with a loss under CONVERGED_LOSS_FLOOR — the zeroed-readback
   *     false-convergence onTrainCollapsed exists to reject.
   *  "resume" is acked with an unchanged-batches "training" state (see below,
   *  needed for the pause/resume-race test); any other posted message
   *  (pause/reset/predict/...) is ignored — none of those paths are under
   *  test here. */
  async function installScriptedWorker(
    page: import("@playwright/test").Page,
    mode: "stall" | "collapse",
  ) {
    await page.addInitScript((mode: string) => {
      class ScriptedTrainerWorker {
        onmessage: ((ev: MessageEvent) => void) | null = null;
        onerror: ((ev: unknown) => void) | null = null;
        onmessageerror: ((ev: unknown) => void) | null = null;
        constructor(_url: string | URL, _opts?: unknown) {}
        postMessage(msg: { t: string; gen: number }) {
          const post = (data: Record<string, unknown>) => {
            this.onmessage?.({ data } as MessageEvent);
          };
          if (msg.t === "resume") {
            // A real worker's ack always lands on a later task than the
            // synchronous trainer.resume() call that sent it — by which point
            // resumeTraining() has already set statusRef.current itself, so
            // onUpdate sees trainer.status === statusRef.current and takes
            // the steady-state branch (the "Resume re-arm" logic under test),
            // not the transition branch (which unconditionally re-arms
            // regardless of pauseTraining's own behavior, masking the bug
            // this test exists to catch). Match that with a microtask, same
            // as the "start" ack below — a synchronous ack here would land
            // before that assignment and false-pass this test.
            void Promise.resolve().then(() => {
              post({
                t: "state",
                gen: msg.gen,
                status: "training",
                errorReason: null,
                loss: 0.5,
                smoothLoss: 0.5,
                initialLoss: 1,
                batches: 1,
              });
            });
            return;
          }
          if (msg.t !== "start") return; // pause/reset/predict/... unused here
          // Microtask, not a timer: fires regardless of a fake/paused clock,
          // matching a real worker's async-but-immediate first reply.
          void Promise.resolve().then(() => {
            post({
              t: "state",
              gen: msg.gen,
              status: "training",
              errorReason: null,
              loss: 0.5,
              smoothLoss: 0.5,
              initialLoss: 1,
              batches: 1,
            });
            if (mode === "collapse") {
              void Promise.resolve().then(() => {
                post({
                  t: "state",
                  gen: msg.gen,
                  status: "converged",
                  errorReason: null,
                  loss: 0, // below CONVERGED_LOSS_FLOOR (1e-4): non-physical
                  smoothLoss: 0,
                  initialLoss: 1,
                  batches: 2,
                });
              });
            }
            // mode "stall": deliberately no further message, ever.
          });
        }
        terminate() {}
      }
      (window as unknown as { Worker: unknown }).Worker = ScriptedTrainerWorker;
    }, mode);
  }

  test("a worker that goes silent mid-run trips the training-stalled watchdog", async ({
    page,
    isMobile,
  }) => {
    // Regression this catches: TRAIN_STALL_MS firing too late/never, or
    // releaseWorkerToIdle() not actually tearing the run down, would leave
    // the bar spinning on "Training" forever with no way back for the viewer.
    await installScriptedWorker(page, "stall");
    await page.clock.install();

    await openHero(page, isMobile);
    await page.locator(primaryBtn).first().click();

    // The scripted worker's one real batch must land for real (not from the
    // fast-forward below) — proves the run genuinely reached "training"
    // before going quiet, not that the watchdog fired on a run that never
    // started.
    await expect(page.locator(status)).toHaveText("Training", { timeout: 10_000 });

    // Jump past TRAIN_STALL_MS (20s) with no further batch — this is the
    // condition onTrainStalled exists to catch.
    await page.clock.fastForward("00:21");

    await expect(page.locator(status)).toHaveText("Training stalled — reload");
    // hostFailure always renders "Reload" (window.location.reload()), never
    // "Start Training"/"Try again" — a fresh `new Worker(...)` in this tab
    // isn't safe from the same GPU-context exhaustion (see Hero.tsx), so
    // offering an in-place retry here would be the actual regression.
    await expect(page.locator(primaryBtn).first()).toHaveText("Reload");
  });

  test("a converged report with non-physical loss trips the training-collapsed watchdog", async ({
    page,
    isMobile,
  }) => {
    // Regression this catches: the false-convergence guard (Hero.tsx, the
    // CONVERGED_LOSS_FLOOR check) silently accepting a zeroed/dead readback as
    // a real trained policy instead of rejecting it via onTrainCollapsed.
    await installScriptedWorker(page, "collapse");

    await openHero(page, isMobile);
    await page.locator(primaryBtn).first().click();

    // No clock trickery needed: the scripted worker's second message reports
    // "converged" with a non-physical loss right after the first, and the
    // guard's reject branch is synchronous — no watchdog interval to wait out.
    await expect(page.locator(status)).toHaveText("Training collapsed — reload", {
      timeout: 10_000,
    });
    await expect(page.locator(primaryBtn).first()).toHaveText("Reload");
  });

  test("pausing just before the stall deadline and resuming doesn't falsely trip the watchdog", async ({
    page,
    isMobile,
  }) => {
    // Regression this catches: pauseTraining() previously left the stall
    // watchdog armed with its PRE-pause deadline. A pause/resume landing
    // close enough to that deadline would leave the stale timer pending
    // through the resume, and it would fire moments later against the
    // resumed (and healthy) "training" status — tearing down a run that
    // never actually stalled. See pauseTraining's own comment.
    await installScriptedWorker(page, "stall");
    await page.clock.install();

    await openHero(page, isMobile);
    await page.locator(primaryBtn).first().click();
    await expect(page.locator(status)).toHaveText("Training", { timeout: 10_000 });

    // Well short of TRAIN_STALL_MS (20s) — wide margin against real-browser
    // jitter in exactly when the arm happened relative to this fast-forward
    // (the stall timer is still pending either way).
    await page.clock.fastForward("00:10");
    await page.locator(primaryBtn).first().click(); // Pause
    await expect(page.locator(status)).toHaveText("Paused");

    await page.locator(primaryBtn).first().click(); // Resume
    await expect(page.locator(status)).toHaveText("Training");
    // Flush the scripted worker's synchronous resume ack (see
    // installScriptedWorker) before advancing the clock below — real-time
    // wait, independent of the virtualized clock.
    await page.waitForTimeout(200);

    // Cross the OLD (pre-pause) deadline (~20s from arm) by a wide margin,
    // while staying well short of a FRESH deadline armed at the resume point
    // (~10s + 20s = ~30s). The buggy version's stale timer fires somewhere
    // around the old deadline, against the now-resumed "training" status,
    // and falsely stalls the run; the fixed version re-armed on resume, so
    // this alone must NOT trip it.
    await page.clock.fastForward("00:15");
    await expect(page.locator(status)).toHaveText("Training");

    // The watchdog isn't disabled outright, just correctly re-timed from the
    // resume point: cross the fresh deadline (~30s total) with still no
    // further batch, and confirm it still fires.
    await page.clock.fastForward("00:10");
    await expect(page.locator(status)).toHaveText("Training stalled — reload");
  });
});

test("a real load that never progresses past warmup trips the two-tier load-stuck watchdog", async ({
  page,
  isMobile,
}) => {
  // Regression this catches: either tier of the load watchdog (onLoadStuck,
  // then onReplayLoadStuck) failing to fire — or onLoadStuck incorrectly
  // killing the run the moment the package swaps to the replay, instead of
  // deferring to the longer replay-load ceiling per its own comment.
  //
  // Real network requests hang forever (never resolve/reject) rather than
  // abort — the genuine "flaky link" case these two watchdogs exist to catch,
  // as opposed to hero-error.spec.ts's aborts, which the package/host both
  // recover from via other paths (retry / replay) long before any watchdog is
  // needed.
  await page.route("**/vla/**", () => new Promise<never>(() => {}));
  await page.clock.install();

  await openHero(page, isMobile);
  await page.locator(primaryBtn).first().click();
  await expect(page.locator(status)).toHaveText("Language Warmup", { timeout: 10_000 });

  // Past REPLAY_WATCHDOG_MS (6s, Hero's override): the package's own watchdog
  // gives up on the real (also-hung) path and transparently swaps to the
  // replay — still "loading", now off a second hung fetch (the replay's own
  // timeout-free asset load). The chip is the only host-visible sign this
  // happened; it also proves the run hasn't failed outright yet.
  await page.clock.fastForward("00:07");
  await expect(page.locator(".vla-replay-chip")).toBeVisible();
  await expect(page.locator(status)).toHaveText("Language Warmup");

  // Past LOADING_WATCHDOG_MS (15s total): onLoadStuck fires, but MUST NOT
  // tear the run down here — trainerRef.current.usingReplay is true, so it
  // has to defer to the second, longer ceiling instead of killing a swap
  // that's still (legitimately, from the host's view) in flight.
  await page.clock.fastForward("00:09");
  await expect(page.locator(status)).toHaveText("Language Warmup");

  // Past REPLAY_LOAD_WATCHDOG_MS (20s further, ~35s total): the replay's own
  // hung fetch has now outlasted its budget too — onReplayLoadStuck tears the
  // run down for real.
  await page.clock.fastForward("00:21");
  await expect(page.locator(status)).toHaveText("Stuck — reload to retry");
  await expect(page.locator(primaryBtn).first()).toHaveText("Reload");
});
