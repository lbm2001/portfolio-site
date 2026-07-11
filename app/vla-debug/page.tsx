"use client";

// On-device isolation harness for "training never starts on iPhone".
//
// Every failure path in the trainer is console.error-only, and a phone has no
// console. Worse, the two symptoms we are chasing are both SILENT:
//
//  - "stuck on Language Warmup" — that string is what Hero renders for status
//    `loading`, and VLATrainer.start() sets `loading` OPTIMISTICALLY on the main
//    thread before the worker has run a single line. Seeing it proves nothing
//    about the worker. Status only leaves `loading` once the worker posts
//    `training`, which is behind: import(tfjs) → tf.ready() → loadEmbeddings →
//    buildVLAModel → languageWarmup (≤200 × batch 256) → one full trainStep.
//
//  - "sent back to the homepage" — `showDemo` is useState(false) and only
//    closeDemo() clears it. There is no router/location/scroll call anywhere in
//    Hero. So the stack collapsing back to the landing hero means the component
//    REMOUNTED, i.e. Safari reloaded the page out from under it.
//
// So the log has to survive the reload. Every line is mirrored into
// localStorage; on the next load the page replays the previous run at the top.
// If the tab dies during a stage, the breadcrumb names that stage.
//
// Delete this route once the cause is pinned.

import { useCallback, useEffect, useRef, useState } from "react";
import { VLATrainer, type TrainerStatus } from "mini-vla/trainer";
import { VLATrainerCore } from "mini-vla/trainer.core";
import {
  DESKTOP_RUN_CONFIG,
  MOBILE_RUN_CONFIG,
  setRunConfig,
  type RunConfig,
} from "mini-vla/config";
import { VLA_ASSET_BASE } from "@/lib/vla-assets";
import type { ProbeResponse } from "./probe.worker";

const LOG_KEY = "vla-debug-log";
const DONE = "— run finished cleanly —";

export default function VlaDebug() {
  const [lines, setLines] = useState<string[]>([]);
  const [prev, setPrev] = useState<string[]>([]);
  const [busy, setBusy] = useState(false);
  const linesRef = useRef<string[]>([]);
  const workerRef = useRef<Worker | null>(null);
  const trainerRef = useRef<VLATrainer | null>(null);
  const coreRef = useRef<VLATrainerCore | null>(null);
  const t0Ref = useRef(0);

  const append = useCallback((line: string) => {
    const at = ((performance.now() - t0Ref.current) / 1000).toFixed(1);
    const next = [...linesRef.current, `[${at.padStart(6)}s] ${line}`];
    linesRef.current = next;
    setLines(next);
    // mirrored every line: a crash-kill gives us no chance to flush later
    try {
      localStorage.setItem(LOG_KEY, JSON.stringify(next));
    } catch {
      /* private mode / quota — the on-screen log still works */
    }
  }, []);

  // Hand the PREVIOUS load's breadcrumbs to the screen, then take the key over
  // for this run. A previous log with no DONE line is the reload we're hunting.
  useEffect(() => {
    t0Ref.current = performance.now();
    try {
      const raw = localStorage.getItem(LOG_KEY);
      // one-shot read of an external system (the PREVIOUS page load's crash
      // breadcrumbs) — there is no event to subscribe to, and it must also
      // clear the key, so a lazy initializer can't replace it
      // eslint-disable-next-line react-hooks/set-state-in-effect
      if (raw) setPrev(JSON.parse(raw) as string[]);
      localStorage.removeItem(LOG_KEY);
    } catch {
      /* ignore */
    }

    const onErr = (ev: ErrorEvent) =>
      append(`window.onerror: ${ev.message} @ ${ev.filename}:${ev.lineno}`);
    const onRej = (ev: PromiseRejectionEvent) =>
      append(`unhandledrejection: ${String(ev.reason)}`);
    window.addEventListener("error", onErr);
    window.addEventListener("unhandledrejection", onRej);

    const n = navigator as Navigator & {
      deviceMemory?: number;
      standalone?: boolean;
    };
    append(`UA: ${navigator.userAgent}`);
    append(
      `dpr=${window.devicePixelRatio} · cores=${navigator.hardwareConcurrency ?? "?"} · deviceMemory=${n.deviceMemory ?? "?"}GB`
    );
    append(
      `main thread: Worker=${typeof Worker} · OffscreenCanvas=${typeof OffscreenCanvas}`
    );
    append(
      `→ mini-vla workerSupported() would return ${
        typeof Worker !== "undefined" && typeof OffscreenCanvas !== "undefined"
      }`
    );
    append(`stacked layout (≤1099px): ${window.matchMedia("(max-width: 1099px)").matches}`);

    return () => {
      window.removeEventListener("error", onErr);
      window.removeEventListener("unhandledrejection", onRej);
      workerRef.current?.terminate();
      trainerRef.current?.destroy();
      coreRef.current?.reset();
    };
  }, [append]);

  /** One probe stage in the worker, resolved on its done/error reply — or on a
      watchdog timeout. Without the timeout a stage that HANGS (the whole point
      of this harness — a wedged tf.ready()/readback never posts done) would
      leave `busy` stuck true and every stage button disabled forever, which is
      exactly what happened on the iPad: the log froze mid-stage and nothing was
      clickable. The timeout logs the wedge, unlocks the UI, and — critically —
      lets the user proceed to the main-thread trainer stage to A/B it. Each
      worker log message refreshes the deadline: a stage that is merely SLOW
      (cpu-backend batches) keeps itself alive, only true silence trips it. */
  const runStage = useCallback(
    (stage: "caps" | "tf" | "bench" | "fence", quietMs = 25_000) =>
      new Promise<void>((resolve) => {
        if (!workerRef.current) {
          append("creating module worker…");
          const w = new Worker(new URL("./probe.worker.ts", import.meta.url), {
            type: "module",
          });
          w.onerror = (ev) => append(`WORKER onerror: ${ev.message || "(no message)"}`);
          w.onmessageerror = () => append("WORKER onmessageerror (unclonable)");
          workerRef.current = w;
        }
        const w = workerRef.current;
        let timer = 0;
        const done = () => {
          clearTimeout(timer);
          w.removeEventListener("message", onMsg);
          resolve();
        };
        const arm = () => {
          clearTimeout(timer);
          timer = window.setTimeout(() => {
            append(
              `⏱ NO worker output for ${quietMs / 1000}s during stage "${stage}" — ` +
                `it is WEDGED (a hung GPU call blocks this worker thread; its last ` +
                `log line above names where). Reload to reset the worker; try the ` +
                `main-thread trainer to compare.`
            );
            done();
          }, quietMs);
        };
        const onMsg = (e: MessageEvent<ProbeResponse>) => {
          const m = e.data;
          if (m.t === "log") {
            append(m.line);
            arm(); // progress — restart the silence clock
            return;
          }
          if (m.t === "error") append(`STAGE "${m.stage}" FAILED: ${m.msg}`);
          done();
        };
        w.addEventListener("message", onMsg);
        append(`→ stage "${stage}" …`);
        arm();
        w.postMessage({ t: stage });
      }),
    [append]
  );

  const guard = useCallback(
    async (fn: () => Promise<void>) => {
      setBusy(true);
      try {
        await fn();
      } catch (err) {
        append(`THREW: ${String(err)}`);
      } finally {
        setBusy(false);
      }
    },
    [append]
  );

  /** Log status transitions and an occasional batch — never every batch, or the
      log itself becomes the bottleneck (and blows the localStorage quota). */
  const watch = (label: string, read: () => { status: TrainerStatus; batches: number }) => {
    let last: TrainerStatus | null = null;
    return () => {
      const { status, batches } = read();
      if (status !== last) {
        append(`${label}: status → ${status}`);
        last = status;
      }
      if (batches > 0 && batches % 20 === 0) append(`${label}: ${batches} batches`);
    };
  };

  /** `cfg` matters: DESKTOP_RUN_CONFIG (numColors:8, maxBlocks:4) trains a
      bigger color head over bigger synthesized scenes than MOBILE_RUN_CONFIG
      (4/3) — the exact task Hero picks on a ≥1100px viewport. A device that
      survives the mobile-sized task but wedges on the desktop-sized one has a
      capacity/timing problem, not a categorical worker/WebGL failure. `reset()`
      first: the proxy's start() no-ops unless status is idle/error, and a prior
      stage may have left it "converged". */
  const runWorkerTrainer = (cfg: RunConfig, label: string) =>
    guard(async () => {
      append(`→ real VLATrainer (worker path, ${label} cfg)`);
      const t = (trainerRef.current ??= new VLATrainer({
        assetBase: VLA_ASSET_BASE,
        replayFallback: true,
      }));
      t.reset();
      t.start(watch(`worker-trainer(${label})`, () => t), cfg);
    });

  // The decisive A/B: same core, same model, same warm-up — main thread. If this
  // trains and the worker path does not, the worker (its WebGL context) is the
  // culprit, not the workload. Expect UI jank; that is the point of the worker.
  const runInlineTrainer = (cfg: RunConfig, label: string) =>
    guard(async () => {
      append(`→ VLATrainerCore inline (main thread, ${label} cfg, no worker/OffscreenCanvas GL)`);
      coreRef.current?.reset();
      setRunConfig(cfg);
      const c = (coreRef.current ??= new VLATrainerCore());
      void c.start(watch(`inline-core(${label})`, () => c), VLA_ASSET_BASE);
    });

  const finish = () => append(DONE);

  const copy = () =>
    navigator.clipboard?.writeText([...prev, "", ...lines].join("\n"));

  const btn: React.CSSProperties = {
    display: "block",
    width: "100%",
    padding: "14px",
    marginBottom: 8,
    fontSize: 16,
    borderRadius: 8,
    border: "1px solid currentColor",
    background: "transparent",
    color: "inherit",
  };
  const pre: React.CSSProperties = {
    whiteSpace: "pre-wrap",
    wordBreak: "break-word",
    fontFamily: "ui-monospace, monospace",
    fontSize: 12,
    lineHeight: 1.5,
  };

  const crashed = prev.length > 0 && !prev.includes(DONE);

  return (
    <main style={{ padding: 16, maxWidth: 720, margin: "0 auto" }}>
      <h1 style={{ fontSize: 20 }}>VLA device probe</h1>
      <p style={{ fontSize: 13, opacity: 0.8 }}>
        Run the stages in order. If the page reloads on its own, come back here —
        the previous run&apos;s log is replayed below and its last line names the
        stage that killed the tab.
      </p>

      {prev.length > 0 && (
        <section
          style={{
            border: "1px solid currentColor",
            borderRadius: 8,
            padding: 12,
            marginBottom: 16,
          }}
        >
          <strong style={{ fontSize: 13 }}>
            {crashed
              ? "⚠ Previous run ended WITHOUT finishing — the page reloaded itself."
              : "Previous run (completed)."}
          </strong>
          <pre style={pre}>{prev.join("\n")}</pre>
        </section>
      )}

      <button style={btn} disabled={busy} onClick={() => guard(() => runStage("caps"))}>
        1 · Worker boots? WebGL in worker?
      </button>
      <button style={btn} disabled={busy} onClick={() => guard(() => runStage("tf"))}>
        2 · Load tfjs in worker (which backend?)
      </button>
      <button style={btn} disabled={busy} onClick={() => guard(() => runStage("bench"))}>
        3 · Time a real-shape conv2d
      </button>
      <button style={btn} disabled={busy} onClick={() => guard(() => runStage("fence"))}>
        4 · Fence stress (200 GPU readbacks)
      </button>
      <button
        style={btn}
        disabled={busy}
        onClick={() => runWorkerTrainer(MOBILE_RUN_CONFIG, "mobile")}
      >
        5 · Real trainer — worker path (mobile cfg: 4 colors)
      </button>
      <button
        style={btn}
        disabled={busy}
        onClick={() => runWorkerTrainer(DESKTOP_RUN_CONFIG, "desktop")}
      >
        6 · Real trainer — worker path (desktop cfg: 8 colors — what Hero
        picks ≥1100px)
      </button>
      <button
        style={btn}
        disabled={busy}
        onClick={() => runInlineTrainer(MOBILE_RUN_CONFIG, "mobile")}
      >
        7 · Real trainer — main thread (mobile cfg)
      </button>
      <button
        style={btn}
        disabled={busy}
        onClick={() => runInlineTrainer(DESKTOP_RUN_CONFIG, "desktop")}
      >
        8 · Real trainer — main thread (desktop cfg)
      </button>
      <button style={btn} onClick={finish}>
        Mark run finished
      </button>
      <button style={btn} onClick={copy}>
        Copy full log
      </button>

      <pre style={pre}>{lines.join("\n")}</pre>
    </main>
  );
}
