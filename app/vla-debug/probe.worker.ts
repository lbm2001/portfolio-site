// Capability probe for the VLA trainer's worker environment, staged so the
// page can tell WHICH layer fails on a device with no attachable console.
//
// mini-vla's trainer.ts decides to go off-main-thread on
// `typeof Worker !== "undefined" && typeof OffscreenCanvas !== "undefined"`.
// Neither of those implies the thing tfjs actually needs there: a WebGL context
// on an OffscreenCanvas INSIDE a worker. iOS Safari ships OffscreenCanvas, so
// it takes the worker path regardless. This worker probes each requirement
// separately and reports over postMessage, so a hang or a web-process crash
// pins itself to one stage instead of showing up as a silent "Language Warmup"
// that never ends.
//
// Stages are driven one at a time from the page (never auto-chained here): if a
// stage kills the tab, the page's breadcrumb log names the stage that did it.

type ProbeRequest = { t: "caps" } | { t: "tf" } | { t: "bench" };

export type ProbeResponse =
  | { t: "log"; line: string }
  | { t: "done"; stage: ProbeRequest["t"] }
  | { t: "error"; stage: ProbeRequest["t"]; msg: string };

// This file is a worker, but the project compiles against lib.dom (which types
// `self` as a Window, whose postMessage demands a targetOrigin) and cannot add
// lib.webworker without colliding with it. Narrow to just what we call.
interface WorkerScope {
  postMessage(message: unknown): void;
  onmessage: ((e: MessageEvent<ProbeRequest>) => void) | null;
}
const ctx = self as unknown as WorkerScope;
const post = (m: ProbeResponse) => ctx.postMessage(m);
const log = (line: string) => post({ t: "log", line });

const since = (t0: number) => `${Math.round(performance.now() - t0)}ms`;

let tf: typeof import("@tensorflow/tfjs") | null = null;

/** Does a WebGL context exist on an OffscreenCanvas in THIS thread? A fresh
    canvas per attempt — getContext only ever returns one context kind. */
function probeContexts() {
  for (const id of ["webgl2", "webgl"] as const) {
    try {
      const gl = new OffscreenCanvas(64, 64).getContext(id);
      if (!gl) {
        log(`  getContext("${id}") → null (unsupported in a worker)`);
        continue;
      }
      // Naming the GPU proves the context is real, not a stub that fails later.
      const dbg = (gl as WebGLRenderingContext).getExtension(
        "WEBGL_debug_renderer_info"
      );
      const renderer = dbg
        ? (gl as WebGLRenderingContext).getParameter(
            (dbg as { UNMASKED_RENDERER_WEBGL: number }).UNMASKED_RENDERER_WEBGL
          )
        : "(renderer hidden)";
      log(`  getContext("${id}") → OK — ${renderer}`);
    } catch (err) {
      log(`  getContext("${id}") THREW — ${String(err)}`);
    }
  }
}

ctx.onmessage = async (e: MessageEvent<ProbeRequest>) => {
  const stage = e.data.t;
  try {
    if (stage === "caps") {
      log(`worker alive · OffscreenCanvas=${typeof OffscreenCanvas}`);
      if (typeof OffscreenCanvas === "undefined") {
        log("  → no OffscreenCanvas here; tfjs cannot use WebGL in a worker");
      } else {
        // The stage most likely to hard-crash the WebKit web process on iOS.
        probeContexts();
      }
    }

    if (stage === "tf") {
      const t0 = performance.now();
      tf = await import("@tensorflow/tfjs");
      log(`import("@tensorflow/tfjs") resolved in ${since(t0)}`);
      const t1 = performance.now();
      await tf.ready();
      // THE headline number: "cpu" here means every gradient step on this phone
      // runs on the CPU, and the trainer's warm-up (200 × batch 256) plus its
      // 64×64×3 conv steps are sized for a GPU.
      log(
        `tf.ready() in ${since(t1)} · backend=${tf.getBackend()} · tfjs ${tf.version.tfjs}`
      );
    }

    if (stage === "bench") {
      if (!tf) throw new Error("run the tfjs stage first");
      // The real training-step shape: batch 32 of 64×64×3 silhouettes.
      const x = tf.randomNormal<import("@tensorflow/tfjs").Rank.R4>([
        32, 64, 64, 3,
      ]);
      const f = tf.randomNormal<import("@tensorflow/tfjs").Rank.R4>([
        3, 3, 3, 16,
      ]);
      // Discard the first pass — it pays WebGL's one-time shader compile.
      await tf.conv2d(x, f, 1, "same").data();
      for (let i = 0; i < 3; i++) {
        const t0 = performance.now();
        const y = tf.conv2d(x, f, 1, "same");
        await y.data(); // forces the readback, so the timing is not a lie
        y.dispose();
        log(`  conv2d [32,64,64,3] pass ${i + 1}: ${since(t0)}`);
      }
      x.dispose();
      f.dispose();
      const m = tf.memory();
      log(`tf.memory(): ${m.numTensors} tensors · ${(m.numBytes / 1e6).toFixed(1)} MB`);
    }

    post({ t: "done", stage });
  } catch (err) {
    post({ t: "error", stage, msg: String(err) });
  }
};
