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

type ProbeRequest = { t: "caps" } | { t: "tf" } | { t: "bench" } | { t: "fence" };

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

/** Does a USABLE WebGL context exist on an OffscreenCanvas in THIS thread? A
    fresh canvas per attempt — getContext only ever returns one context kind.
    getContext succeeding is necessary but NOT sufficient: iPadOS Safari can
    hand back a context that then wedges the web process the moment tfjs
    allocates textures + compiles shaders on it (which is what tf.ready() and
    the first op do). So beyond naming the GPU, actually clear the buffer and
    read one pixel back — a minimal real use — before declaring it OK. Each
    context is explicitly released via WEBGL_lose_context so the probe's own
    contexts don't compete with the tfjs backend created in the "tf" stage
    (iPadOS caps live WebGL contexts aggressively). */
function probeContexts() {
  for (const id of ["webgl2", "webgl"] as const) {
    try {
      const gl = new OffscreenCanvas(64, 64).getContext(id) as
        | WebGLRenderingContext
        | null;
      if (!gl) {
        log(`  getContext("${id}") → null (unsupported in a worker)`);
        continue;
      }
      // Naming the GPU proves the context is real, not a stub that fails later.
      const dbg = gl.getExtension("WEBGL_debug_renderer_info");
      const renderer = dbg
        ? gl.getParameter(
            (dbg as { UNMASKED_RENDERER_WEBGL: number }).UNMASKED_RENDERER_WEBGL
          )
        : "(renderer hidden)";
      // Minimal real use: clear + synchronous readback. If getContext lies,
      // this is where it surfaces (throw, or a lost-context web-process kill).
      gl.clearColor(0, 0, 0, 1);
      gl.clear(gl.COLOR_BUFFER_BIT);
      const px = new Uint8Array(4);
      gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, px);
      const lost = gl.isContextLost();
      log(
        `  getContext("${id}") → OK — ${renderer} · readPixels=[${px.join(",")}]${
          lost ? " · CONTEXT LOST" : ""
        }`
      );
      gl.getExtension("WEBGL_lose_context")?.loseContext();
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
      // The worker-side UA is what mini-vla's fence workaround gates on
      // (WorkerNavigator has no `vendor`). iPadOS masquerades as desktop macOS
      // here, and non-Safari iPad browsers masquerade as desktop Chrome/Edge/
      // Firefox — which this regex (copied verbatim from trainer.core.ts's
      // maybeDisableWebGLFence) does NOT treat as WebKit, so the fence bug
      // workaround never engages on them.
      const ua = navigator.userAgent ?? "";
      log(`worker UA: ${ua}`);
      const gateFires =
        /AppleWebKit/.test(ua) && !/Chrome|Chromium|Edg\//.test(ua);
      log(
        `→ mini-vla's iOS fence workaround ${gateFires ? "WILL" : "will NOT"} engage here`
      );
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
      // Split init from first-use so a wedge pins itself to ONE of them. On the
      // iPad this stage's last line was the import above and nothing after —
      // these breadcrumbs say whether tf.ready() (WebGL context + capability
      // probing) or the first real GPU readback is the wall.
      log("calling tf.ready() …");
      const t1 = performance.now();
      await tf.ready();
      // THE headline number: "cpu" here means every gradient step on this phone
      // runs on the CPU, and the trainer's warm-up (200 × batch 256) plus its
      // 64×64×3 conv steps are sized for a GPU.
      log(
        `tf.ready() in ${since(t1)} · backend=${tf.getBackend()} · tfjs ${tf.version.tfjs}`
      );
      // The first actual op + async readback on the chosen backend. On webgl
      // this is the first time a fence-gated download runs — the exact call
      // that wedges under the iOS WebKit fence bug (see the "fence" stage).
      log("first GPU readback …");
      const t2 = performance.now();
      const lib = tf; // narrowed local: TS loses the module-var narrowing in the tidy closure
      const a = lib.tidy(() => lib.scalar(2).square());
      const v = await a.data();
      a.dispose();
      log(`first readback ok in ${since(t2)} · got ${v[0]}`);
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

    if (stage === "fence") {
      if (!tf) throw new Error("run the tfjs stage first");
      // The iOS WebKit fence bug: tfjs resolves every async `tensor.data()` on
      // the webgl backend by polling a fenceSync, and on WebKit the fence stops
      // signalling after some tens of reads — the promise then never settles.
      // The trainer does one such read per trainOnBatch output head, so a wedge
      // here IS the "Language Warmup forever" / "batches frozen" hang. tfjs's
      // default flag value is left untouched on purpose: this measures what a
      // browser mini-vla's UA gate misjudged would actually do. If the counter
      // below stops advancing, the fence bug is confirmed on this device —
      // reload the page to un-wedge the worker.
      log(
        `WEBGL_FENCE_API_ENABLED=${tf.env().getBool("WEBGL_FENCE_API_ENABLED")} · backend=${tf.getBackend()}`
      );
      const t0 = performance.now();
      for (let i = 1; i <= 200; i++) {
        const x = tf.scalar(i);
        const y = x.square();
        await y.data(); // one fence-gated readback, like a trainOnBatch loss
        x.dispose();
        y.dispose();
        if (i % 20 === 0) log(`  read ${i}/200 ok (${since(t0)})`);
      }
    }

    post({ t: "done", stage });
  } catch (err) {
    post({ t: "error", stage, msg: String(err) });
  }
};
