"use client";

import {
  Fragment,
  useCallback,
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
} from "react";
import Link from "next/link";
import { profile, resumeDownloadName } from "@/lib/content";
import { REST, clamp } from "mini-vla/geometry";
import {
  paintScene,
  paintSilhouette,
  sceneMap,
  type ScenePalette,
} from "mini-vla/scene";
import {
  COLORS,
  DEFAULT_LAYOUT,
  DEFAULT_SENTENCE,
  MAX_SEQ_LEN,
  activePalette,
  randomLayout,
  sampleCommand,
  tokenize,
  DEMO_PERIOD_MS,
  demoPose,
  makeDemoPlan,
  type BlockPos,
  type Layout,
  type Sentence,
  type DemoPlan,
} from "mini-vla/task";
import {
  CONFIG,
  DESKTOP_RUN_CONFIG,
  MOBILE_RUN_CONFIG,
  setRunConfig,
  type RunConfig,
} from "mini-vla/config";
import { IMG_SIZE } from "mini-vla/model";
import {
  VLATrainer,
  type TrainerError,
  type TrainerStatus,
} from "mini-vla/trainer";
import { VLA_ASSET_BASE } from "@/lib/vla-assets";
import { BUILD_ID_PATH, type BuildIdPayload } from "@/lib/build-id";
import {
  RolloutEngine,
  type RolloutFrame,
  type RolloutPhase,
} from "mini-vla/rollout";
import {
  CAPTION_ALMOST,
  CAPTION_DEMO,
  CAPTION_GAZE,
  CAPTION_ROLLOUT,
  InfoDot,
  type InfoId,
  loadHints,
  saveHints,
  TIPS,
  TRY_PLACEHOLDERS,
} from "@/components/hero/guidance";

// Live Vision-Language-Action hero: four pipeline boxes ringed around the
// name (Demonstration left w/ floating prompt above, Vision Encoder top,
// Language Encoder bottom, Rollout right). "Start Training" runs a GENUINE
// TensorFlow.js behavioral-cloning loop — hosted in a Web Worker off the main
// thread (mini-vla's trainer.worker.ts, via the mini-vla/trainer proxy) so
// gradient steps never fight this component's 60fps rAF loop — against an
// analytical-IK expert on pick-up commands (the viewport's task profile — see
// DESKTOP_RUN_CONFIG / MOBILE_RUN_CONFIG): each scene's slot-grammar command
// names one block to pick up. The displayed demonstration swaps every cycle while
// thousands of examples train invisibly; the Rollout runs policy-driven
// episodes — the arm approaches with an open gripper and the LEARNED gripper
// action closes it over the block (attaching it), then the carry phase (lift
// it home) follows the policy's predictions. Once the action
// loss converges, training stops and the Rollout box becomes interactive:
// type your own command, run it, reshuffle the blocks.

const ACCENT = "#e12d1a"; // = --red; canvases can't read CSS vars cheaply

// The demo/rollout scenes' cosmetic look is owned HERE (host-side) and passed
// into paintScene as a palette. The light palette matches the renderer's
// defaults, so the visible output is unchanged — the point is that the look now
// lives with the host, not the model. (paintSilhouette takes no palette: it is
// the literal model input — a fixed white-background view the trained
// checkpoints expect — and MUST NOT be recolored for a theme.)
const LIGHT_SCENE_PALETTE: ScenePalette = {
  floor: "#e6e6e6",
  pedestal: "#2b2b2b",
  link: "#8a8a8a",
  joint: "#fff",
  effectorOpen: "#fff",
  effectorOpenEdge: "#6f6f6f",
  effectorClosed: "#6f6f6f",
  // faint stroke so same-tone blocks (e.g. white) stay legible on the light floor
  blockEdge: "rgba(0,0,0,0.12)",
};
// Dark-mode arm/scene look: the near-black pedestal becomes a light structural
// tone, the white joints/open-jaw stay light, and the floor line lifts just
// above --bg. link stays mid-grey (reads on both). Default guesses — see the
// mini-vla handoff; the block colours themselves live in the package.
const DARK_SCENE_PALETTE: ScenePalette = {
  floor: "#333333",
  pedestal: "#c9c9c9",
  link: "#8a8a8a",
  joint: "#f2f2f2",
  effectorOpen: "#f2f2f2",
  effectorOpenEdge: "#8a8a8a",
  effectorClosed: "#9a9a9a",
  // the trained "black" block (#1c1c1c) vanishes against the dark floor — remap
  // it to a light structural tone for display only (paintSilhouette is untouched,
  // so the model input keeps the real black); stroke keeps the rest legible.
  blockColor: (_i, hex) => (hex.toLowerCase() === "#1c1c1c" ? "#d8d8d8" : hex),
  blockEdge: "rgba(255,255,255,0.18)",
};

// Canvases can't read CSS vars cheaply, so the theme is resolved here and
// mirrored into a ref the rAF loop reads each frame. Effective theme = the
// <html data-theme> override the viewer set via the nav toggle, else the OS
// preference — the same rule the CSS dark block uses; keep them in step.
type CanvasTheme = {
  palette: ScenePalette;
  lossBaseline: string; // the loss curve's zero-line
  visionBorder: string; // the vision panel's frame stroke
};
const LIGHT_CANVAS: CanvasTheme = {
  palette: LIGHT_SCENE_PALETTE,
  lossBaseline: "#efefef",
  visionBorder: "rgba(0,0,0,.12)",
};
const DARK_CANVAS: CanvasTheme = {
  palette: DARK_SCENE_PALETTE,
  lossBaseline: "#2c2c2c",
  visionBorder: "rgba(255,255,255,0.14)",
};
const resolveCanvasTheme = (): CanvasTheme => {
  const t = document.documentElement.getAttribute("data-theme");
  const dark =
    t === "dark" ||
    (t !== "light" &&
      window.matchMedia("(prefers-color-scheme: dark)").matches);
  return dark ? DARK_CANVAS : LIGHT_CANVAS;
};

// The rollout state machine — phases (reach → carry → hold → return), the
// learned-grasp gate, carry attachment, and stepping toward the last async-
// predicted target — lives in mini-vla/rollout (RolloutEngine). Hero only
// drives it and paints the RolloutFrames it returns. LANG_MS is the shared
// throttle for the language + vision-gaze readouts (CONFIG.rollout).
const LANG_MS = CONFIG.rollout.langMs;

// First-line stall watchdog, handed to the package as replayWatchdogMs: how
// long the real WebGL run may sit in "loading" (Language Warmup) without a first
// training batch before the package swaps in the CPU replay. It overrides the
// package default (CONFIG.replay.watchdogMs, 7.5s). Lowered to 6s: healthy
// devices reach a first batch in ≤5s and clear the watchdog before it fires, so
// this only ever affects a STALLED device (the iOS/iPadOS dead-context wedge) —
// where 6s gets the replay going ~1.5s sooner than the default. The ~1s margin
// above the observed ≤5s warm-up keeps a merely-slow-but-healthy run (cold
// tfjs, thermal throttle, low-end GPU) from being swapped to the replay before
// its own first batch lands — a false swap costs that device its genuine live
// training. Don't drop below ~5.5s.
const REPLAY_WATCHDOG_MS = 6_000;

// How long "loading" (Language Warmup) may run before the host gives up. With
// replayFallback on, the package itself owns first-line load-stuck recovery: on
// a stall it swaps to the replay at REPLAY_WATCHDOG_MS (6s), and the replay's
// own load (tfjs-cpu + embeddings + checkpoints, ~1–2s) reaches "training"
// around ~7.5s. So this host watchdog is no longer the first responder.
// When it trips it splits on which run is still loading (see onLoadStuck): a
// real run that never swapped (usingReplay === false) is genuinely wedged and
// gets torn down here; a swap already underway (usingReplay === true) is the
// replay loading, so it hands off to a second, longer ceiling
// (REPLAY_LOAD_WATCHDOG_MS) rather than either killing a healthy swap or
// standing down forever. 15s gives that ~7.5s replay-ready path clear headroom
// before the net trips. On real hardware a healthy live warm-up is a few
// seconds — 200 text-only batches that early-stop the moment the color head
// plateaus (~tens of steps), which a GPU eats easily — so 15s still misfires
// only on a device already having a bad time.
const LOADING_WATCHDOG_MS = 15_000;

// Second-tier load ceiling, measured from the first watchdog's trip, that bounds
// the REPLAY's own load once the package has swapped to it. The replay fetches
// its manifest + checkpoint ladder with plain, timeout-free fetch() calls
// (mini-vla trainer.replay.ts), so a response body that stalls without
// rejecting — the flaky-mobile network this fallback exists for — would pin the
// run in "loading" forever with no error to surface. This is the only net for
// that. The healthy replay load is a ~300KB ladder plus tfjs-cpu init, seconds
// even on a slow link, so 20s past the first watchdog (~35s from Start) clears
// every working case and only a true stall trips it.
const REPLAY_LOAD_WATCHDOG_MS = 20_000;

// A backgrounded hero (tab hidden or scrolled off-screen) keeps pausing
// gradient steps as before, but after this grace period we go further and
// RELEASE the worker entirely — terminating it returns its WebGL context to the
// browser's process-wide pool. That pool is the scarce resource behind the iPad
// wedge (a fresh context can be evicted on arrival when too many live or
// bfcache'd contexts exist), so a context pinned behind a suspended tab is
// exactly what starves the next visit. 10s is long enough that a quick
// tab-switch doesn't nuke a run, short enough to free the context promptly.
const IDLE_TEARDOWN_GRACE_MS = 10_000;

// Training-progress watchdog: if no new batch lands within this window while the
// status still reads "training", the worker has gone silent mid-run — the same
// dead-context failure as the loading wedge, just after training started. A
// healthy batch is well under a second, and even the slow cpu-backend fallback
// stays far under this, so only a genuine stall trips it.
const TRAIN_STALL_MS = 20_000;

// Consecutive batches whose action loss reads non-physical (0 or non-finite)
// before we declare the run dead. A live Huber action loss for this task floors
// around ~0.012 and is never exactly 0; a run of zeros is the zeroed GPU
// readback of a lost WebGL context. Catching it here stops the worthless policy
// from silently reaching the batch-107 false-convergence (minBatches 100 +
// streak 8, all satisfied by a zeroed loss).
const DEAD_LOSS_LIMIT = 8;

// A genuinely converged action loss lands ~0.012–0.015 (the convergence
// threshold itself). Anything at or below this floor — or non-finite — at the
// moment the worker declares "converged" is a zeroed-readback false convergence
// against a dead context, not a trained policy: reject it as a failure.
const CONVERGED_LOSS_FLOOR = 1e-4;

// Host-detected trainer failures mini-vla itself has no signal for — all three
// trace to the browser losing the worker's WebGL context. "load-stuck": the
// worker went silent during Language Warmup (context died on arrival, tf.ready
// hangs). "train-stalled": batches stopped landing mid-run. "train-collapsed":
// the loss went non-physical (zeroed readback), including a false convergence.
// All three are process-wide GPU-context exhaustion, so the only reliable way
// out is a full page reload — the bar offers exactly that. Kept separate from
// TrainerError (mini-vla's own closed union): this is host-side detection of
// failures the worker cannot report from inside the context it lost.
type HostFailure = "load-stuck" | "train-stalled" | "train-collapsed";

const fmtAngle = (v: number) => `${v >= 0 ? "+" : "−"}${Math.abs(v).toFixed(2)}`;

// Below this width the ring can't fit around the name: the hero is a plain
// centered intro until the viewer opens the demo, which stacks the same five
// cards vertically. Kept in sync with the `max-width: 1099px` block in
// globals.css — layoutWires needs to know which geometry it is drawing for.
const STACKED_MQ = "(max-width: 1099px)";

// What gets trained is no longer the viewer's choice (there is no ⚙ menu): the
// viewport picks it. Desktop trains the full task; a phone trains a smaller one
// — fewer colors, sparser scenes — because it pays for every gradient step in
// battery and heat.
//
// mini-vla cannot make this call itself. The trainer runs in a Web Worker, and
// a Worker has no `matchMedia`, so the host resolves the profile and ships it
// through `trainer.start(onUpdate, cfg)` — which also installs it on this
// thread's samplers via setRunConfig (the two threads hold separate copies).
//
// The two profiles and the "which colors does this profile train on" rule are
// the package's to define (DESKTOP_RUN_CONFIG / MOBILE_RUN_CONFIG /
// activePalette); the host only chooses between them. Re-deriving the palette
// here as COLORS.slice(0, numColors) was a second copy of a rule that lives in
// examples.ts, and it would silently drift the moment the package changed it.

/** Resize a canvas to its CSS box at devicePixelRatio; returns a cleared ctx. */
function fitCanvas(c: HTMLCanvasElement, fallbackW = 190, fallbackH = 186) {
  // capped at 2: phones report DPR 3, which costs 2.25x the pixels per frame
  // with nothing visible to show for it in panels this small
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const W = c.offsetWidth || fallbackW;
  const H = c.offsetHeight || fallbackH;
  if (c.width !== Math.round(W * dpr)) c.width = Math.round(W * dpr);
  if (c.height !== Math.round(H * dpr)) c.height = Math.round(H * dpr);
  const ctx = c.getContext("2d")!;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, W, H);
  return { ctx, W, H };
}

const SIL_RENDER = CONFIG.rollout.silRender; // silhouette render size before the imgSize downsample

export default function Hero() {
  const stageRef = useRef<HTMLElement>(null);
  const flowRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLDivElement>(null);
  const promptRef = useRef<HTMLDivElement>(null);
  const visionCardRef = useRef<HTMLDivElement>(null);
  const langCardRef = useRef<HTMLDivElement>(null);
  const actionCardRef = useRef<HTMLDivElement>(null);
  const outputCardRef = useRef<HTMLDivElement>(null);
  const demoRef = useRef<HTMLCanvasElement>(null);
  const visionRef = useRef<HTMLCanvasElement>(null);
  const armRef = useRef<HTMLCanvasElement>(null);
  const lossRef = useRef<HTMLCanvasElement>(null);
  const actionValsRef = useRef<HTMLDivElement>(null);

  const trainerRef = useRef<VLATrainer | null>(null);
  const statusRef = useRef<TrainerStatus>("idle");
  // the DISPLAY pose for the non-episode states (idle / converged-waiting sway,
  // not-ready rest). The rollout's own arm pose during an episode lives inside
  // the RolloutEngine and comes back on each RolloutFrame.
  const armState = useRef({ a1: REST[0], a2: REST[1] });
  // the whole policy-rollout state machine (episode, arm, target, gaze, trail,
  // predict throttle/in-flight guard) — Hero drives it and paints its frames.
  const engineRef = useRef<RolloutEngine | null>(null);
  if (engineRef.current === null) engineRef.current = new RolloutEngine();
  // live-model attention over the DEMONSTRATION state — the Vision Encoder
  // panel's heatmap (ATTN_GRID² weights, peak-normalized by the trainer)
  const visGazeRef = useRef<number[] | null>(null);
  const lastHudRef = useRef(0);
  const lastLangRef = useRef(0);
  const lastVisGazeRef = useRef(0);
  // in-flight guards for the async trainer-worker round-trips: never queue a
  // second language-readout / gaze request while one is outstanding. On a slow
  // GPU a request can take longer than its throttle interval — an unbounded
  // FIFO backlog in the worker would starve training entirely. (The rollout's
  // own predict guard lives inside the RolloutEngine.)
  const langInFlightRef = useRef(false);
  const visGazeInFlightRef = useRef(false);
  // paused-duration accounting so the demo cycle resumes exactly where it
  // left off instead of jumping ahead by the real wall-clock pause length
  const pausedAccumRef = useRef(0);
  const pauseStartRef = useRef<number | null>(null);
  // wall-clock origin of the demonstration trajectory clock. Anchored to the
  // click (not the huge performance.now uptime) so the first cycle begins at
  // t=0 — the resting default demonstration already on screen — instead of
  // snapping into a random mid-reach. Offset +500ms so the arm holds that
  // initial pose for half a second before it starts moving (no harsh jump on
  // click). Pause duration is folded in via pausedAccumRef, not this.
  const trainStartRef = useRef(0);

  // demonstration state — a fresh layout/sentence/trajectory every cycle
  const demoLayoutRef = useRef(DEFAULT_LAYOUT);
  const demoSentenceRef = useRef<Sentence>(DEFAULT_SENTENCE);
  const demoPlanRef = useRef<DemoPlan | null>(null);
  const lastCycleRef = useRef(-1);
  const demoPoseRef = useRef({
    a1: REST[0],
    a2: REST[1],
    carry: null as number | null,
    grip: 0 as 0 | 1,
  });

  // rollout state
  const rolloutLayoutRef = useRef(DEFAULT_LAYOUT);
  const activeTokensRef = useRef<number[]>(DEFAULT_SENTENCE.tokens);
  const userSentenceRef = useRef<Sentence | null>(null);
  // which demo cycle the current training rollout episode belongs to — the
  // rollout re-syncs to the demonstration (same scene + command, fresh
  // attempt) whenever this falls behind the demo's cycle counter
  const rolloutCycleRef = useRef(-1);

  // offscreen canvases: the literal silhouette pipeline the model sees
  const silRef = useRef<HTMLCanvasElement | null>(null);
  const silThumbRef = useRef<HTMLCanvasElement | null>(null);
  // Current theme colours for the canvases, read every frame by the rAF loop.
  // Updated by the effect below on OS change or a nav-toggle data-theme flip.
  const canvasThemeRef = useRef<CanvasTheme>(LIGHT_CANVAS);

  const [status, setStatus] = useState<TrainerStatus>("idle");
  const [hud, setHud] = useState({ lossText: "—", samples: 0, batches: 0 });
  // The task profile this run trains on, resolved from the viewport. Desktop is
  // the SSR default: the bar is display:none below the breakpoint until the
  // viewer opens the demo, so the post-mount correction is never seen. Mirrored
  // into a ref because the rAF-loop closures (demo-cycle command sampling) mount
  // once and would otherwise capture the initial value.
  const [runCfg, setRunCfgState] = useState<RunConfig>(DESKTOP_RUN_CONFIG);
  const runCfgRef = useRef<RunConfig>(DESKTOP_RUN_CONFIG);
  // Latched from Start until Reset. setRunConfig installs per-thread module
  // state, so letting the profile follow the media query mid-run would leave
  // this thread's randomLayout() sampling scenes the worker isn't training on.
  const cfgLockedRef = useRef(false);
  const syncRunCfg = useCallback(() => {
    if (cfgLockedRef.current) return;
    const next = window.matchMedia(STACKED_MQ).matches
      ? MOBILE_RUN_CONFIG
      : DESKTOP_RUN_CONFIG;
    runCfgRef.current = next;
    setRunCfgState(next);
  }, []);
  // Resolve the profile on mount, and follow the breakpoint while idle — a
  // viewer who rotates a tablet before pressing Start gets the right task, and
  // the bar's readout never lies about what Start would train.
  useEffect(() => {
    const narrow = window.matchMedia(STACKED_MQ);
    syncRunCfg();
    narrow.addEventListener("change", syncRunCfg);
    return () => narrow.removeEventListener("change", syncRunCfg);
  }, [syncRunCfg]);
  // demonstrations the CURRENTLY rolled-out policy has seen: the sample count
  // frozen into the per-cycle snapshot during training, or the final count
  // once converged (see drawArm's snapshot boundary + onUpdate).
  const [rolloutSamples, setRolloutSamples] = useState(0);
  const [demoSentence, setDemoSentence] = useState<Sentence>(DEFAULT_SENTENCE);
  const [userSentence, setUserSentence] = useState<Sentence | null>(null);
  const [tryText, setTryText] = useState("");
  const chipRowRef = useRef<HTMLDivElement | null>(null);
  const [tokenBars, setTokenBars] = useState<number[]>([]);
  const [decoded, setDecoded] = useState<{
    name: string;
    hex: string;
    prob: number;
  } | null>(null);
  const [tryNote, setTryNote] = useState<string | null>(null);
  // Vision Encoder panel: toggle between the host view and the inverted tensor
  // the CNN receives. Hover handles it on desktop (CSS); this drives the tap
  // toggle on touch. Which view is the RESTING one flips per theme (dark rests
  // on the inverted view — see --vision-*-invert in globals.css), so this is
  // just "flipped from the theme default", not "showing the model input".
  const [modelView, setModelView] = useState(false);

  // Below STACKED_MQ the pipeline is hidden behind a "Mini VLA Demo" CTA and
  // unrolls into a vertical stack when opened. Mirrored into a ref because the
  // mount-once rAF loop needs it every frame (path geometry + draw skipping).
  // Above the breakpoint this state is inert: the CTA and ✕ are display:none
  // and every `demo-open` rule lives inside the mobile media query.
  const [showDemo, setShowDemo] = useState(false);
  const showDemoRef = useRef(false);
  // Auto-pause bookkeeping (tab hidden / hero scrolled off screen). Only a
  // pause WE initiated may be auto-resumed — a user's manual Pause is sacred.
  const autoPausedRef = useRef(false);
  const heroOnScreenRef = useRef(true);
  // Why the trainer is in "error" (see onUpdate). Decides the recovery the bar
  // offers: "worker" (dead chunk URL) and "context" (lost WebGL context) can
  // only be escaped by a reload; everything else can be retried in place.
  const [errorReason, setErrorReason] = useState<TrainerError | null>(null);
  // Why the run failed on the host side (null while healthy). See HostFailure:
  // all three are the browser losing the worker's WebGL context, detected here
  // because the worker can't report a failure from inside the context it lost.
  const [hostFailure, setHostFailure] = useState<HostFailure | null>(null);
  // Mirror of trainer.usingReplay: true once the package has transparently
  // swapped the live WebGL run for the CPU-backend replay (the iOS/iPadOS
  // path). Drives the small "replay" chip — honesty that this device is showing
  // a captured-policy replay, not live training. Folded into onUpdate's mirror.
  const [usingReplay, setUsingReplay] = useState(false);
  // The chip's disclosure text is behind a tap, not a hover title, since the
  // devices that land here (iPad/iOS) are touch-only and never see a title
  // tooltip. Reset whenever the replay flag itself changes off.
  const [showReplayInfo, setShowReplayInfo] = useState(false);
  // ---- guidance layer state ----
  // which ⓘ popover is open — one at a time across the whole hero
  const [openInfo, setOpenInfo] = useState<InfoId | null>(null);
  const toggleInfo = useCallback(
    (id: InfoId) => setOpenInfo((v) => (v === id ? null : id)),
    []
  );
  // The ⓘ layer exists only while the pipeline is actually running:
  // training/paused shows it on every card, everything else drops the chrome.
  // The warmup gets none (it is over in seconds — no time to read anything),
  // dormant idle boxes need no footnotes, and a converged hero belongs to
  // the try-row.
  const infoVisible =
    status === "training" || status === "paused";
  // one-shot red flash on the idle hero's way in, after ~3s of looking at it:
  // the demo CTA on mobile, the Start Training button on desktop — the same
  // state drives both, since only one of them is ever visible at idle
  const [flashCta, setFlashCta] = useState(false);
  // the converged twin: flash the prompt box after ~3s of converged
  // inactivity. triedRef = the viewer has already touched the try-row
  // (typed, ran, dragged, shuffled) — no nudge needed.
  const [flashTry, setFlashTry] = useState(false);
  const triedRef = useRef(false);
  // the one retirement path for that nudge — every try-row interaction goes
  // through here, so a future interaction site can't forget half the pair
  const retireTryNudge = useCallback(() => {
    triedRef.current = true;
    setFlashTry(false);
  }, []);
  // progress-keyed caption under the bar; the stage counter is forward-only
  // so a late-arriving trigger can never step the narration backwards
  const [caption, setCaption] = useState<string | null>(null);
  const captionStageRef = useRef(0);
  // converged-unlock glow + autofocus bookkeeping
  const [justConverged, setJustConverged] = useState(false);
  const prevStatusForUnlockRef = useRef<TrainerStatus>("idle");
  const tryInputRef = useRef<HTMLInputElement>(null);
  // rotating try-box placeholder index
  const [phIdx, setPhIdx] = useState(0);
  // post-run tip chain, shown one per successful episode in the try-note slot
  const [tip, setTip] = useState<string | null>(null);
  const tipsShownRef = useRef<Set<number>>(new Set());
  const userDraggedRef = useRef(false); // viewer already found block-dragging
  const userShuffledRef = useRef(false); // viewer already found ⟳ shuffle
  // episode-completion detector for the tip chain: an episode was live, and
  // it reached "hold" (the lift succeeded) before it ended
  const episodeLiveRef = useRef(false);
  const episodeHeldRef = useRef(false);
  const loadWatchdogRef = useRef<number | null>(null);
  // Training-phase watchdog: re-armed on every batch, fires if batches stop
  // landing while status still reads "training". Cleared on pause/converge.
  const trainWatchdogRef = useRef<number | null>(null);
  // Consecutive non-physical (0 / non-finite) batch losses seen — a dead WebGL
  // context zeroes GPU readbacks, so a run of these means the policy is garbage.
  const deadLossRunRef = useRef(0);
  // Last batch index the training watchdog acted on (its progress detector).
  const lastWatchedBatchRef = useRef(0);
  // Grace-period timer that releases the worker after the hero stays
  // backgrounded (see IDLE_TEARDOWN_GRACE_MS).
  const graceTimerRef = useRef<number | null>(null);

  const setStatusBoth = (s: TrainerStatus) => {
    statusRef.current = s;
    setStatus(s);
  };

  // The one pause/resume mechanism. Wall-clock bookkeeping (pauseStartRef /
  // pausedAccumRef) keeps the demonstration cycle resuming exactly where it
  // left off instead of jumping ahead by the real pause duration. Every caller
  // — the bar button, closing the demo, the visibility/viewport guards — goes
  // through these two.
  const pauseTraining = useCallback(() => {
    const trainer = trainerRef.current;
    if (!trainer || trainer.status !== "training") return;
    trainer.pause();
    pauseStartRef.current = performance.now();
    // Disarm the stall watchdog for the pause's duration. Its deadline was
    // computed from the last batch BEFORE the pause, so left running it can
    // fire moments after a resume that lands just under that stale deadline —
    // tearing down a healthy, just-resumed run as a false "train-stalled".
    // resumeTraining's entry-arm (see the "Resume re-arm" comment below) only
    // re-arms when the ref is null, so clearing it here — not just on the
    // long-pause-already-fired path — makes that check correct for a short
    // pause too.
    if (trainWatchdogRef.current !== null) {
      window.clearTimeout(trainWatchdogRef.current);
      trainWatchdogRef.current = null;
    }
    flowRef.current
      ?.querySelectorAll<HTMLElement>(".vla-payload")
      .forEach((el) => {
        el.style.animationPlayState = "paused";
      });
    statusRef.current = "paused";
    setStatus("paused");
  }, []);

  const resumeTraining = useCallback(() => {
    const trainer = trainerRef.current;
    if (!trainer || trainer.status !== "paused") return;
    if (pauseStartRef.current !== null) {
      pausedAccumRef.current += performance.now() - pauseStartRef.current;
      pauseStartRef.current = null;
    }
    flowRef.current
      ?.querySelectorAll<HTMLElement>(".vla-payload")
      .forEach((el) => {
        el.style.animationPlayState = "";
      });
    trainer.resume();
    statusRef.current = "training";
    setStatus("training");
  }, []);

  // Keep the canvas theme in step with the OS preference AND the nav toggle's
  // data-theme override. The rAF loop reads canvasThemeRef every frame, so the
  // arm/loss/vision recolour on the next painted frame with no extra repaint.
  useEffect(() => {
    const apply = () => {
      canvasThemeRef.current = resolveCanvasTheme();
    };
    apply();
    const mq = window.matchMedia("(prefers-color-scheme: dark)");
    mq.addEventListener("change", apply);
    const obs = new MutationObserver(apply);
    obs.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ["data-theme"],
    });
    return () => {
      mq.removeEventListener("change", apply);
      obs.disconnect();
    };
  }, []);

  // Advance the narration to `stage` (forward-only). Stable identity — the
  // mount-once rAF closures call it.
  const advanceCaption = useCallback((stage: number, text: string) => {
    if (captionStageRef.current >= stage) return;
    captionStageRef.current = stage;
    setCaption(text);
  }, []);

  // Show the next unseen post-run tip; a tip whose action the viewer already
  // performed on their own is retired unshown. Persisted, so a returning
  // visitor is never re-toured.
  const advanceTip = useCallback(() => {
    const shown = tipsShownRef.current;
    let next: number | null = null;
    for (let i = 0; i < TIPS.length; i++) {
      if (shown.has(i)) continue;
      if (i === 0 && userDraggedRef.current) {
        shown.add(i);
        continue;
      }
      if (i === 1 && userShuffledRef.current) {
        shown.add(i);
        continue;
      }
      next = i;
      break;
    }
    if (next !== null) {
      shown.add(next);
      setTip(TIPS[next]);
    }
    const h = loadHints();
    h.tips = [...shown];
    saveHints(h);
  }, []);

  // ---- the single rAF loop: wires + all four canvases, every frame ----
  useEffect(() => {
    const reduced = window.matchMedia("(prefers-reduced-motion: reduce)");
    const narrow = window.matchMedia(STACKED_MQ);
    /** Cards are stacked vertically (mobile, demo open) rather than ringed. */
    const stacked = () => narrow.matches && showDemoRef.current;

    const layoutWires = () => {
      const stage = stageRef.current;
      const flow = flowRef.current;
      if (
        !stage ||
        !flow ||
        !inputRef.current ||
        !promptRef.current ||
        !visionCardRef.current ||
        !langCardRef.current ||
        !actionCardRef.current ||
        !outputCardRef.current
      )
        return;
      const R = stage.getBoundingClientRect();
      if (R.width === 0 || R.height === 0) return;
      // Card box in PX relative to the stage (offset-path lives in the flow
      // overlay's own px coordinate space, no viewBox stretching).
      const m = (el: Element) => {
        const c = el.getBoundingClientRect();
        return {
          cx: c.left - R.left + c.width / 2,
          cy: c.top - R.top + c.height / 2,
          left: c.left - R.left,
          right: c.right - R.left,
          top: c.top - R.top,
          bottom: c.bottom - R.top,
        };
      };
      const I = m(inputRef.current);
      const P = m(promptRef.current);
      const V = m(visionCardRef.current);
      const L = m(langCardRef.current);
      const A = m(actionCardRef.current);
      const O = m(outputCardRef.current);
      const r = (n: number) => n.toFixed(1);
      // No drawn connectors any more — a payload token glides across each gap,
      // so the "connection" is carried by motion. Each hop is a single cubic
      // Bézier that leaves the CENTRE of the source card's facing edge and
      // lands on the CENTRE of the target's facing edge. `sDir`/`eDir` lock the
      // tangent at each end to that edge's normal ("h" = horizontal, "v" =
      // vertical), so a token slides straight out of one card and straight into
      // the next, sweeping a quarter-turn in between when the two differ.
      type Dir = "h" | "v";
      const arc = (
        sx: number,
        sy: number,
        sDir: Dir,
        ex: number,
        ey: number,
        eDir: Dir
      ) => {
        const dx = (ex - sx) * 0.5;
        const dy = (ey - sy) * 0.5;
        const [c1x, c1y] = sDir === "h" ? [sx + dx, sy] : [sx, sy + dy];
        const [c2x, c2y] = eDir === "h" ? [ex - dx, ey] : [ex, ey - dy];
        return `path("M ${r(sx)} ${r(sy)} C ${r(c1x)} ${r(c1y)}, ${r(c2x)} ${r(
          c2y
        )}, ${r(ex)} ${r(ey)}")`;
      };
      // Stacked hops run straight down the column the payload is feeding, so
      // each token visibly drops out of one card and into the next. The Action
      // Head takes its two inputs at the thirds of its top edge (vision left,
      // language right) rather than both at dead centre.
      const aw = A.right - A.left;
      const paths: Record<string, string> = stacked()
        ? {
            p1: arc(V.cx, I.bottom, "v", V.cx, V.top, "v"),
            // the prompt is folded inside the Demonstration card on mobile, so
            // "prompt → language" leaves from the card's bottom edge, under it
            p2: arc(L.cx, I.bottom, "v", L.cx, L.top, "v"),
            p3: arc(V.cx, V.bottom, "v", A.left + aw * 0.32, A.top, "v"),
            p4: arc(L.cx, L.bottom, "v", A.left + aw * 0.68, A.top, "v"),
            p5: arc(A.cx, A.bottom, "v", O.cx, O.top, "v"),
          }
        : // Side-by-side: Vision sits above Demonstration and Language below
          // Prompt, so the first pair exits vertically and enters horizontally;
          // the encoders then hand off horizontally into the Action Head's top
          // and bottom edges. Action → Rollout is a near-straight sidestep.
          {
            p1: arc(I.cx, I.top, "v", V.left, V.cy, "h"),
            p2: arc(P.cx, P.bottom, "v", L.left, L.cy, "h"),
            p3: arc(V.right, V.cy, "h", A.cx, A.top, "v"),
            p4: arc(L.right, L.cy, "h", A.cx, A.bottom, "v"),
            p5: arc(A.right, A.cy, "h", O.left, O.cy, "h"),
          };
      flow.querySelectorAll<HTMLElement>("[data-flow]").forEach((el) => {
        const d = paths[el.dataset.flow ?? ""];
        if (d && el.style.offsetPath !== d) el.style.offsetPath = d;
      });
    };

    // the Action Head's live output — the policy's current predicted target
    // joint angles plus the gripper state (open/closed). null angles → em
    // dashes; grip undefined → em dash, 0 → open, 1 → closed.
    const setActionVals = (
      t: [number, number] | null,
      grip?: 0 | 1 | undefined
    ) => {
      const el = actionValsRef.current;
      if (!el || el.children.length < 3) return;
      const v0 = el.children[0].lastElementChild;
      const v1 = el.children[1].lastElementChild;
      const v2 = el.children[2].lastElementChild;
      if (v0) v0.textContent = t ? fmtAngle(t[0]) : "—";
      if (v1) v1.textContent = t ? fmtAngle(t[1]) : "—";
      if (v2)
        v2.textContent = grip === 1 ? "closed" : grip === 0 ? "open" : "—";
    };

    // small idle sway around the rest pose — "there is something here"
    const wiggle = (now: number, p1: number, p2: number): [number, number] =>
      reduced.matches
        ? REST
        : [
            REST[0] + 0.05 * Math.sin(now * 0.0011 + p1),
            0.07 * Math.sin(now * 0.0008 + p2),
          ];

    const drawDemo = (now: number) => {
      const c = demoRef.current;
      if (!c) return;
      const { ctx, W, H } = fitCanvas(c);
      const st = statusRef.current;

      if (st === "paused") {
        // frozen mid-cycle: redraw the last computed pose, advance nothing
        const p = demoPoseRef.current;
        paintScene(ctx, W, H, {
          a1: p.a1,
          a2: p.a2,
          layout: demoLayoutRef.current,
          accent: ACCENT,
          carry: p.carry,
          grip: p.grip,
          palette: canvasThemeRef.current.palette,
        });
        return;
      }

      let a1: number;
      let a2: number;
      let carry: number | null = null;
      let grip: 0 | 1 | undefined;

      if (st === "training") {
        // clock measured from the moment the warm-up handed off to training
        // (trainStartRef), minus accumulated paused time so the cycle resumes
        // exactly where it left off
        const effectiveNow = now - trainStartRef.current - pausedAccumRef.current;
        if (effectiveNow < 0) {
          // the half-second hold right after the handoff: sit on the initial
          // demonstration (t=0 → resting pose over the default layout) so the
          // pipeline eases in instead of the arm snapping into motion
          if (!demoPlanRef.current)
            demoPlanRef.current = makeDemoPlan(
              demoLayoutRef.current,
              demoSentenceRef.current
            );
          const pose = demoPose(demoPlanRef.current, 0);
          demoPoseRef.current = {
            a1: pose.a1,
            a2: pose.a2,
            carry: pose.carry,
            grip: pose.grip,
          };
          paintScene(ctx, W, H, {
            a1: pose.a1,
            a2: pose.a2,
            layout: demoLayoutRef.current,
            accent: ACCENT,
            carry: pose.carry,
            grip: pose.grip,
            palette: canvasThemeRef.current.palette,
          });
          return;
        }
        const cycle = Math.floor(effectiveNow / DEMO_PERIOD_MS);
        if (cycle !== lastCycleRef.current || !demoPlanRef.current) {
          const first = lastCycleRef.current === -1;
          lastCycleRef.current = cycle;
          if (!first) {
            demoLayoutRef.current = randomLayout();
            // a pick-up command naming one of the scene's own blocks
            const s = sampleCommand(demoLayoutRef.current);
            demoSentenceRef.current = s;
            setDemoSentence(s);
            // the language panel follows the demo until a user command locks it
            if (!userSentenceRef.current) activeTokensRef.current = s.tokens;
          }
          demoPlanRef.current = makeDemoPlan(
            demoLayoutRef.current,
            demoSentenceRef.current
          );
        }

        const t = reduced.matches
          ? 0.2
          : (effectiveNow % DEMO_PERIOD_MS) / DEMO_PERIOD_MS;
        const plan = demoPlanRef.current;
        const pose = demoPose(plan, t);
        a1 = pose.a1;
        a2 = pose.a2;
        carry = pose.carry;
        grip = pose.grip;
      } else {
        // idle, language warm-up OR converged: no demonstrations are being
        // consumed — the arm rests in its sway (the CSS also holds the box at
        // its dormant, pre-training transparency). During the warm-up only the
        // language twin is training, so the demonstration has nothing to show
        // yet. A resting gripper is OPEN (including before training starts).
        [a1, a2] = wiggle(now, 0, 1.7);
        grip = 0;
      }

      demoPoseRef.current = { a1, a2, carry, grip: grip ?? 0 };
      paintScene(ctx, W, H, {
        a1,
        a2,
        layout: demoLayoutRef.current,
        accent: ACCENT,
        carry,
        grip: grip ?? 0,
        palette: canvasThemeRef.current.palette,
      });
    };

    // The policy's gaze on the Rollout scene: the soft-argmax of the spatial
    // attention map that rode along with the last prediction. xy is in the
    // SILHOUETTE's [0,1] image coords — invert that renderer's sceneMap back
    // to workspace units, then forward-map into this canvas. Drawn only while
    // a policy-driven phase is live (that's when the prediction is current).
    const drawGaze = (
      ctx: CanvasRenderingContext2D,
      W: number,
      H: number,
      gaze: [number, number] | null,
      phase: RolloutPhase | null
    ) => {
      if (!gaze || (phase !== "reach" && phase !== "carry")) return;
      const sc = CONFIG.render.sceneScale;
      const wx = 0.5 + (gaze[0] - 0.5) / sc;
      const wy = (CONFIG.render.floorY - gaze[1]) / sc;
      const m = sceneMap(W, H);
      const px = m.X(wx);
      const py = m.Y(wy);
      const rad = 14;
      const grad = ctx.createRadialGradient(px, py, 0, px, py, rad);
      grad.addColorStop(0, "rgba(225,45,26,0.30)");
      grad.addColorStop(1, "rgba(225,45,26,0)");
      ctx.fillStyle = grad;
      ctx.beginPath();
      ctx.arc(px, py, rad, 0, 7);
      ctx.fill();
      ctx.strokeStyle = "rgba(225,45,26,0.75)";
      ctx.lineWidth = 1.2;
      ctx.beginPath();
      ctx.arc(px, py, 4.5, 0, 7);
      ctx.stroke();
    };

    // The async policy call the engine steps against: the FROZEN per-cycle
    // snapshot (training) / final weights (converged). The engine keeps stepping
    // toward its last target until each reply lands (see RolloutEngine.step).
    const predictFrozen = (
      a1: number,
      a2: number,
      tokens: number[],
      layout: Layout,
      carry: number | null
    ) =>
      trainerRef.current
        ? trainerRef.current.predictFrozenTarget(a1, a2, tokens, layout, carry)
        : Promise.resolve(null);

    // The engine keeps its trail in workspace effector coords (renderer-neutral);
    // map to canvas px here — identical to the old effectorPx pushes, since
    // effectorPx === sceneMap ∘ fk.
    const trailToPx = (
      trail: { x: number; y: number }[],
      W: number,
      H: number
    ) => {
      const m = sceneMap(W, H);
      return trail.map((p) => ({ x: m.X(p.x), y: m.Y(p.y) }));
    };

    // rollout: during training the episode runs in LOCKSTEP with the
    // demonstration — same scene + command, restarted each demo cycle — so
    // the two can be compared side by side; converged runs user commands;
    // paused freezes it.
    const drawArm = (now: number) => {
      const c = armRef.current;
      if (!c) return;
      const { ctx, W, H } = fitCanvas(c);
      const trainer = trainerRef.current;
      const arm = armState.current;
      const st = statusRef.current;
      const engine = engineRef.current!;

      if (st === "paused") {
        // frozen: redraw the engine's current pose, advance nothing (matches
        // the demo). The rollout arm lives in the engine now, so the frozen pose
        // is engine.frame() — NOT the wiggling armState.
        const f = engine.frame();
        paintScene(ctx, W, H, {
          a1: f.a1,
          a2: f.a2,
          layout: rolloutLayoutRef.current,
          accent: ACCENT,
          carry: f.carry,
          grip: f.grip,
          palette: canvasThemeRef.current.palette,
        });
        drawGaze(ctx, W, H, f.gaze, f.phase);
        setActionVals(f.target, f.grip);
        return;
      }

      // f = the engine frame when an episode is (or was) live this frame; null
      // for the non-episode display states (idle / converged-waiting sway,
      // not-ready rest) that paint the wiggling armState with an empty rollout.
      let f: RolloutFrame | null = null;

      if (st === "idle") {
        [arm.a1, arm.a2] = wiggle(now, 2.6, 4.1);
      } else if (trainer?.ready) {
        if (st === "converged") {
          if (engine.hasEpisode) {
            f = engine.step(now, rolloutLayoutRef.current, predictFrozen);
            episodeLiveRef.current = true;
            // "hold" = the block is aloft: the lift succeeded, whatever ends
            // the episode after this (return finishing, a drag, a reset)
            if (f.phase === "hold") episodeHeldRef.current = true;
          } else {
            // an episode just ended — if it got as far as the hold, the run
            // paid off, so this is the moment for the next post-run tip
            if (episodeLiveRef.current) {
              episodeLiveRef.current = false;
              if (episodeHeldRef.current) {
                episodeHeldRef.current = false;
                advanceTip();
              }
            }
            [arm.a1, arm.a2] = wiggle(now, 2.6, 4.1); // waiting for command
          }
        } else {
          // training: a fresh synced attempt at every new demonstration cycle,
          // run against a policy snapshot frozen at this boundary so the whole
          // attempt reflects one policy generation (not a live-drifting target)
          if (rolloutCycleRef.current !== lastCycleRef.current) {
            rolloutCycleRef.current = lastCycleRef.current;
            // per-block CLONE of the demo scene, not a shared reference: the
            // viewer can drag the rollout's blocks once converged, so sharing
            // objects would leak one box's positions into the other
            rolloutLayoutRef.current = demoLayoutRef.current.map((b) => ({ ...b }));
            trainer.snapshotPolicy();
            // the snapshot freezes the policy at THIS many seen demonstrations —
            // surface it on the Rollout so the attempt is read as "this is what
            // N examples of training buys you" (updates once per demo cycle)
            setRolloutSamples(trainer.samples);
            // cycle 0 lands together with the training handoff — hold this
            // caption for the SECOND synced attempt, after the reader has
            // actually seen the rollout try once. Gated on the gaze caption
            // having shown (stage 2): its trigger is an async worker reply,
            // and advanceCaption is forward-only, so firing 3 first would
            // skip 2 for good. This trigger repeats at EVERY cycle boundary,
            // so a slow gaze reply just defers this line one cycle.
            if (lastCycleRef.current >= 1 && captionStageRef.current >= 2)
              advanceCaption(3, CAPTION_ROLLOUT);
            engine.begin(
              demoSentenceRef.current.color,
              demoSentenceRef.current.tokens
            );
          }
          // step the live attempt, or hold at REST (engine.frame()) between the
          // end of one attempt and the next cycle's re-sync
          f = engine.hasEpisode
            ? engine.step(now, rolloutLayoutRef.current, predictFrozen)
            : engine.frame();
        }
      } else {
        arm.a1 = REST[0];
        arm.a2 = REST[1];
      }

      // single paint: from the engine frame when an episode is live, else the
      // wiggling/rest display pose with an empty rollout state.
      paintScene(ctx, W, H, {
        a1: f ? f.a1 : arm.a1,
        a2: f ? f.a2 : arm.a2,
        layout: rolloutLayoutRef.current,
        accent: ACCENT,
        trail:
          f && (f.phase === "reach" || f.phase === "carry")
            ? trailToPx(f.trail, W, H)
            : null,
        lossNorm: trainer?.lossNorm() ?? 1,
        carry: f ? f.carry : null,
        grip: f ? f.grip : 0,
        palette: canvasThemeRef.current.palette,
      });
      drawGaze(ctx, W, H, f ? f.gaze : null, f ? f.phase : null);
      setActionVals(st === "idle" ? null : f ? f.target : null, f ? f.grip : 0);
    };

    const drawVision = (now: number) => {
      const c = visionRef.current;
      if (!c) return;
      // the encoder is dormant on both sides of the run: during the language
      // warm-up the vision branch is not in the graph at all, and once trained
      // it has no further role. Either way, stop feeding it the demo silhouette
      // (the CSS fades the canvas back to transparent). The last frame stays
      // underneath the fade-out, which is fine — it's hidden.
      if (statusRef.current === "converged" || statusRef.current === "loading")
        return;
      const { ctx, W } = fitCanvas(c, 176, 176);

      // offscreen silhouette pipeline (the literal model input view)
      if (!silRef.current) {
        silRef.current = document.createElement("canvas");
        silRef.current.width = SIL_RENDER;
        silRef.current.height = SIL_RENDER;
        silThumbRef.current = document.createElement("canvas");
        silThumbRef.current.width = IMG_SIZE;
        silThumbRef.current.height = IMG_SIZE;
      }
      const sil = silRef.current;
      const thumb = silThumbRef.current!;
      const p = demoPoseRef.current;
      paintSilhouette(
        sil.getContext("2d")!,
        SIL_RENDER,
        p.a1,
        p.a2,
        demoLayoutRef.current,
        p.carry
      );
      const tctx = thumb.getContext("2d", { willReadFrequently: true })!;
      tctx.imageSmoothingEnabled = true;
      tctx.clearRect(0, 0, IMG_SIZE, IMG_SIZE);
      tctx.drawImage(sil, 0, 0, SIL_RENDER, SIL_RENDER, 0, 0, IMG_SIZE, IMG_SIZE);

      // 32x32 input, blown up pixelated
      ctx.imageSmoothingEnabled = false;
      ctx.drawImage(thumb, 0, 0, IMG_SIZE, IMG_SIZE, 0, 0, W, W);

      // "where the model looks": the live model's spatial attention over THIS
      // demonstration state, as a per-cell alpha overlay (peak-normalized by
      // the trainer). Early in training it's a diffuse smear; it visibly
      // sharpens onto the commanded block as the policy learns to bind the
      // command to the scene.
      const gaze = visGazeRef.current;
      if (gaze && statusRef.current !== "idle") {
        const G = Math.round(Math.sqrt(gaze.length));
        const cell = W / G;
        for (let i = 0; i < G; i++)
          for (let j = 0; j < G; j++) {
            const a = gaze[i * G + j];
            if (a < 0.04) continue;
            ctx.fillStyle = `rgba(225,45,26,${(0.45 * a).toFixed(3)})`;
            ctx.fillRect(j * cell, i * cell, cell + 0.5, cell + 0.5);
          }
      }

      ctx.strokeStyle = canvasThemeRef.current.visionBorder;
      ctx.lineWidth = 1;
      ctx.strokeRect(0.5, 0.5, W - 1, W - 1);

      // refresh the gaze on a throttle (same cadence as the language panel),
      // against the demonstration's current pose + command + carry state, on
      // the LIVE model — this is training-progress chrome, like the decoded-
      // command readout. One request in flight, ever (see the in-flight
      // guards note above).
      const trainer = trainerRef.current;
      if (
        trainer?.ready &&
        statusRef.current === "training" &&
        now - lastVisGazeRef.current > LANG_MS &&
        !visGazeInFlightRef.current
      ) {
        lastVisGazeRef.current = now;
        visGazeInFlightRef.current = true;
        const p = demoPoseRef.current;
        void trainer
          .predictLive(
            p.a1,
            p.a2,
            demoSentenceRef.current.tokens,
            demoLayoutRef.current,
            p.carry
          )
          .then((r) => {
            if (r) {
              visGazeRef.current = r.attn;
              // the gaze overlay just became visible — point the reader at it
              advanceCaption(2, CAPTION_GAZE);
            }
          })
          // clear the guard even if the round-trip rejects — otherwise a single
          // failed request would freeze the gaze overlay for the rest of the session
          .finally(() => {
            visGazeInFlightRef.current = false;
          });
      }
    };

    const drawLossCurve = () => {
      const c = lossRef.current;
      if (!c) return;
      const { ctx, W, H } = fitCanvas(c, 300, 34);
      const raw = trainerRef.current?.lossHistory ?? [];
      const pad = 3;

      ctx.strokeStyle = canvasThemeRef.current.lossBaseline;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(0, H - pad);
      ctx.lineTo(W, H - pad);
      ctx.stroke();
      if (raw.length < 2) return;

      // Smooth out the batch-to-batch noise so the curve shows the trend, not
      // the jitter: a trailing rolling mean over the last SMOOTH_WINDOW batches
      // (prefix sum → O(n)). The window shrinks near the start so the curve
      // still begins at the first real loss.
      const SMOOTH_WINDOW = 30;
      const prefix = [0];
      for (let i = 0; i < raw.length; i++) prefix.push(prefix[i] + raw[i]);
      const hist = raw.map((_, i) => {
        const lo = Math.max(0, i - SMOOTH_WINDOW + 1);
        return (prefix[i + 1] - prefix[lo]) / (i + 1 - lo);
      });

      const maxLoss = Math.max(trainerRef.current?.initialLoss ?? 0, ...hist);
      const n = hist.length;
      const x = (i: number) => (i / (n - 1)) * W;
      const y = (v: number) =>
        pad + (1 - Math.min(1, v / maxLoss)) * (H - pad * 2);

      ctx.beginPath();
      ctx.moveTo(0, y(hist[0]));
      hist.forEach((v, i) => ctx.lineTo(x(i), y(v)));
      ctx.lineTo(W, H - pad);
      ctx.lineTo(0, H - pad);
      ctx.closePath();
      ctx.fillStyle = ACCENT;
      ctx.globalAlpha = 0.08;
      ctx.fill();
      ctx.globalAlpha = 1;

      ctx.beginPath();
      ctx.moveTo(0, y(hist[0]));
      hist.forEach((v, i) => ctx.lineTo(x(i), y(v)));
      ctx.strokeStyle = ACCENT;
      ctx.lineWidth = 1.6;
      ctx.lineJoin = "round";
      ctx.lineCap = "round";
      ctx.stroke();

      ctx.fillStyle = ACCENT;
      ctx.beginPath();
      ctx.arc(x(n - 1), y(hist[n - 1]), 2.6, 0, 7);
      ctx.fill();
    };

    // real language-encoder readouts: decoded command (color) + each token's
    // live ATTENTION weight (see attentionWeights in trainer.core.ts — the
    // exact masked-softmax weights the attention-pooling layer uses,
    // recomputed from the linear scorer's weights, not an approximation).
    // Both are async round-trips to the trainer worker; LANG_MS throttling
    // means at most one pair is ever in flight, so replies apply in order.
    const langViz = (now: number) => {
      const trainer = trainerRef.current;
      if (!trainer?.ready || now - lastLangRef.current < LANG_MS) return;
      if (langInFlightRef.current) return;
      lastLangRef.current = now;
      langInFlightRef.current = true;
      const tokens = activeTokensRef.current;
      const decodeP = trainer.decodeCommand(tokens).then((d) => {
        if (!d) return;
        setDecoded({
          name: COLORS[d.color].name,
          hex: COLORS[d.color].hex,
          prob: d.colorProb,
        });
      });
      const attnP = trainer.attentionWeights(tokens).then((bars) => {
        if (bars) setTokenBars(bars);
      });
      // clear the guard in .finally so a rejected round-trip can't wedge it
      // true and freeze the language readouts for the rest of the session
      void Promise.all([decodeP, attnP]).finally(() => {
        langInFlightRef.current = false;
      });
    };

    let raf = 0;
    const loop = (now: number) => {
      // mobile, demo closed: every card is display:none, so painting them would
      // just burn a phone battery rasterising invisible fallback-sized canvases
      if (narrow.matches && !showDemoRef.current) {
        raf = requestAnimationFrame(loop);
        return;
      }
      layoutWires();
      drawDemo(now);
      drawArm(now);
      drawVision(now);
      drawLossCurve();
      langViz(now);
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);
    window.addEventListener("resize", layoutWires);
    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener("resize", layoutWires);
      trainerRef.current?.destroy(); // reset + terminate the trainer worker
      if (loadWatchdogRef.current !== null) window.clearTimeout(loadWatchdogRef.current);
      if (trainWatchdogRef.current !== null) window.clearTimeout(trainWatchdogRef.current);
    };
    // both are stable useCallbacks — the effect still mounts exactly once
  }, [advanceCaption, advanceTip]);

  // ---- teardown + host-side failure detection ----
  // Return every piece of local view state to its pristine idle values. Shared
  // by the manual Reset button and the worker-teardown paths below; the only
  // difference between them is whether the trainer is reset() (kept warm) or
  // destroy()ed (worker terminated, context released), which the callers do.
  const resetToIdle = useCallback(() => {
    autoPausedRef.current = false;
    // back to idle: the profile follows the viewport again (it may have been
    // resized across the breakpoint during the run we just threw away)
    cfgLockedRef.current = false;
    syncRunCfg();
    setErrorReason(null);
    setHostFailure(null);
    setUsingReplay(false);
    setShowReplayInfo(false);
    if (loadWatchdogRef.current !== null) {
      window.clearTimeout(loadWatchdogRef.current);
      loadWatchdogRef.current = null;
    }
    if (trainWatchdogRef.current !== null) {
      window.clearTimeout(trainWatchdogRef.current);
      trainWatchdogRef.current = null;
    }
    deadLossRunRef.current = 0;
    lastWatchedBatchRef.current = 0;
    engineRef.current!.reset();
    rolloutCycleRef.current = -1;
    visGazeRef.current = null;
    armState.current = { a1: REST[0], a2: REST[1] };
    // clones — the rollout's blocks are draggable; DEFAULT_LAYOUT stays pristine
    demoLayoutRef.current = DEFAULT_LAYOUT.map((b) => ({ ...b }));
    demoSentenceRef.current = DEFAULT_SENTENCE;
    demoPlanRef.current = null;
    lastCycleRef.current = -1;
    rolloutLayoutRef.current = DEFAULT_LAYOUT.map((b) => ({ ...b }));
    activeTokensRef.current = DEFAULT_SENTENCE.tokens;
    userSentenceRef.current = null;
    pausedAccumRef.current = 0;
    pauseStartRef.current = null;
    trainStartRef.current = 0;
    setDemoSentence(DEFAULT_SENTENCE);
    setUserSentence(null);
    setTryText("");
    setTryNote(null);
    setTokenBars([]);
    setDecoded(null);
    setRolloutSamples(0);
    setHud({ lossText: "—", samples: 0, batches: 0 });
    // guidance layer: narration + tips restart with the run (the one-time
    // localStorage nudges deliberately do not)
    captionStageRef.current = 0;
    setCaption(null);
    setTip(null);
    setOpenInfo(null);
    setPhIdx(0);
    setFlashTry(false);
    triedRef.current = false;
    episodeLiveRef.current = false;
    episodeHeldRef.current = false;
    statusRef.current = "idle";
    setStatus("idle");
  }, [syncRunCfg]);

  // Terminate the worker (releasing its WebGL context back to the browser's
  // process-wide pool) and return to idle — as opposed to reset(), which keeps
  // the worker warm. Every context-pressure teardown goes through here.
  // Restart-to-idle: an interrupted run is NOT resumed; the viewer presses
  // Start again, which builds a genuinely fresh worker.
  const releaseWorkerToIdle = useCallback(() => {
    trainerRef.current?.destroy();
    trainerRef.current = null;
    resetToIdle();
  }, [resetToIdle]);

  // Fires when "training" outlives TRAIN_STALL_MS with no new batch — the worker
  // went silent mid-run (the mid-training twin of the loading wedge). Release
  // the worker (freeing the lost context) and surface a recoverable failure.
  const onTrainStalled = useCallback(() => {
    trainWatchdogRef.current = null;
    if (statusRef.current !== "training") return; // paused/converged; stale fire
    console.warn(
      `VLA trainer: no batch progress within ${TRAIN_STALL_MS}ms — releasing the worker`
    );
    releaseWorkerToIdle();
    setHostFailure("train-stalled");
  }, [releaseWorkerToIdle]);

  // The run's loss went non-physical (zeroed GPU readback of a dead context) —
  // either a run of dead batches (DEAD_LOSS_LIMIT) or a "converged" that landed
  // below CONVERGED_LOSS_FLOOR. Either way the policy is worthless: tear the
  // worker down and surface it instead of handing off a garbage "trained" model.
  const onTrainCollapsed = useCallback(() => {
    console.warn(
      "VLA trainer: action loss collapsed to a non-physical value (dead WebGL context?) — releasing the worker"
    );
    releaseWorkerToIdle();
    setHostFailure("train-collapsed");
  }, [releaseWorkerToIdle]);

  const clearTrainWatchdog = () => {
    if (trainWatchdogRef.current !== null) {
      window.clearTimeout(trainWatchdogRef.current);
      trainWatchdogRef.current = null;
    }
  };
  const armTrainWatchdog = () => {
    clearTrainWatchdog();
    trainWatchdogRef.current = window.setTimeout(onTrainStalled, TRAIN_STALL_MS);
  };

  // ---- controls ----
  // Second-tier load net, armed by onLoadStuck once the package has swapped to
  // the replay (usingReplay === true) yet is STILL in "loading" past the first
  // watchdog. The replay loads off timeout-free fetches (mini-vla
  // trainer.replay.ts), so a body that stalls without rejecting — the
  // flaky-mobile case this whole fallback targets — never surfaces as an error;
  // this is the only thing that catches it. Generous by design (see
  // REPLAY_LOAD_WATCHDOG_MS): the healthy replay load finishes well under this
  // budget, so a trip means a real stall, not a slow link. Same recovery as a
  // wedged real load — destroy() also disposes the replay's tf models.
  const onReplayLoadStuck = useCallback(() => {
    loadWatchdogRef.current = null;
    if (statusRef.current !== "loading") return; // replay reached training; stale
    console.warn(
      `VLA replay: still loading ${REPLAY_LOAD_WATCHDOG_MS}ms past the first watchdog — a stalled asset fetch; tearing down`
    );
    releaseWorkerToIdle();
    setHostFailure("load-stuck");
  }, [releaseWorkerToIdle]);

  // Fires when "loading" outlasts LOADING_WATCHDOG_MS with no word from the
  // worker — see HostFailure's "load-stuck" for the failure this catches. The
  // worker itself never posts an error here (it isn't dead, just wedged
  // against a WebGL context that died on arrival), so nothing else would ever
  // move status off "loading". Tearing the worker down on detection matters as
  // much as reporting it: destroy() releases the lost context instead of
  // leaving it pinned for the rest of the tab's life.
  const onLoadStuck = useCallback(() => {
    loadWatchdogRef.current = null;
    if (statusRef.current !== "loading") return; // already moved on; stale fire
    // With replayFallback on the package owns first-line load-stuck recovery: it
    // swaps to the replay at ~7.5s and the replay re-enters loading → training
    // on its own. If that swap is underway (usingReplay === true), a still-
    // "loading" status is the REPLAY loading, not the wedged real run — so don't
    // kill it here. But don't stand down for good either: the replay's own load
    // is a chain of timeout-free fetches, so a stalled-but-not-rejected asset
    // fetch on a flaky connection would leave it pinned in "loading" with no way
    // out — the very hang this path exists to kill, one layer down. Hand off to
    // a second, longer ceiling that bounds the replay load itself. (A replay
    // whose assets cleanly 404 needs no net — the package lands on status
    // "error"/errorReason "assets", which the bar already surfaces.)
    if (trainerRef.current?.usingReplay) {
      loadWatchdogRef.current = window.setTimeout(
        onReplayLoadStuck,
        REPLAY_LOAD_WATCHDOG_MS
      );
      return;
    }
    console.warn(
      `VLA trainer: no progress out of Language Warmup within ${LOADING_WATCHDOG_MS}ms — releasing the worker`
    );
    releaseWorkerToIdle(); // destroy the worker + release its lost context
    setHostFailure("load-stuck");
  }, [releaseWorkerToIdle, onReplayLoadStuck]);

  const onUpdate = () => {
    const trainer = trainerRef.current;
    if (!trainer) return; // torn down mid-flight (a watchdog just released it)
    // Mirror the package's transparent swap to the replay so the "replay" chip
    // tracks it. Read every tick — the swap can flip mid-run; React bails on an
    // unchanged value, so this is free.
    setUsingReplay(trainer.usingReplay);
    if (trainer.status !== statusRef.current) {
      // the language warm-up has just handed off to the coupled loop: this is
      // when the demonstration cycle actually begins, so stamp its clock here
      // rather than at the click (the warm-up's duration varies with the
      // machine, and the arm must not arrive mid-cycle). Resume-from-pause is
      // NOT a handoff — only loading → training re-stamps.
      if (statusRef.current === "loading" && trainer.status === "training") {
        trainStartRef.current = performance.now() + 500; // half-second ease-in
        advanceCaption(1, CAPTION_DEMO); // the narration opens on the expert
      }
      // Arm the stuck-loading watchdog for exactly the "loading" span: entering
      // it starts the clock, leaving it (to training OR to a real error the
      // worker DID manage to report) means the run is progressing on its own.
      if (trainer.status === "loading" && statusRef.current !== "loading") {
        if (loadWatchdogRef.current !== null)
          window.clearTimeout(loadWatchdogRef.current);
        loadWatchdogRef.current = window.setTimeout(onLoadStuck, LOADING_WATCHDOG_MS);
      } else if (statusRef.current === "loading" && trainer.status !== "loading") {
        if (loadWatchdogRef.current !== null) {
          window.clearTimeout(loadWatchdogRef.current);
          loadWatchdogRef.current = null;
        }
      }
      // Training-progress watchdog (B3): run it for exactly the "training" span.
      // Entering training arms the stall clock and resets the dead-loss
      // counters; leaving it (pause / converge / error / idle) disarms it — a
      // pause legitimately stops batches and must not be read as a stall.
      if (trainer.status === "training") {
        lastWatchedBatchRef.current = trainer.batches;
        deadLossRunRef.current = 0;
        armTrainWatchdog();
      } else {
        clearTrainWatchdog();
      }
      // Failures arrive as status "error" (mini-vla >= 0.4.0) with a reason
      // that decides the way out — see the error branch of the bar below.
      if (trainer.status === "error") setErrorReason(trainer.errorReason);
      setStatusBoth(trainer.status);
      if (trainer.status === "converged") {
        // False-convergence guard (B3): a genuinely converged Huber action loss
        // sits ~0.012–0.015; at/below CONVERGED_LOSS_FLOOR (or non-finite) it is
        // the zeroed readback of a dead context, not a trained policy. Reject it
        // as a collapse instead of unlocking "try it" on a worthless model.
        if (!Number.isFinite(trainer.loss) || trainer.loss < CONVERGED_LOSS_FLOOR) {
          onTrainCollapsed();
          return;
        }
        // auto-episodes end; the rollout waits for the user's command
        engineRef.current!.reset();
        visGazeRef.current = null;
        armState.current = { a1: REST[0], a2: REST[1] };
        // the interactive policy is the final model — show the full training
        // budget it saw, not the last per-cycle snapshot count
        setRolloutSamples(trainer.samples);
        // the narration's job is done — the try-row takes over from here,
        // and the ⓘ layer retires with the rest of the pipeline chrome
        captionStageRef.current = 5;
        setCaption(null);
        setOpenInfo(null);
      }
    } else if (trainer.status === "training") {
      // Resume re-arm: resumeTraining() pre-sets status to "training", so the
      // worker's resume-ack "state" is NOT a transition and misses the entry-arm
      // above. pauseTraining() always clears the ref on pause (see its own
      // comment), so every resume needs this re-arm, not just a long one — a
      // resume onto a context that died while paused produces no batches and
      // would hang with no reload prompt. Arm here whenever training is
      // running unwatched.
      if (trainWatchdogRef.current === null) armTrainWatchdog();
      // Steady-state training (no status change): a new batch landed. Reset the
      // stall clock and watch for a run of non-physical losses — DEAD_LOSS_LIMIT
      // zeroed/NaN readbacks in a row means the WebGL context died mid-run and
      // every subsequent gradient is worthless, so bail before the garbage
      // policy silently reaches the batch-107 false-convergence.
      if (trainer.batches !== lastWatchedBatchRef.current) {
        lastWatchedBatchRef.current = trainer.batches;
        armTrainWatchdog();
        const l = trainer.loss;
        if (!Number.isFinite(l) || l === 0) {
          if (++deadLossRunRef.current >= DEAD_LOSS_LIMIT) {
            onTrainCollapsed();
            return;
          }
        } else {
          deadLossRunRef.current = 0;
          // within 2x of the convergence threshold: tell the reader the wait
          // is almost over (a lucky single-batch dip firing early is harmless)
          if (l < CONFIG.trainer.converge.loss * 2)
            advanceCaption(4, CAPTION_ALMOST);
        }
      }
    }
    const now = performance.now();
    if (now - lastHudRef.current > 150 && !Number.isNaN(trainer.loss)) {
      lastHudRef.current = now;
      setHud({
        lossText: trainer.loss.toFixed(3),
        samples: trainer.samples,
        batches: trainer.batches,
      });
    }
  };

  const onPrimary = () => {
    const trainer = (trainerRef.current ??= new VLATrainer({
      assetBase: VLA_ASSET_BASE,
      // On a stall or error the package transparently swaps in the CPU-backend
      // replay (real rollouts off a captured policy ladder, scripted loss
      // curve) behind the same surface — the fix for the iOS/iPadOS WebGL
      // context cap that no backend switch can rescue. See §4/§6: the host
      // watchdogs become a thin outer net and a "replay" chip flags it.
      replayFallback: true,
      // Swap a stalled run to the replay at 6s instead of the package's 7.5s
      // default (see REPLAY_WATCHDOG_MS) — gets iOS/iPadOS to a visible run
      // sooner without risking a false swap on a healthy device.
      replayWatchdogMs: REPLAY_WATCHDOG_MS,
    }));
    // any press of the bar is a deliberate choice: it ends any auto-pause, so
    // becoming visible again never overrides what the viewer just asked for
    autoPausedRef.current = false;
    if (trainer.status === "training") {
      pauseTraining();
    } else if (trainer.status === "paused") {
      resumeTraining();
    } else if (trainer.status === "idle" || trainer.status === "error") {
      // "error" restarts in place: start() clears errorReason and refetches
      // (loadEmbeddings un-caches a rejected promise). Only "worker" can't be
      // retried this way, and that arm offers Reload instead of this button.
      engineRef.current!.reset();
      rolloutCycleRef.current = -1;
      visGazeRef.current = null;
      armState.current = { a1: REST[0], a2: REST[1] };
      userSentenceRef.current = null;
      pausedAccumRef.current = 0;
      pauseStartRef.current = null;
      setUserSentence(null);
      setDecoded(null);
      setTokenBars([]);
      setRolloutSamples(0);
      // install the viewport's task profile on this thread's samplers before
      // anything draws a command (trainer.start also ships it to the worker),
      // and freeze it: a resize across the breakpoint must not change the task
      // out from under a run in progress
      const cfg = runCfgRef.current;
      cfgLockedRef.current = true;
      setRunConfig(cfg);
      // per-block clone of the default scene — the rollout's blocks are
      // draggable once converged, so the module-level DEFAULT_LAYOUT must
      // stay pristine
      demoLayoutRef.current = DEFAULT_LAYOUT.map((b) => ({ ...b }));
      rolloutLayoutRef.current = DEFAULT_LAYOUT.map((b) => ({ ...b }));
      setErrorReason(null); // mirrors start()'s own clear; a retry shows no stale reason
      setHostFailure(null);
      // guidance layer: a fresh run narrates from the top (Start was pressed,
      // so the idle flash's class condition drops with the status change)
      setFlashCta(false);
      captionStageRef.current = 0;
      setCaption(null);
      setTip(null);
      episodeLiveRef.current = false;
      episodeHeldRef.current = false;
      void trainer.start(onUpdate, cfg);
    }
  };

  const onReset = () => {
    // reset() keeps the worker warm (tfjs + embeddings stay cached for a fast
    // restart); resetToIdle() returns all the local view state to idle.
    trainerRef.current?.reset();
    resetToIdle();
  };

  // ---- mobile: open / close the stacked demo ----
  // "Closed" is not a trainer state — closing only hides the UI and, if a run
  // was in progress, pauses it through the normal path so status/loss/bar stay
  // consistent when it is reopened. Reopening deliberately does NOT resume:
  // the viewer restarts it from the bar, exactly as after any manual pause.
  const openDemo = () => {
    showDemoRef.current = true;
    setShowDemo(true);
    setFlashCta(false); // the nudge did its job (or the viewer beat it to it)
  };
  const closeDemo = () => {
    showDemoRef.current = false;
    setShowDemo(false);
    autoPausedRef.current = false;
    pauseTraining();
  };

  // Battery + context guards. A training run behind a hidden tab or scrolled
  // far off screen is invisible work: pause it immediately (battery). If it
  // STAYS backgrounded past the grace period, go further and release the whole
  // worker (A1/A2) — the pause alone leaves the WebGL context pinned, and a
  // context pinned behind a suspended tab is what starves the next visit
  // against the browser's process-wide cap. Only a pause WE initiated is
  // auto-resumed (a manual pause never sets the flag, so it is never undone).
  useEffect(() => {
    const stage = stageRef.current;
    // "Backgrounded" = the hero can't be seen: tab hidden OR scrolled off
    // screen. Foreground requires BOTH visible and on-screen.
    const backgrounded = () =>
      document.visibilityState === "hidden" || !heroOnScreenRef.current;

    // A2: after IDLE_TEARDOWN_GRACE_MS backgrounded, release the worker so its
    // context returns to the pool. Armed on entry into the backgrounded state;
    // a return to the foreground cancels it before it fires.
    const armIdleTeardown = () => {
      if (graceTimerRef.current !== null) return; // already ticking
      graceTimerRef.current = window.setTimeout(() => {
        graceTimerRef.current = null;
        // re-check: only tear down if still backgrounded and a worker exists.
        // A CONVERGED run is spared — it's a finished, interactive "try it"
        // session, and nuking its trained policy on a casual tab-away is worse
        // than holding its one context. Training/loading/idle workers are the
        // frequent, expensive contributors to the cap and still get released.
        if (!backgrounded() || !trainerRef.current) return;
        if (statusRef.current === "converged") return;
        releaseWorkerToIdle();
      }, IDLE_TEARDOWN_GRACE_MS);
    };
    const cancelIdleTeardown = () => {
      if (graceTimerRef.current !== null) {
        window.clearTimeout(graceTimerRef.current);
        graceTimerRef.current = null;
      }
    };

    // The one reconciler for both signals (visibility + intersection). Pausing
    // gradient steps is immediate; releasing the worker waits out the grace.
    const sync = () => {
      if (backgrounded()) {
        if (statusRef.current === "training") {
          pauseTraining();
          autoPausedRef.current = true;
        }
        armIdleTeardown();
      } else {
        cancelIdleTeardown();
        if (autoPausedRef.current) {
          autoPausedRef.current = false;
          resumeTraining();
        }
      }
    };

    document.addEventListener("visibilitychange", sync);
    // A1: pagehide means the page is about to be frozen into bfcache (or
    // unloaded). A frozen page keeps its worker + WebGL context ALIVE but its
    // timers stop, so the grace teardown would never fire — release the worker
    // NOW so its context isn't pinned against the cap for the whole suspension.
    // A bfcache restore lands on idle; the viewer presses Start again.
    const onPageHide = () => {
      cancelIdleTeardown();
      // spare a converged run (see the grace-teardown note) — a bfcache restore
      // then brings the viewer back to their trained "try it" instead of idle.
      if (trainerRef.current && statusRef.current !== "converged")
        releaseWorkerToIdle();
    };
    window.addEventListener("pagehide", onPageHide);
    // A2: pageshow with persisted===true means a bfcache restore — the browser
    // thawed a frozen snapshot. pagehide spares converged runs so the viewer
    // returns to their trained policy, but a frozen WebGL context is usually dead
    // on thaw: the GPU process recycled it during suspension. Destroy everything
    // and land on idle so Start builds a genuinely fresh worker + context.
    const onPageShow = (e: PageTransitionEvent) => {
      if (!e.persisted) return;
      cancelIdleTeardown();
      if (trainerRef.current) releaseWorkerToIdle();
    };
    window.addEventListener("pageshow", onPageShow);
    // fires once on observe, which is how heroOnScreenRef gets its real value
    const io = new IntersectionObserver(([entry]) => {
      heroOnScreenRef.current = entry.isIntersecting;
      sync();
    });
    if (stage) io.observe(stage);
    return () => {
      document.removeEventListener("visibilitychange", sync);
      window.removeEventListener("pagehide", onPageHide);
      window.removeEventListener("pageshow", onPageShow);
      cancelIdleTeardown();
      io.disconnect();
    };
  }, [pauseTraining, resumeTraining, releaseWorkerToIdle]);

  // Stale-tab guard: a tab left open across a deploy runs old JS against the
  // current deploy's assets. On tab-return, fetch the deploy's build id and
  // reload if it no longer matches the one baked into this bundle.
  useEffect(() => {
    const buildId = process.env.NEXT_PUBLIC_BUILD_ID;
    if (!buildId) return;
    const check = async () => {
      if (document.hidden) return;
      try {
        const r = await fetch(BUILD_ID_PATH, { cache: "no-store" });
        if (!r.ok) return;
        const { id }: BuildIdPayload = await r.json();
        // A malformed-but-200 body (id missing/not a string) isn't evidence
        // of a stale deploy — treat it the same as the network/parse
        // failures below (skip), not as an automatic "reload".
        if (typeof id === "string" && id !== buildId) window.location.reload();
      } catch { /* offline / fetch failed — skip */ }
    };
    document.addEventListener("visibilitychange", check);
    return () => document.removeEventListener("visibilitychange", check);
  }, []);

  // ---- guidance layer: one-time nudges + converged affordances ----
  // Hydrate the seen-tips set. Client-only — localStorage does not exist on
  // the server render, so this cannot live in the useState initializer.
  useEffect(() => {
    tipsShownRef.current = new Set(loadHints().tips ?? []);
  }, []);

  // Idle-nudge flash: after ~3 CONSECUTIVE on-screen seconds while idle, flash
  // the NEXT step of the path in the language-warmup red. flashCta lands on both
  // the "Try mini-vla" CTA and the Start Training button; CSS only ever shows
  // one of them, so the same flag drives whichever is currently on screen.
  //
  // Re-armed on every showDemo change so the whole mobile path gets nudged in
  // turn: closed → flash the CTA; opening the demo clears it (openDemo) and this
  // re-runs, so after another ~3s it flashes the now-visible Start button. On
  // desktop showDemo is inert, so this stays a single flash of the Start button.
  // (The converged prompt box is the path's last step — see the flashTry twin.)
  //
  // Consecutive, so a drive-by scroll doesn't bank progress. Leaving idle (Start
  // pressed) retires it.
  //
  // Repeats: each beat is a 3s untouched-idle wait + the 6s flash (vla-*-flash =
  // 1.5s × 4); at the end of the cycle the flag is cleared and a fresh 3s wait
  // re-arms it, so an unattended hero keeps re-inviting until Start is pressed
  // (or the mobile demo opens). Clearing the flag between beats drops the
  // is-flash class, so re-adding it restarts the CSS animation cleanly.
  useEffect(() => {
    const FLASH_AT = 3; // seconds of untouched idle before a beat fires
    const CYCLE_END = 9; // FLASH_AT + 6s flash — clear here and re-arm
    let seen = 0; // consecutive on-screen idle seconds within this cycle
    const id = window.setInterval(() => {
      if (statusRef.current !== "idle") {
        window.clearInterval(id);
        return;
      }
      if (!heroOnScreenRef.current) {
        seen = 0;
        setFlashCta(false);
        return;
      }
      seen += 1;
      if (seen === FLASH_AT) {
        setFlashCta(true);
      } else if (seen >= CYCLE_END) {
        setFlashCta(false); // beat done; the next 3s wait starts a new cycle
        seen = 0;
      }
    }, 1000);
    return () => window.clearInterval(id);
  }, [showDemo]);

  // The converged twin of the idle nudge: the try-row is the payoff, but a
  // viewer who just watched a minute of training may not realize the input is
  // now theirs. After ~3 CONSECUTIVE on-screen seconds at converged with no
  // interaction (typing, running, dragging, shuffling all count), flash the
  // prompt box on the same beat as the other nudges.
  //
  // Repeats on the same 3s-wait + 6s-flash cycle as the idle nudge, re-arming
  // until the viewer engages: any try-row interaction sets triedRef and retires
  // it for good. So it keeps inviting a converged-but-idle viewer, but the first
  // keystroke (or run/drag/shuffle) stops it permanently.
  useEffect(() => {
    if (status !== "converged") return;
    // no setFlashTry(false) here: every exit from converged runs resetToIdle,
    // which already clears it — a synchronous setState in an effect body is
    // both redundant and a lint error (react-hooks/set-state-in-effect)
    triedRef.current = false;
    const FLASH_AT = 3; // seconds of untouched converged idle before a beat fires
    const CYCLE_END = 9; // FLASH_AT + 6s flash — clear here and re-arm
    let seen = 0; // consecutive on-screen idle seconds within this cycle
    const id = window.setInterval(() => {
      if (triedRef.current) {
        window.clearInterval(id); // engaged — retire the nudge for good
        return;
      }
      if (!heroOnScreenRef.current) {
        seen = 0;
        setFlashTry(false);
        return;
      }
      seen += 1;
      if (seen === FLASH_AT) {
        setFlashTry(true);
      } else if (seen >= CYCLE_END) {
        setFlashTry(false); // beat done; the next 3s wait starts a new cycle
        seen = 0;
      }
    }, 1000);
    return () => window.clearInterval(id);
  }, [status]);

  // Converged unlock: glow the try-input once and (desktop only — focusing on
  // a phone yanks the keyboard up) hand it the caret. Keyed to the genuine
  // training → converged transition, not to converged renders in general.
  useEffect(() => {
    const prev = prevStatusForUnlockRef.current;
    prevStatusForUnlockRef.current = status;
    if (status !== "converged" || prev !== "training") return;
    setJustConverged(true);
    if (!window.matchMedia(STACKED_MQ).matches && heroOnScreenRef.current)
      tryInputRef.current?.focus();
    const t = window.setTimeout(() => setJustConverged(false), 2600);
    return () => window.clearTimeout(t);
  }, [status]);

  // Rotating placeholder: cycles example commands while the box sits empty —
  // documentation of the grammar (and its synonyms) disguised as a hint.
  useEffect(() => {
    if (status !== "converged") return;
    const id = window.setInterval(
      () => setPhIdx((i) => (i + 1) % TRY_PLACEHOLDERS.length),
      4000
    );
    return () => window.clearInterval(id);
  }, [status]);

  /** "Try it" mode: run a sentence — the viewer's own, or a preset chip's —
      through the trained policy. The color head decodes which block to pick up;
      the motion itself is entirely the policy's. */
  const runCommand = async (command?: string) => {
    const trainer = trainerRef.current;
    if (!trainer?.ready || statusRef.current !== "converged") return;
    retireTryNudge(); // the viewer found the try-row
    const text = (command ?? tryText).trim();
    if (!text) return;
    const words = text
      .toLowerCase()
      .replace(/[^a-z ]/g, "")
      .split(" ")
      .filter(Boolean)
      .slice(0, MAX_SEQ_LEN);

    // The color head is 8-wide whatever the run config, but only the active
    // palette ever appears in a label — so a command naming one of the colors
    // this run never saw doesn't fail loudly, it quietly answers with the
    // nearest color it DOES know and picks up the wrong block. Say so instead.
    const cfg = runCfgRef.current;
    const untrained = COLORS.slice(cfg.numColors).find((c) =>
      c.synonyms.some((s) => words.includes(s))
    );
    if (untrained) {
      const known = activePalette(cfg)
        .map((c) => c.name)
        .join(", ");
      setTryNote(`This run never learned ${untrained.name} — only ${known}`);
      setDecoded(null); // nothing ran: don't leave the previous answer standing
      return;
    }

    const tokens = tokenize(text);
    // worker round-trip (~ms); swallow a rejected decode so a transient worker
    // failure just no-ops the command instead of surfacing an unhandled rejection
    let d: Awaited<ReturnType<typeof trainer.decodeCommand>>;
    try {
      d = await trainer.decodeCommand(tokens);
    } catch {
      return;
    }
    if (!d || statusRef.current !== "converged") return;
    // a color the policy knows, but that this scene doesn't contain: the reach
    // would silently fall back to some other block
    if (!rolloutLayoutRef.current.some((b) => b.color === d.color)) {
      setTryNote(`No ${COLORS[d.color].name} block in this scene`);
      setDecoded(null);
      return;
    }
    const sentence: Sentence = {
      color: d.color,
      text,
      words,
      tokens,
    };
    userSentenceRef.current = sentence;
    setUserSentence(sentence);
    activeTokensRef.current = tokens;
    setDecoded({
      name: COLORS[d.color].name,
      hex: COLORS[d.color].hex,
      prob: d.colorProb,
    });
    setTryNote(null);
    setTip(null); // a fresh command retires the previous post-run tip
    armState.current = { a1: REST[0], a2: REST[1] };
    engineRef.current!.begin(d.color, tokens);
  };

  /** Preset chips (mobile): the free-text box is the demo's payoff, but on a
      phone it is also the highest-friction step AND the palette is half the
      size — so offer one tap per color the run actually trained on. Derived
      from the profile, never hand-written, so they cannot drift from it. */
  const runPreset = (color: string) => {
    const command = `pick up the ${color} block`;
    setTryText(command);
    void runCommand(command);
  };

  const randomizeBlocks = () => {
    rolloutLayoutRef.current = randomLayout();
    // abort any in-flight episode so the arm re-plans against the new scene
    engineRef.current!.reset();
    armState.current = { a1: REST[0], a2: REST[1] };
    setTryNote(null);
    userShuffledRef.current = true; // the shuffle tip is now moot
    retireTryNudge();
    setTip(null);
  };

  // ---- drag the Rollout blocks (converged / "try it" mode only) ----
  // Once the policy is trained the viewer can reposition either block by
  // dragging it, constrained to that block's cleanly-reachable side band
  // (CONFIG.task.placeLeft/Right — the centre is a near-singular dead zone).
  // The block object is mutated in place, so the rAF loop redraws it moving
  // live; any in-flight attempt is cancelled so the arm re-plans on the next
  // Run against wherever the blocks now sit.
  const dragRef = useRef<{ block: BlockPos; band: [number, number] } | null>(null);

  // Invert sceneMap's X: canvas CSS-pixel x → workspace x. (X = W/2 + (x-0.5)*S.)
  const canvasX = (c: HTMLCanvasElement, clientX: number) => {
    const rect = c.getBoundingClientRect();
    const S = CONFIG.render.sceneScale * rect.height;
    return 0.5 + (clientX - rect.left - rect.width * 0.5) / S;
  };

  // The block under a pointer, if any (a few px of touch padding around the box).
  const blockAt = (c: HTMLCanvasElement, clientX: number, clientY: number) => {
    const rect = c.getBoundingClientRect();
    const S = CONFIG.render.sceneScale * rect.height;
    const floorY = CONFIG.render.floorY * rect.height;
    const px = clientX - rect.left;
    const py = clientY - rect.top;
    const pad = 6;
    for (const b of rolloutLayoutRef.current) {
      const cx = rect.width * 0.5 + (b.x - 0.5) * S;
      const half = (b.size * S) / 2;
      const bottom = floorY - (b.y ?? 0) * S; // rest height (normally floor)
      if (
        px >= cx - half - pad &&
        px <= cx + half + pad &&
        py >= bottom - b.size * S - pad &&
        py <= bottom + pad
      )
        return b;
    }
    return null;
  };

  const onBlockPointerDown = (e: React.PointerEvent<HTMLCanvasElement>) => {
    if (statusRef.current !== "converged") return;
    const c = armRef.current;
    if (!c) return;
    const b = blockAt(c, e.clientX, e.clientY);
    if (!b) return;
    e.preventDefault();
    c.setPointerCapture(e.pointerId);
    // lock to the side band the block starts on (bands never cross centre)
    dragRef.current = {
      block: b,
      band: b.x < 0.5 ? CONFIG.task.placeLeft : CONFIG.task.placeRight,
    };
    // cancel the current attempt; the viewer re-runs once blocks are placed
    engineRef.current!.reset();
    armState.current = { a1: REST[0], a2: REST[1] };
    setTryNote(null);
    userDraggedRef.current = true; // the drag tip is now moot
    retireTryNudge();
    setTip(null);
    // a dragged block rests on the floor
    b.y = 0;
    c.style.cursor = "grabbing";
  };

  const onBlockPointerMove = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const c = armRef.current;
    if (!c) return;
    const drag = dragRef.current;
    if (!drag) {
      // hover affordance: show a grab cursor over a draggable block
      if (statusRef.current === "converged")
        c.style.cursor = blockAt(c, e.clientX, e.clientY) ? "grab" : "default";
      return;
    }
    drag.block.x = clamp(canvasX(c, e.clientX), drag.band[0], drag.band[1]);
  };

  const onBlockPointerUp = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const c = armRef.current;
    if (dragRef.current && c) {
      c.releasePointerCapture?.(e.pointerId);
      c.style.cursor = "grab";
    }
    dragRef.current = null;
  };

  const active = userSentence ?? demoSentence;

  // Keep the language-encoder chips on a single line for any sentence length:
  // measure the row's natural (unwrapped) width against its width budget and
  // scale the whole row down to fit. Short sentences render at 1:1.
  useLayoutEffect(() => {
    const row = chipRowRef.current;
    if (!row) return;
    const fit = () => {
      row.style.setProperty("--chip-fit", "1");
      const budget = row.clientWidth; // the max-width budget from CSS
      const natural = row.scrollWidth; // full single-line content width
      const scale = natural > budget ? budget / natural : 1;
      row.style.setProperty("--chip-fit", `${scale}`);
    };
    fit();
    window.addEventListener("resize", fit);
    return () => window.removeEventListener("resize", fit);
    // showDemo: opening/closing the stack changes the row's width budget
    // without ever firing a resize event, so re-measure on the toggle too
  }, [active.text, showDemo]);

  // "error" is a dead run, not a live one: it shows the idle chrome plus a way out.
  const live = status !== "idle" && status !== "error";
  // The curve is only meaningful while batches are actually flowing, so it is
  // absent before the run (idle / language warm-up) and again once the run has
  // converged and the bar's job is to get out of the way of the command box.
  const showLoss = status === "training" || status === "paused";
  // Each failure path reads distinctly so a live run is self-reporting: the
  // exact phrase on screen tells us the mechanism without any telemetry.
  //   "Graphics context lost" — mini-vla (v0.4.1) itself caught a lost WebGL
  //     context, mid-run or via its zero-loss guard: direct evidence of GL loss.
  //   "Stuck — reload to retry" — never left warmup: either the real run's
  //     context was dead on arrival with no swap (tf.ready hung), or the replay
  //     it swapped to stalled on a timeout-free asset fetch (a flaky link).
  //   "Training stalled" — host watchdog; batches stopped with no package error
  //     (thermal/throttle/memory, or a context death that fired no event).
  //   "Training collapsed" — host watchdog; non-physical losses the package did
  //     not flag as context (fp16 NaN, or undetected silent-zero readbacks).
  //   "Load failed" — assets/worker/train: NOT a GPU-context failure.
  //   (Page silently self-reloads with no label — iOS evicted the whole tab.)
  const statusText = hostFailure
    ? hostFailure === "load-stuck"
      ? "Stuck — reload to retry"
      : hostFailure === "train-stalled"
        ? "Training stalled — reload"
        : "Training collapsed — reload"
    : status === "error"
      ? errorReason === "context"
        ? "Graphics context lost — reload"
        : "Load failed"
      : status === "idle"
        ? "Idle"
        : status === "loading"
          ? "Language Warmup"
          : status === "paused"
            ? "Paused"
            : status === "converged"
              ? "Ready"
              : "Training";
  const stateClass =
    status === "idle" || status === "error"
      ? "is-idle"
      : status === "loading"
        ? "is-live is-loading"
        : status === "paused"
          ? "is-live is-paused"
          : status === "converged"
            ? "is-live is-converged"
            : "is-live";

  return (
    <header
      className={`hero ${stateClass}${showDemo ? " demo-open" : ""}`}
      ref={stageRef}
    >
      {/* ✕ + training bar travel together: stacked, they pin to the top of the
          viewport as one sticky unit, so the controls AND the way out stay
          reachable however far down the pipeline the reader has scrolled. On
          desktop the wrapper is `display: contents` — it leaves no box behind,
          and the bar keeps floating at the bottom of the ring exactly as before. */}
      <div className="vla-topbar">
        {/* mobile only: leaves the stacked demo, pausing any run behind it */}
        <button
          className="vla-close"
          onClick={closeDemo}
          type="button"
          aria-label="Close the VLA demo"
        >
          ✕
        </button>

        {/* training control bar */}
        <div className={`vla-bar${showLoss ? "" : " is-compact"}`}>
          <div className="vla-status">
            {/* the pulse means "work is happening": the language warm-up is
                already loading embeddings, so it beats there too */}
            <span
              className={`vla-dot${
                status === "training" || status === "loading" ? " is-on" : ""
              }${status === "converged" ? " is-done" : ""}`}
            />
            <div className="vla-status-col">
              <span className="vla-status-text">{statusText}</span>
              {/* Honesty marker: the package swapped live training for the
                  CPU-backend replay (iOS/iPadOS, where the live WebGL run can't
                  get going). Understated by intent — the point is disclosure,
                  not a banner. A title tooltip never reaches these devices
                  (touch, no hover), so the "?" is a tap target instead. */}
              {usingReplay && (
                <div className="vla-replay-chip-wrap">
                  <button
                    type="button"
                    className="vla-replay-chip"
                    onClick={() => setShowReplayInfo((v) => !v)}
                    aria-expanded={showReplayInfo}
                  >
                    replay
                    <span className="vla-replay-info-icon" aria-hidden="true">
                      i
                    </span>
                  </button>
                  {showReplayInfo && (
                    <span className="vla-replay-info" role="note">
                      Your browser can&apos;t provide the compute needed to
                      show the live training process, so a replay of an
                      actual training run is shown instead.
                    </span>
                  )}
                </div>
              )}
              {/* a bare status word + disabled button reads as broken; say
                  what the wait is and that it is short */}
              {status === "loading" && (
                <span className="vla-status-sub vla-warm-note">
                  Loading word embeddings (takes a few seconds)
                </span>
              )}
              {live && hud.samples > 0 && (
                <>
                  <span className="vla-status-sub">
                    {hud.samples.toLocaleString()} examples
                  </span>
                  <span className="vla-status-sub">
                    {hud.batches.toLocaleString()} batches
                  </span>
                </>
              )}
            </div>
          </div>
          {/* straight into the write-up: what this pipeline is and how it was
              built. It rides the middle slot of the bar whether or not the loss
              curve is there to host it — centred in the gap while the bar is
              waiting, tucked into the curve's caption row while it plots. */}
          {showLoss ? (
            <>
              <div className="vla-loss">
                <div className="vla-loss-head">
                  <div className="vla-loss-label">Huber Loss</div>
                  <Link className="vla-project-link" href="/projects/mini-vla">
                    mini-vla ↗︎
                  </Link>
                </div>
                <canvas className="vla-loss-canvas" ref={lossRef} />
              </div>
              <div className="vla-loss-val">{hud.lossText}</div>
            </>
          ) : (
            <div className="vla-link-slot">
              {/* the teaser sentence: what Start runs, that it is genuinely
                  live, how long it takes — with the write-up link folded into
                  the words (the bare standalone link crowded the compact bar).
                  Embedded in prose the link drops the ↗︎ and marks itself with
                  a red underline instead (.vla-teaser .vla-project-link); the
                  arrow stays on the standalone loss-head link. */}
              <span className="vla-teaser">
                Trains the{" "}
                <Link className="vla-project-link" href="/projects/mini-vla">
                  mini-vla
                </Link>{" "}
                model in your browser (takes&nbsp;~60s)
              </span>
            </div>
          )}
          {hostFailure ? (
            // Same reasoning as errorReason "worker" below: every host-detected
            // failure is process-wide GPU-context exhaustion (other tabs'
            // suspended pages can be holding contexts), so a fresh
            // `new Worker(...)` in THIS tab is not reliably safe from hitting it
            // again — only a reload (or closing other tabs first) clears it.
            <button
              className="vla-btn"
              onClick={() => window.location.reload()}
              type="button"
            >
              Reload
            </button>
          ) : status === "error" ? (
            // "worker": a content-hashed chunk a redeploy deleted under this
            // open tab. A fresh `new Worker(...)` resolves the same dead URL,
            // so only a page load can help. "context": the worker's WebGL
            // context was lost (mini-vla v0.4.1 detects dead-on-arrival and
            // mid-run silent-zero readbacks) — a lost context never recovers
            // without a new backend, and a fresh one is liable to be evicted
            // again under the same memory pressure, so reload too. "assets"/
            // "train": start() refetches (the rejected promise is un-cached)
            // and can genuinely succeed.
            errorReason === "worker" || errorReason === "context" ? (
              <button
                className="vla-btn"
                onClick={() => window.location.reload()}
                type="button"
              >
                Reload
              </button>
            ) : (
              <button className="vla-btn" onClick={onPrimary} type="button">
                Try again
              </button>
            )
          ) : status === "idle" || status === "loading" ? (
            <button
              className={`vla-btn${
                flashCta && status === "idle" ? " is-flash" : ""
              }`}
              onClick={onPrimary}
              type="button"
              disabled={status === "loading"}
            >
              Start Training
            </button>
          ) : status === "converged" ? (
            <button className="vla-btn vla-btn-ghost" onClick={onReset} type="button">
              Reset
            </button>
          ) : (
            <>
              <button className="vla-btn" onClick={onPrimary} type="button">
                {status === "training" ? "Pause" : "Resume"}
              </button>
              <button
                className="vla-btn vla-btn-ghost"
                onClick={onReset}
                type="button"
              >
                Reset
              </button>
            </>
          )}
        </div>
      </div>

      {/* No drawn connectors — the pipeline is stitched together purely by
          motion: a payload token detaches from each source card and glides
          across the gap into the next (offset-path trajectories set in
          layoutWires). Staggered a→b→c so the flow reads source → encoder →
          action head → rollout. Hidden until training starts. */}
      <div className="vla-flow" ref={flowRef} aria-hidden="true">
        <span className="vla-payload vla-flow-a" data-flow="p1" />
        <span className="vla-payload vla-flow-a" data-flow="p2" />
        <span className="vla-payload vla-flow-b" data-flow="p3" />
        <span className="vla-payload vla-flow-b" data-flow="p4" />
        <span className="vla-payload vla-flow-c" data-flow="p5" />
      </div>

      {/* Demonstration — far-left anchor; prompt floats above it */}
      <div className="vla-node vla-input" ref={inputRef}>
        <div className="vla-prompt" ref={promptRef}>
          {demoSentence.text}
          <span className="vla-grip" aria-hidden="true" />
        </div>
        <div className="vla-label">
          Demonstration
          {infoVisible && (
            <InfoDot id="demo" open={openInfo === "demo"} onToggle={toggleInfo} />
          )}
        </div>
        <canvas className="vla-canvas" ref={demoRef} />
        {/* the progress-keyed narration; keyed so each new line re-runs the
            fade-in. Anchored to the Demonstration — the expert scene it
            narrates: floats just below the card on desktop, folds in as the
            card's last row (below the canvas + prompt) when stacked. */}
        {caption && (status === "training" || status === "paused") && (
          <div className="vla-caption" key={caption}>
            {caption}
          </div>
        )}
      </div>

      {/* Vision Encoder — 32x32 CNN input, blown up pixelated. Hover (or tap on
          touch) flips the panel to the exact tensor the CNN receives: the input
          is 1 - pixel (see visionTensor in trainer.core.ts), a per-channel invert, so
          a CSS invert(1) reproduces the model's-eye view precisely. */}
      <div
        className={`vla-node vla-vision${modelView ? " model-view" : ""}`}
        ref={visionCardRef}
        onClick={() => setModelView((v) => !v)}
      >
        <div className="vla-label">
          Vision Encoder
          {infoVisible && (
            <InfoDot
              id="vision"
              open={openInfo === "vision"}
              onToggle={toggleInfo}
            />
          )}
        </div>
        <canvas className="vla-vision-canvas" ref={visionRef} />
        <div className="vla-vision-hint" aria-hidden="true" />
      </div>

      {/* Language Encoder — frozen pretrained GloVe embeddings, attention-
          pooled (a learned scorer weights each token so filler + padding stop
          diluting the color/verb word); near-synonyms the grammar never
          trained on ("gold") resolve via the pretrained geometry */}
      <div className="vla-node vla-lang" ref={langCardRef}>
        <div className="vla-label">
          Language Encoder
          {infoVisible && (
            <InfoDot id="lang" open={openInfo === "lang"} onToggle={toggleInfo} />
          )}
        </div>
        <div className="vla-prompt-echo">&quot;{active.text}&quot;</div>
        <div className="vla-chip-row" ref={chipRowRef}>
          {active.words.map((w, i) => (
            <Fragment key={`${w}-${i}`}>
              {i > 0 && (
                <span className="vla-arrow" aria-hidden="true">
                  →
                </span>
              )}
              <div className="vla-tok">
                <span className="vla-chip">{w}</span>
                <div className="vla-attn">
                  <div
                    className="vla-attn-fill"
                    style={{
                      width: `${Math.round(
                        Math.max(0.06, tokenBars[i] ?? 0.06) * 100
                      )}%`,
                    }}
                  />
                </div>
              </div>
            </Fragment>
          ))}
        </div>
        <div className="vla-decoded">
          decoded target:{" "}
          {decoded ? (
            <>
              <span
                className="vla-swatch"
                style={{ background: decoded.hex }}
              />
              {decoded.name} {(decoded.prob * 100).toFixed(0)}%
            </>
          ) : (
            "—"
          )}
        </div>
      </div>

      {/* Action Head — where the vision + language wires merge; shows the
          policy's current output (the predicted target joint angles) */}
      <div className="vla-node vla-action" ref={actionCardRef}>
        <div className="vla-label">
          Action Head
          {infoVisible && (
            <InfoDot
              id="action"
              open={openInfo === "action"}
              onToggle={toggleInfo}
            />
          )}
        </div>
        <div className="vla-action-vals" ref={actionValsRef}>
          <span>
            <span className="vla-av-k">shoulder</span>
            <span className="vla-av-v">—</span>
          </span>
          <span>
            <span className="vla-av-k">elbow</span>
            <span className="vla-av-v">—</span>
          </span>
          <span>
            <span className="vla-av-k">gripper</span>
            <span className="vla-av-v">open</span>
          </span>
        </div>
      </div>

      {/* Rollout — policy episodes; becomes an interactive prompt when done */}
      <div className="vla-node vla-output" ref={outputCardRef}>
        {status === "converged" && (
          <div className="vla-try">
            <input
              className={`vla-try-input${justConverged ? " is-unlock" : ""}${
                flashTry ? " is-flash" : ""
              }`}
              ref={tryInputRef}
              placeholder={TRY_PLACEHOLDERS[phIdx]}
              value={tryText}
              onChange={(e) => {
                setTryText(e.target.value);
                setTryNote(null); // the note described the previous command
                retireTryNudge(); // typing counts
              }}
              onKeyDown={(e) => {
                if (e.key === "Enter") void runCommand();
              }}
            />
            <button
              className="vla-try-btn"
              onClick={() => void runCommand()}
              type="button"
            >
              Run
            </button>
            <button
              className="vla-try-btn vla-try-shuffle"
              onClick={randomizeBlocks}
              type="button"
              title="Randomize blocks"
            >
              Shuffle
            </button>
            {/* one chip per trained color — display:none above the breakpoint,
                where all eight colors are trained and typing is cheap */}
            <div className="vla-try-presets">
              {activePalette(runCfg).map((c) => (
                <button
                  key={c.name}
                  className="vla-try-chip"
                  onClick={() => runPreset(c.name)}
                  type="button"
                >
                  <span
                    className="vla-try-chip-dot"
                    style={{ background: c.hex }}
                  />
                  {c.name}
                </button>
              ))}
            </div>
            {/* error notes and post-run tips share the slot; a real note about
                the command that just ran always outranks a tip */}
            {(tryNote ?? tip) && (
              <div className="vla-try-note">{tryNote ?? tip}</div>
            )}
          </div>
        )}
        <div className="vla-out-head">
          <div className="vla-label">
            {status === "converged" ? "Your command" : "Rollout"}
            {infoVisible && (
              <InfoDot
                id="output"
                open={openInfo === "output"}
                onToggle={toggleInfo}
              />
            )}
          </div>
          {/* the payoff teaser: the single strongest reason to wait out the
              run, retired at converged where the real try-row replaces it */}
          {(status === "training" || status === "paused") && (
            <div className="vla-locked">Your command (unlocks when trained)</div>
          )}
          {rolloutSamples > 0 &&
            (status === "training" ||
              status === "paused" ||
              status === "converged") && (
              <div className="vla-seen">
                Trained on {rolloutSamples.toLocaleString()} demos
              </div>
            )}
        </div>
        <canvas
          className="vla-canvas"
          ref={armRef}
          style={status === "converged" ? { touchAction: "none" } : undefined}
          onPointerDown={onBlockPointerDown}
          onPointerMove={onBlockPointerMove}
          onPointerUp={onBlockPointerUp}
          onPointerCancel={onBlockPointerUp}
        />
      </div>

      <div className="hero-content">
        <h1>{profile.name}</h1>
        <div className="hero-tag">{profile.field}</div>
        <div className="hero-cta">
          <a className="btn-primary" href={profile.links.email}>
            Contact
          </a>
          <a className="btn-outline" href="/resume.pdf" download={resumeDownloadName()}>
            Download Resume
          </a>
          {/* the pipeline's way in on phones, where it cannot ring the name.
              Always rendered (SSR/hydration parity) — CSS hides it on desktop,
              where the ring is already on screen, and drops it onto its own row
              below the other two so neither of them has to wrap. */}
          <button
            className={`btn-outline hero-demo-btn${flashCta ? " is-flash" : ""}`}
            onClick={openDemo}
            type="button"
          >
            Try mini-vla
          </button>
        </div>
      </div>
    </header>
  );
}
