// ─────────────────────────────────────────────────────────────────────────
// VLA hero — the ONE place to tune the demo.
//
// Every knob the pipeline exposes (model architecture, optimizer, convergence,
// task difficulty, arm geometry, demo timing, rollout control) lives here.
// The lib/vla/*.ts modules and components/Hero.tsx read their constants from
// this object instead of hard-coding literals, so tuning is a single-file edit
// that hot-reloads in dev — no build/codegen step, type-checked, and shared by
// both the SSR pass and the browser bundle.
//
// It's a plain typed object (not YAML) so it stays synchronous at import time
// in every environment; the grouping + comments below give it the same
// "readable knobs sheet" feel. Deep rationale for each value lives next to it
// as a comment — read those before turning a knob, several are load-bearing.
// ─────────────────────────────────────────────────────────────────────────

/** One convolution stage of the vision encoder (relu activation). */
export interface ConvLayer {
  filters: number;
  kernel: number;
  /** conv stride (default 1). */
  stride?: number;
  /** "same" keeps the spatial size (default); "valid" shrinks it by the
      kernel. */
  padding?: "same" | "valid";
  /** apply 2x2 max-pool after this conv. */
  pool?: boolean;
}

export const CONFIG = {
  // ── Model architecture + optimizer (lib/vla/model.ts) ──────────────────
  model: {
    /** Square input resolution fed to the CNN. Raised 32→64 lets the policy
        resolve WHERE in its placement band a block sits (block ~8px, band
        ~12px) so it reaches precisely; at 32 the position blurred to ~3px and
        it could only learn the per-band mean target. The CNN and attention
        grid adapt symbolically, so this is the only downstream knob — but
        per-batch vision compute scales ~4x vs 32. Keep RENDER_SIZE ≈ 4x this. */
    imgSize: 64,
    /** Adam learning rate. 0.005 won the 2026-07 sweep at batchSize 32 /
        imgSize 64: 0.008 was collapse-prone (side-binding failure on bad
        seeds) and 0.003 measurably slower without being more reliable. */
    learningRate: 0.005,
    /** Weight of the auxiliary color-classification loss vs. the action loss.
        0.4 was the most collapse-resistant setting in the sweep (0 side-binding
        collapses across 7 seeds vs 1/7 at 0.2); 0.2 peaked slightly higher on
        lucky seeds but is riskier. */
    colorLossWeight: 0.6,
    /** Weight of the auxiliary attention-map loss (see model.ts): cross-
        entropy between the spatial attention map and the commanded block's
        grid cell. WHY IT EXISTS: the action loss alone cannot train the
        attention — with a near-uniform 16×16 map the softmax Jacobian dilutes
        its gradient by ~1/256, and the map never sharpens (measured: loss
        flat at the ~0.78 language-only plateau for 300+ batches). CE through
        the softmax has an undiluted (map − label) gradient, and the
        supervision is free — the expert already knows which block it
        labeled. */
    mapLossWeight: 1.5,
    /** Scale of the frozen soft-argmax coordinate kernel: the fusion sees the
        gaze as (imageCoord − 0.5) × this gain. WHY: in raw [0,1] units the
        within-band position signal spans only ~0.16 while the other ~74
        fusion inputs swing ~1.0, so the coordinate pathway's gradients are
        ~10x smaller and the action head parks on a per-side-mean policy
        (measured: gaze accurate to 0.003 while the reach still missed by
        0.10). This is plain feature standardization — the kernel is frozen,
        so the gain is exact, not a learned scale that early training could
        squash. */
    attnCoordGain: 32,
    /** Huber transition point for the action loss. The two IK target clusters
        (commanded block left vs. right) sit ~4.3 rad apart, so plain MSE lets
        the rare (~1%) wrong-side pick (cost ~9.3) dominate over regression
        precision — floors the loss near ~0.09 and thrashes the gradient. Huber
        is quadratic below DELTA (precise on correct-side ~0.1-rad jitter) and
        linear above (caps a wrong-side pick at ~2.5), dropping the floor to
        ~0.025 and smoothing the descent. 0.6 keeps correct-side samples in the
        quadratic zone while catching wrong-side picks early. */
    actionHuberDelta: 0.6,
    /** Vision CNN stack, in order. Add/remove entries to change depth; edit
        filters/kernel/stride/pool to retune a stage. The LAST stage's output
        map is what the language-conditioned spatial attention scores (see
        model.ts) — its spatial size sets the attention grid (64 → two pools →
        16×16 here), and its `filters` sets the attention query width. Keep the
        final map reasonably fine: the soft-argmax readout interpolates BELOW
        cell size, but the "does this cell match the command" scoring can only
        separate blocks that land in different cells. */
    conv: [
      { filters: 8, kernel: 3, pool: true },
      { filters: 16, kernel: 3, pool: true },
      { filters: 24, kernel: 3 },
    ] as ConvLayer[],
    /** Units in the fused hidden layer before the heads. The fusion input is
        now small and structured — soft-argmax (x̂,ŷ) + attended features +
        language vector, not a flattened feature map — so this mostly learns
        the coordinate→angles map. */
    fusionUnits: 64,
  },

  // ── Training loop + convergence (lib/vla/trainer.ts) ───────────────────
  trainer: {
    /** Samples synthesized + gradient-stepped per batch. 32 is load-bearing
        for RELIABILITY, not just speed: in the 2026-07 headless sweep (M4,
        ~100ms/batch at imgSize 64) batchSize 16 collapsed onto always-one-side
        policies on bad seeds (wrong-side rate 0.4-0.7, loss stuck ~0.78) where
        32 stayed healthy on the same seeds. Don't lower it. */
    batchSize: 32,
    /** Silhouettes are drawn at this px then averaged down to imgSize — drawn
        at target size directly the sub-pixel arm strokes alias away. Keep ≈4x
        imgSize to preserve the tuned antialiasing headroom. */
    renderSize: 256,
    /** Minimum ms yielded back to the rAF render loop between batches, so
        training never starves 60fps rendering. ~8ms leaves the loop its slice
        while fitting ~25% more gradient steps than the old 30ms. */
    batchGapMs: 8,
    /** Fraction of samples posed NEAR the commanded block's IK solution (rest
        uniform over the full pose range). The label is pose-independent now,
        but the rendered silhouette isn't — this keeps vision trained on what
        the scene looks like as the rollout closes in, not just far away. */
    nearTargetFrac: 0.5,
    /** Gaussian spread (rad) of that near-target pose jitter. */
    nearTargetStd: 0.5,
    /** Fraction of samples synthesized MID-CARRY: the commanded block is
        rendered at the effector of the sampled pose, the carry_flag input is
        1, and the label is the carry-phase target (REST — bring the grasped
        block home). This is what makes the carry phase policy-driven. The
        phase cue is the PROPRIOCEPTIVE FLAG plus the carried block's pixels —
        pre-flag, pixels were the ONLY cue, and when vision missed it the
        action head averaged the grasp/carry modes (see the carry-flag note
        in model.ts). UNTUNED. */
    carryFrac: 0.5,
    /** Chance a non-color token becomes <unk> in training, so the encoder
        learns to shrug off unknown words in free user text. */
    wordDropout: 0.1,
    /** LANGUAGE WARM-UP: text-only gradient steps run during the Loading phase,
        BEFORE the main vision→action loop starts (see languageWarmup in
        trainer.core). They train ONLY the pure-text color head plus its conv
        scorer, on synthesized sentences with the vision branch untouched, so
        the color decoding is already correct when the coupled policy starts
        and the attention query gets a clean language slot from batch 0
        instead of co-adapting against a still-moving language branch.

        This is the CAP: warm-up early-stops once the head's loss plateaus
        (the color head converges in tens of steps), so the typical Loading
        cost is well under this. Set 0 to disable. */
    warmupBatches: 200,
    /** Batch size for warm-up steps ONLY (the main loop keeps batchSize=32,
        which is load-bearing for its reliability). Bigger on purpose: warm-up
        is a tiny text-only graph with NO images, so it's bound by fixed
        per-step WebGL dispatch overhead, not compute — a larger batch does
        many more samples per dispatch at almost no extra cost, cutting the
        wall-clock for equal learning. Cheap in memory (int32 tokens + small
        one-hot labels, no pixel tensors). */
    warmupBatchSize: 256,
    // Convergence: mean action loss over the last `window` batches stays under
    // `loss` for `streak` consecutive batches (after `minBatches` warmup) →
    // training ends, "try it" mode unlocks. `maxBatches` is the fixed fallback.
    converge: {
      /** Handoff threshold on the trailing-window HUBER action loss.
          Calibration (2026-07 sweep, M4, ~100ms/batch → ~10 batches/s):
          healthy runs cross 0.02 at 150-280 batches ≈ 15-28s of training and
          score ~0.7-0.85 closed-loop reach success at handoff (0 wrong-side);
          the residual ~0.03 reach error is a vision-resolution floor, not
          undertraining. 2026-07 carry-flag re-gauge (headless SwiftShader ≈
          0.5x real GPU): pick-up 8c/4b converges ~0.012 at ~410 batches. */
      loss: 0.015,
      /** Trailing window (batches) the convergence mean is taken over. Small =
          low detection lag as old high losses roll off; the streak guards
          against a lucky dip. */
      window: 10,
      /** Consecutive in-threshold batches required before declaring converged. */
      streak: 8,
      /** Hard floor of batches before convergence can fire. Earliest genuine
          crossing observed in the sweep was ~155 batches, so 100 is pure
          lucky-dip insurance and never binds on healthy runs. */
      minBatches: 100,
      /** Fixed-budget fallback: converge regardless of loss at this batch
          (~45s at ~10 batches/s). Slow-but-healthy seeds (~1 in 3) land here
          or shortly before it with a usable policy. NOTE (sweep finding,
          pre-carry-flag): ~1 in 8 inits collapses to an always-one-side
          policy (loss flat ~0.78) and NEVER recovers — no swept parameter
          fixes it, so a longer budget only delays the fallback. Detectable
          early (smoothLoss > 0.4 at batch ~120); an auto-restart in
          trainer.core is the real fix if this rate bothers us. */
      maxBatches: 450,
    },
  },

  // ── Arm + workspace geometry (lib/vla/geometry.ts) ─────────────────────
  arm: {
    /** Upper-/fore-arm link lengths. Sized so the full reach circle
        (base ± l1+l2 = ±0.58) stays inside the rendered canvas — longer links
        let wild early-training poses swing the forearm out of the box. */
    l1: 0.32,
    l2: 0.26,
    /** Arm base anchor in the y-up unit workspace. */
    base: { x: 0.5, y: 0.2 },
    /** Upright rest pose [θ1, θ2] (straight up). */
    rest: [Math.PI / 2, 0] as [number, number],
    /** Pose-sampling ranges for synthesized training states. θ2 spans BOTH
        elbow configs: floor-block IK solutions sit near |θ2|≈2, so a narrower
        range would leave the expert's own targets unseen and the converged
        rollout out-of-distribution. */
    theta1Range: [-0.3, Math.PI + 0.3] as [number, number],
    theta2Range: [-2.4, 2.4] as [number, number],
  },
  block: {
    /** Reference side length — SSR-default + fallback when a block has no size. */
    ref: 0.12,
    /** Per-scene blocks randomize their side length in [min, max]. Bigger =
        grasped higher (grasp target is the block CENTRE, y=size/2) and shifts
        the near-singular dead zone, which is why the placement bands below are
        sized for the largest block. */
    min: 0.08,
    max: 0.16,
  },

  // ── Task / language space (lib/vla/examples.ts) ────────────────────────
  task: {
    /** Token slots per command (padded/truncated to this). 14 leaves ample
        headroom over the longest pick-up form (filler + verb + article +
        color + noun + please) for free user text. */
    maxSeqLen: 14,
    /** The two cleanly-reachable floor BANDS [lo, hi] blocks are placed in per
        side (the centre is a near-singular dead zone; see examples.ts). Inner
        edges (0.31/0.69) are set for the LARGEST block's elbow limit. */
    placeLeft: [0.11, 0.31] as [number, number],
    placeRight: [0.69, 0.89] as [number, number],
    // (Scene density — how many blocks, from how many colors, which tasks —
    // is the USER's ⚙ run config now: lib/vla/run-config.ts, not a knob here.)
    /** Extra clearance (workspace units) required between two same-side blocks'
        silhouettes, on top of their (boosted) half-widths — keeps them from
        occluding each other AND lands them in different attention cells (one
        16×16 cell ≈ 0.078 units) so the map can score them apart. */
    minBlockGap: 0.03,
    /** Grammar sampling probabilities: chance a sentence gets a leading filler
        word, and a trailing "please". */
    fillerProb: 0.25,
    pleaseProb: 0.2,
    /** FORM augmentation: chance each scaffolding element is DROPPED from a
        sampled sentence (verb per sentence; article/noun per occurrence),
        yielding compressed forms like "grab red" or bare "red" alongside the
        full grammar. WHY: the language scorer keys on LOCAL CONTEXT (conv
        window, model.ts), and these drops make training COVER the context
        variants free user text uses ("red" and "the red block" both common)
        instead of only the rigid full-grammar shape. Colors never drop
        (label token, same rule as word-dropout). */
    dropVerbProb: 0.15,
    dropArticleProb: 0.25,
    dropNounProb: 0.25,
  },

  // ── Demonstration trajectory (lib/vla/demo.ts) ─────────────────────────
  demo: {
    /** Synced cycle length. The scripted motion finishes at ~4.26s (phase sums
        below), so 5000 leaves a short REST beat; at ~5s/cycle the viewer sees
        ~5 policy generations before convergence. rollout.reachTimeout (frames)
        must stay ≥ this or a rollout gives up before the cycle resets. */
    periodMs: 5000,
    // Absolute-time trajectory phases (ms), independent of periodMs so the
    // scripted reach keeps its crisp speed regardless of the resting tail.
    // The pick-up trajectory: via → reach → settle → liftMs up → holdMs
    // aloft (ends ~4.26s).
    phases: {
      viaMs: 672, // rest → mid-trajectory waypoint
      reachMs: 672, // waypoint → block centre
      settleMs: 420, // settle on the block centre
      liftMs: 1092, // straight up back to rest
      graspAtMs: 1430, // block grasped mid-settle (carry begins)
      holdMs: 1400, // held aloft after the lift completes
    },
    /** Waypoint/reach noise amplitudes so no two demonstrations are identical:
        grasp x/y jitter, and the mid-trajectory via-point θ1/θ2 jitter. */
    jitter: { graspX: 0.012, graspY: 0.008, viaTheta1: 0.3, viaTheta2: 0.45 },
  },

  // ── Rollout control + episode timing (components/Hero.tsx) ─────────────
  rollout: {
    /** Proportional gain toward the predicted target each frame (0..1). */
    stepGain: 0.08,
    /** How often (ms) the policy re-predicts its target (closed loop). */
    predictMs: 80,
    /** Distance (workspace units) from the block centre that counts as reached. */
    graspEps: 0.07,
    /** Consecutive close frames required to register a grasp. */
    nearFrames: 4,
    /** Frames before a reach gives up as failed. Must exceed the synced demo
        cycle (demo.periodMs in frames) so a rollout isn't cut off early. */
    reachTimeout: 500,
    /** Joint-space closeness (rad, per joint) to the predicted carry-phase
        target that counts as "settled" — the release/lift-complete test now
        that the carry phase is policy-driven (there is no block to be near;
        the arm settles wherever the policy parked it). With stepGain 0.08
        the approach is asymptotic, so this can't be too tight. */
    settleEps: 0.05,
    /** Frames holding the block at the top (~0.5s), arm straight. */
    topHold: 30,
    /** Frames a failed attempt lerps back to rest over. */
    returnFrames: 40,
    /** End-effector trail length (points) drawn behind the rollout arm. */
    trailLen: 64,
    /** Throttle (ms) for refreshing the language-panel readout. */
    langMs: 300,
    /** Render size of the DISPLAYED model's-eye panel before its imgSize
        downsample. Display-only: the policy's actual input renders inside
        trainer.core at trainer.renderSize — this canvas just shows the
        visitor what that input looks like, so keeping it at the same ≈4x
        antialiasing (256, was 128) keeps the panel faithful to what the
        model really sees. */
    silRender: 256,
  },

  // ── Training-time estimate shown in the ⚙ run-config menu ──────────────
  // estimateTrainingSeconds (lib/vla/run-config.ts) is baseSeconds ×
  // colorFactor × blockFactor. GAUGED 2026-07 against the carry-flag pick-up
  // architecture (headless SwiftShader runs ≈ 0.5x real GPU, scaled to ~10
  // batches/s): pick-up 8c/4b ≈ 410 batches ≈ 41s. colors/blocks are small
  // modifiers around the base.
  eta: {
    baseSeconds: 42,
    colorFactor: { 2: 0.9, 4: 1.0, 8: 1.1 } as Record<number, number>,
    blockFactor: { 2: 0.9, 3: 1.0, 4: 1.1 } as Record<number, number>,
  },

  // ── Model's-eye rendering (lib/vla/scene.ts) ───────────────────────────
  render: {
    /** Isotropic workspace→canvas scale (× canvas height). */
    sceneScale: 0.8,
    /** Floor line position (× canvas height). */
    floorY: 0.86,
    /** Blocks render this much larger in the model's-eye silhouette than in the
        display scene — a display-size block is only a few px after the
        downsample; this keeps each color clearly present without touching the
        display. */
    silBlockScale: 1.3,
  },
};
