// ---- Hero guidance layer ----
// Info boxes, nudges and tips that walk a first-time viewer through the flow:
// idle (why press Start) → training (where to look) → converged (what the
// try-row can do). Everything follows the replay chip's precedent: tap-first
// (hover/title tooltips never reach the touch devices), quiet sentence-case
// mono copy, one disclosure open at a time.
//
// Extracted from components/Hero.tsx: these are the static copy tables, the
// persisted-hint helpers, and the InfoDot disclosure widget — all self-contained,
// so the host component keeps only its live pipeline state.

/** ⓘ popover copy, per card. Keyed by a closed union so a typo'd id on an
    InfoDot (or a renamed key here) is a compile error, not an `undefined`
    silently rendered into the popover. */
export type InfoId = "demo" | "vision" | "lang" | "action" | "output";
const INFO: Record<InfoId, string> = {
  demo: "An analytical expert performs each command yielding the policy's training data. Thousands more examples are generated invisibly between the ones you see.",
  vision: "The 32×32 silhouette the CNN actually sees; red = its spatial attention. Hover or tap to flip between this view and the model's-eye (inverted) one.",
  lang: "Frozen GloVe word embeddings, attention-pooled. Each bar is that word's learned weight. Synonyms it never trained on still resolve.",
  action: "The policy's live output: predicted target joint angles and the learned gripper action.",
  output: "The policy attempts the demonstration's scene and command with its currently learned policy.",
};

// Progress-keyed narration under the bar. Each line fires off a REAL signal
// (training handoff, first gaze reply, second synced attempt, loss nearing
// threshold) and holds until the next one — the captions are themselves the
// progress bar, so there is no separate percent readout.
export const CAPTION_DEMO =
  "The policy learns by trying to copy the expert's demonstrations";
export const CAPTION_GAZE =
  "Red = where the model looks; watch it sharpen onto the commanded block";
export const CAPTION_ROLLOUT =
  "The policy's attempt at the current training status";
export const CAPTION_ALMOST =
  "Almost converged. You can type in your command soon";

// Rotating try-box placeholders — documentation of the grammar (and its
// synonyms) disguised as a hint. Every color word maps into the FIRST
// numColors palette entries (crimson→red, golden→yellow), so the cycle is
// valid on both the desktop and the mobile task profile.
export const TRY_PLACEHOLDERS = [
  "e.g. grab the blue cube",
  "e.g. pick up the crimson one",
  "e.g. lift the golden block",
];

// Post-run tips: one per completed successful episode, shown in the try-note
// slot (real error notes there take priority). A tip whose action the viewer
// already found on their own is retired unshown. "gold" is genuinely absent
// from the grammar's synonym lists — it resolves through the pretrained
// embedding geometry, which is the point of the tip.
export const TIPS = [
  "Tip: drag a block somewhere else, then run again",
  "Tip: Shuffle rearranges the scene and the policy re-plans from vision",
  'Tip: "gold" was never in its training grammar, you can try it anyway',
];

// One-time nudges (the post-run tip chain) persist across visits. (`pulse`
// used to live here too; the idle nudge is per-page-load now, so the field
// is gone — a stale copy in an old visitor's storage is simply ignored.)
type HintsSeen = { tips?: number[] };
const HINTS_KEY = "vla-hints-seen";
export const loadHints = (): HintsSeen => {
  try {
    return JSON.parse(
      window.localStorage.getItem(HINTS_KEY) ?? "{}"
    ) as HintsSeen;
  } catch {
    return {};
  }
};
export const saveHints = (h: HintsSeen) => {
  try {
    window.localStorage.setItem(HINTS_KEY, JSON.stringify(h));
  } catch {
    /* storage denied (private mode) — the nudges just repeat next visit */
  }
};

/** Tap-first ⓘ disclosure beside a label: the replay chip's popover pattern,
    generalized. The host owns the single open id, so opening one closes the
    rest; stopPropagation keeps the Vision card's tap-to-flip out of it. */
export function InfoDot({
  id,
  open,
  onToggle,
}: {
  id: InfoId;
  open: boolean;
  onToggle: (id: InfoId) => void;
}) {
  return (
    <span className="vla-info-wrap">
      <button
        type="button"
        className="vla-info-btn"
        aria-label="What is this?"
        aria-expanded={open}
        onClick={(e) => {
          e.stopPropagation();
          onToggle(id);
        }}
      >
        i
      </button>
      {open && (
        <span
          className="vla-info-pop"
          role="note"
          onClick={(e) => e.stopPropagation()}
        >
          {INFO[id]}
        </span>
      )}
    </span>
  );
}
