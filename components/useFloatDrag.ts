import { useEffect, useRef, useState } from "react";

export interface FloatParams {
  towardX: number; // unit vector from this seat toward the hero centre
  towardY: number;
  ampMain: number; // sway amplitude along the toward-centre axis (the larger one)
  ampCross: number; // sway amplitude perpendicular to it (smaller — keeps it organic)
  freqMain: number; // rad/sec
  freqCross: number;
  phase: number;
}

const DRAG_THRESHOLD = 4; // px of movement before a press counts as a drag, not a click
const MAX_OFFSET = 200; // px — keeps a dragged box discoverable rather than lost off-screen
const MOMENTUM_DECAY = 0.93; // per-frame velocity decay after release
// Below 1180px (or on any device without real hover — tablets included) the ring
// itself shrinks (see globals.css); ambient drift and drag roam shrink with it so
// the sway/throw stays proportional to the smaller seats instead of swamping them.
const MOBILE_SCALE = 0.3;
const MOBILE_QUERY = "(max-width: 1180px), (hover: none)";

// Drives a panel's position: a gentle ambient drift (paused while hovered or
// dragged) plus real pointer dragging with a bit of release momentum. The box
// stays wherever it's dropped and keeps floating from there — it doesn't
// spring back to its corner. The ambient drift itself is elongated along each
// seat's toward-centre axis (see FLOAT_PARAMS.towardX/Y) so the sway reads as
// leaning toward the hero name rather than an arbitrary Lissajous wobble.
export function useFloatDrag(params: FloatParams) {
  const ref = useRef<HTMLDivElement | null>(null);
  const wasDraggedRef = useRef(false);
  const [revealed, setRevealed] = useState(false);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;

    const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const mobileMedia = window.matchMedia(MOBILE_QUERY);

    let raf = 0;
    const t0 = performance.now();
    let hovering = false;
    let dragging = false;
    let anchorX = 0;
    let anchorY = 0;
    let vx = 0;
    let vy = 0;
    let lastPX = 0;
    let lastPY = 0;
    let lastMoveT = 0;
    let startPX = 0;
    let startPY = 0;
    let pointerId: number | null = null;
    let frozenFloat = { x: 0, y: 0 };

    const clamp = (v: number) => {
      const max = mobileMedia.matches ? MAX_OFFSET * MOBILE_SCALE : MAX_OFFSET;
      return Math.max(-max, Math.min(max, v));
    };

    // perpendicular to the toward-centre axis, for the smaller cross-sway
    const crossX = -params.towardY;
    const crossY = params.towardX;

    const computeFloat = (t: number) => {
      const scale = mobileMedia.matches ? MOBILE_SCALE : 1;
      const main = params.ampMain * scale * Math.sin(t * params.freqMain + params.phase);
      const cross = params.ampCross * scale * Math.sin(t * params.freqCross + params.phase * 1.7);
      return {
        x: main * params.towardX + cross * crossX,
        y: main * params.towardY + cross * crossY,
      };
    };

    const frame = (now: number) => {
      const t = (now - t0) / 1000;
      if (!dragging) {
        if (!reduceMotion && !hovering) {
          frozenFloat = computeFloat(t);
        } else if (reduceMotion) {
          frozenFloat = { x: 0, y: 0 };
        }
        if (Math.abs(vx) > 0.5 || Math.abs(vy) > 0.5) {
          anchorX = clamp(anchorX + vx / 60);
          anchorY = clamp(anchorY + vy / 60);
          vx *= MOMENTUM_DECAY;
          vy *= MOMENTUM_DECAY;
        }
      }
      el.style.transform = `translate(${anchorX + frozenFloat.x}px, ${anchorY + frozenFloat.y}px)`;
      raf = requestAnimationFrame(frame);
    };
    raf = requestAnimationFrame(frame);

    const onEnter = () => {
      hovering = true;
    };
    const onLeave = () => {
      hovering = false;
    };

    const onPointerDown = (e: PointerEvent) => {
      if ((e.target as HTMLElement).closest("button")) return;
      dragging = true;
      wasDraggedRef.current = false;
      pointerId = e.pointerId;
      el.setPointerCapture(pointerId);
      startPX = e.clientX;
      startPY = e.clientY;
      lastPX = e.clientX;
      lastPY = e.clientY;
      lastMoveT = performance.now();
      vx = 0;
      vy = 0;
      el.classList.add("is-dragging");
    };
    const onPointerMove = (e: PointerEvent) => {
      if (!dragging) return;
      const dx = e.clientX - lastPX;
      const dy = e.clientY - lastPY;
      if (Math.abs(e.clientX - startPX) > DRAG_THRESHOLD || Math.abs(e.clientY - startPY) > DRAG_THRESHOLD) {
        wasDraggedRef.current = true;
      }
      anchorX = clamp(anchorX + dx);
      anchorY = clamp(anchorY + dy);
      const now = performance.now();
      const dt = Math.max(1, now - lastMoveT) / 1000;
      vx = reduceMotion ? 0 : dx / dt;
      vy = reduceMotion ? 0 : dy / dt;
      lastPX = e.clientX;
      lastPY = e.clientY;
      lastMoveT = now;
    };
    const onPointerUp = () => {
      if (!dragging) return;
      dragging = false;
      el.classList.remove("is-dragging");
      if (pointerId !== null) el.releasePointerCapture(pointerId);
      pointerId = null;
      if (reduceMotion) {
        vx = 0;
        vy = 0;
      }
    };

    el.addEventListener("pointerenter", onEnter);
    el.addEventListener("pointerleave", onLeave);
    el.addEventListener("pointerdown", onPointerDown);
    el.addEventListener("pointermove", onPointerMove);
    el.addEventListener("pointerup", onPointerUp);
    el.addEventListener("pointercancel", onPointerUp);

    return () => {
      cancelAnimationFrame(raf);
      el.removeEventListener("pointerenter", onEnter);
      el.removeEventListener("pointerleave", onLeave);
      el.removeEventListener("pointerdown", onPointerDown);
      el.removeEventListener("pointermove", onPointerMove);
      el.removeEventListener("pointerup", onPointerUp);
      el.removeEventListener("pointercancel", onPointerUp);
    };
  }, [
    params.towardX,
    params.towardY,
    params.ampMain,
    params.ampCross,
    params.freqMain,
    params.freqCross,
    params.phase,
  ]);

  // stagger each panel's entrance on load so all six don't pop in at once
  useEffect(() => {
    const delay = 60 + Math.random() * 900;
    const t = window.setTimeout(() => setRevealed(true), delay);
    return () => window.clearTimeout(t);
  }, []);

  return { ref, wasDraggedRef, revealed };
}

// six evenly-spaced seats on a ring around the name (see .mini-* in globals.css)
export type Corner = "top" | "ur" | "lr" | "bot" | "ll" | "ul";

// towardX/Y is the unit vector from each seat's rest position toward the hero
// centre (derived from the .mini-* margins in globals.css), so ampMain leans
// the sway that way instead of an arbitrary per-axis wobble.
export const FLOAT_PARAMS: Record<Corner, FloatParams> = {
  top: { towardX: 0, towardY: 1, ampMain: 32, ampCross: 15, freqMain: 0.27, freqCross: 0.21, phase: 0 },
  ur: { towardX: -0.94, towardY: 0.34, ampMain: 32, ampCross: 15, freqMain: 0.24, freqCross: 0.23, phase: 1.0 },
  lr: { towardX: -0.98, towardY: -0.18, ampMain: 32, ampCross: 15, freqMain: 0.26, freqCross: 0.2, phase: 2.1 },
  bot: { towardX: 0, towardY: -1, ampMain: 32, ampCross: 15, freqMain: 0.23, freqCross: 0.22, phase: 3.1 },
  ll: { towardX: 0.98, towardY: -0.18, ampMain: 32, ampCross: 15, freqMain: 0.25, freqCross: 0.19, phase: 4.2 },
  ul: { towardX: 0.94, towardY: 0.34, ampMain: 32, ampCross: 15, freqMain: 0.22, freqCross: 0.24, phase: 5.2 },
};
