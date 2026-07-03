import { useEffect, useRef } from "react";

export interface FloatParams {
  ampX: number;
  ampY: number;
  freqX: number; // rad/sec
  freqY: number;
  phase: number;
}

const DRAG_THRESHOLD = 4; // px of movement before a press counts as a drag, not a click
const MAX_OFFSET = 200; // px — keeps a dragged box discoverable rather than lost off-screen
const MOMENTUM_DECAY = 0.93; // per-frame velocity decay after release

// Drives a panel's position: a gentle ambient drift (paused while hovered or
// dragged) plus real pointer dragging with a bit of release momentum. The box
// stays wherever it's dropped and keeps floating from there — it doesn't
// spring back to its corner.
export function useFloatDrag(params: FloatParams) {
  const ref = useRef<HTMLDivElement | null>(null);
  const wasDraggedRef = useRef(false);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;

    const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

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

    const clamp = (v: number) => Math.max(-MAX_OFFSET, Math.min(MAX_OFFSET, v));

    const computeFloat = (t: number) => ({
      x: params.ampX * Math.sin(t * params.freqX + params.phase),
      y: params.ampY * Math.sin(t * params.freqY + params.phase * 1.7),
    });

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
  }, [params.ampX, params.ampY, params.freqX, params.freqY, params.phase]);

  return { ref, wasDraggedRef };
}

// six evenly-spaced seats on a ring around the name (see .mini-* in globals.css)
export type Corner = "top" | "ur" | "lr" | "bot" | "ll" | "ul";

export const FLOAT_PARAMS: Record<Corner, FloatParams> = {
  top: { ampX: 34, ampY: 26, freqX: 0.27, freqY: 0.21, phase: 0 },
  ur: { ampX: -32, ampY: 28, freqX: 0.24, freqY: 0.23, phase: 1.0 },
  lr: { ampX: -34, ampY: -26, freqX: 0.26, freqY: 0.2, phase: 2.1 },
  bot: { ampX: 32, ampY: -28, freqX: 0.23, freqY: 0.22, phase: 3.1 },
  ll: { ampX: 34, ampY: -26, freqX: 0.25, freqY: 0.19, phase: 4.2 },
  ul: { ampX: 32, ampY: 28, freqX: 0.22, freqY: 0.24, phase: 5.2 },
};
