export function sizeCanvas(canvas: HTMLCanvasElement): { w: number; h: number } {
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  const r = canvas.getBoundingClientRect();
  const pw = Math.max(1, Math.round(r.width * dpr));
  const ph = Math.max(1, Math.round(r.height * dpr));
  if (canvas.width !== pw || canvas.height !== ph) {
    canvas.width = pw;
    canvas.height = ph;
    const ctx = canvas.getContext("2d");
    if (ctx) ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }
  return { w: r.width, h: r.height };
}
