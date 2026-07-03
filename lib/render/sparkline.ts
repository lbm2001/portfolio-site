// The RETURN / EPISODE reward plot shared by every demo panel. Min–max scaled so
// it shows the trend regardless of each task's reward range (CartPole ~0..220,
// Pendulum can be negative, etc.).
export function drawSparkline(
  ctx: CanvasRenderingContext2D,
  w: number,
  h: number,
  history: number[],
  label = "RETURN / EPISODE"
) {
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = "rgba(17,17,17,0.32)";
  ctx.font = "600 8px ui-monospace,Menlo,monospace";
  ctx.textAlign = "left";
  ctx.fillText(label, 3, 9);
  const n = history.length;
  if (n > 1) {
    let mn = Infinity;
    let mx = -Infinity;
    for (const v of history) {
      if (v < mn) mn = v;
      if (v > mx) mx = v;
    }
    const rng = mx - mn || 1;
    ctx.beginPath();
    history.forEach((v, i) => {
      const px = w * (i / Math.max(n - 1, 1));
      const py = h - 3 - (h - 14) * ((v - mn) / rng);
      if (i === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    });
    ctx.strokeStyle = "rgba(225,45,26,0.55)";
    ctx.lineWidth = 1.4;
    ctx.lineJoin = "round";
    ctx.stroke();
  }
  ctx.strokeStyle = "rgba(17,17,17,0.06)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(0, h - 0.5);
  ctx.lineTo(w, h - 0.5);
  ctx.stroke();
}
