// Inverted Pendulum (Pendulum-v1 swing-up). A single rod hinged at a pivot; the
// policy applies a torque limited below what's needed to lift it directly, so it
// must pump energy to swing up and then balance at the top. Continuous action,
// dense reward = height (−cost), so the return climbs as it learns to get — and
// stay — upright.

import type { RLEnv } from "./types";

const G = 10;
const L = 1;
const M = 1;
const B = 0.08; // damping
const MAX_TORQUE = 25; // generous authority: ARS can drive up and hold within a few
//                        generations, so the swing-up is learned in the ~5-10s a
//                        viewer watches rather than staying stuck at the bottom
const DT = 0.05;
const MAX_STEPS = 160; // shorter episodes → more ARS generations per second → faster
//                        visible learning

const clamp = (v: number, a: number, b: number) => (v < a ? a : v > b ? b : v);

export class PendulumEnv implements RLEnv {
  readonly obsDim = 4;
  readonly actDim = 1;
  readonly title = "Pendulum-v1 · ARS";
  w = 0;
  h = 0;

  th = Math.PI; // angle from upright (0 = up, π = hanging down)
  thd = 0;
  steps = 0;

  setSize(w: number, h: number) {
    this.w = w;
    this.h = h;
  }

  reset() {
    // Start slightly off the bottom (not the exact hanging equilibrium). At the
    // dead-bottom a near-zero policy produces zero torque and the rod never
    // moves — "doing nothing" — and every ARS direction scores the same, so
    // there's no gradient to climb. This small offset gives visible motion and a
    // reward signal ARS can actually follow up toward the top.
    this.th = Math.PI - 0.25;
    this.thd = 0;
    this.steps = 0;
  }

  step(action: number[]): { reward: number; done: boolean } {
    const u = clamp(action[0], -1, 1) * MAX_TORQUE;
    // th measured from upright; gravity is destabilising near the top and
    // restoring near the bottom → +(g/l) sin(th)
    const thacc =
      (G / L) * Math.sin(this.th) + u / (M * L * L) - B * this.thd;
    this.thd += DT * thacc;
    this.thd = clamp(this.thd, -12, 12);
    this.th += DT * this.thd;
    this.steps++;

    // Reward is dominated by height so the pump-up isn't punished; a bonus for
    // being near the top rewards actually balancing there, and a tiny velocity
    // term prefers a steady hold once up. The +1 offset keeps the per-step reward
    // non-negative (0 hanging → up to 3.5 balanced) so the HUD "Return" climbs
    // from 0 upward as it learns, instead of sitting at a clamped 000.
    const height = Math.cos(this.th); // +1 up, −1 down
    // strong, slightly-wider bonus for being near the top so ARS is pulled toward
    // a policy that actually holds it upright rather than just swinging through
    const upright = height > 0.9 ? 2.5 : 0;
    const reward =
      height + 1 + upright - 0.0025 * this.thd * this.thd - 0.0005 * u * u;
    return { reward, done: this.steps >= MAX_STEPS };
  }

  getObs(): number[] {
    // The last feature (thd·cos θ) lets the linear policy add energy when the
    // pendulum is low and remove it when high — resolving the swing-up vs.
    // balance sign conflict a plain [sin, cos, thd] policy can't.
    return [
      Math.sin(this.th),
      Math.cos(this.th),
      this.thd * 0.25,
      this.thd * Math.cos(this.th) * 0.25,
    ];
  }

  draw(ctx: CanvasRenderingContext2D) {
    const w = this.w;
    const h = this.h;
    const px = w / 2;
    const py = h * 0.54;
    const L2 = h * 0.34;
    const bx = px + Math.sin(this.th) * L2;
    const by = py - Math.cos(this.th) * L2;
    // target marker at the top
    ctx.strokeStyle = "rgba(17,17,17,0.12)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(px - 8, py - L2);
    ctx.lineTo(px + 8, py - L2);
    ctx.stroke();
    // rod
    ctx.strokeStyle = "#111";
    ctx.lineWidth = 5;
    ctx.lineCap = "round";
    ctx.beginPath();
    ctx.moveTo(px, py);
    ctx.lineTo(bx, by);
    ctx.stroke();
    // bob
    ctx.fillStyle = "#E12D1A";
    ctx.beginPath();
    ctx.arc(bx, by, 6, 0, 6.283);
    ctx.fill();
    // pivot
    ctx.fillStyle = "#fff";
    ctx.beginPath();
    ctx.arc(px, py, 3.2, 0, 6.283);
    ctx.fill();
    ctx.strokeStyle = "#111";
    ctx.lineWidth = 1.4;
    ctx.stroke();
  }
}
