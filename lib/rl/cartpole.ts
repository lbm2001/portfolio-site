// CartPole — the canonical first RL task. Balance a pole hinged on a cart by
// pushing the cart left/right. Classic control dynamics (Barto/Sutton, as in
// Gym's CartPole-v1), with a continuous force so the same linear-policy ARS
// trainer applies. Reward is +1 per timestep upright, so the return is simply
// "how long it balanced" — a clean learning curve.

import type { RLEnv } from "./types";
import { roundRect } from "./env";

const G = 9.8;
const MC = 1.0; // cart mass
const MP = 0.1; // pole mass
const LEN = 0.5; // half pole length
const PML = MP * LEN;
const MT = MC + MP;
const FORCE = 10;
const TAU = 0.02;
const X_LIMIT = 2.4;
const TH_LIMIT = 0.21; // ~12°
const MAX_STEPS = 220;

const clamp = (v: number, a: number, b: number) => (v < a ? a : v > b ? b : v);

export class CartPoleEnv implements RLEnv {
  readonly obsDim = 4;
  readonly actDim = 1;
  readonly title = "CartPole-v1 · balance · ARS";
  w = 0;
  h = 0;

  x = 0;
  xd = 0;
  th = 0; // pole angle from vertical
  thd = 0;
  steps = 0;

  setSize(w: number, h: number) {
    this.w = w;
    this.h = h;
  }

  reset() {
    // fixed small tilt so paired ARS rollouts share a start state
    this.x = 0;
    this.xd = 0;
    this.th = 0.05;
    this.thd = 0;
    this.steps = 0;
  }

  step(action: number[]): { reward: number; done: boolean } {
    const force = clamp(action[0], -1, 1) * FORCE;
    const ct = Math.cos(this.th);
    const st = Math.sin(this.th);
    const temp = (force + PML * this.thd * this.thd * st) / MT;
    const thacc = (G * st - ct * temp) / (LEN * (4 / 3 - (MP * ct * ct) / MT));
    const xacc = temp - (PML * thacc * ct) / MT;
    this.x += TAU * this.xd;
    this.xd += TAU * xacc;
    this.th += TAU * this.thd;
    this.thd += TAU * thacc;
    this.steps++;

    const done =
      Math.abs(this.x) > X_LIMIT ||
      Math.abs(this.th) > TH_LIMIT ||
      this.steps >= MAX_STEPS;
    // +1 for every step it stays up
    const reward = 1;
    return { reward, done };
  }

  getObs(): number[] {
    return [this.x, this.xd, this.th * 3, this.thd];
  }

  draw(ctx: CanvasRenderingContext2D) {
    const w = this.w;
    const h = this.h;
    const gy = h * 0.72;
    const sx = Math.min(w * 0.16, 120);
    const cx = w / 2 + this.x * sx;
    // track
    ctx.strokeStyle = "rgba(17,17,17,0.16)";
    ctx.lineWidth = 1.4;
    ctx.beginPath();
    ctx.moveTo(w * 0.1, gy);
    ctx.lineTo(w * 0.9, gy);
    ctx.stroke();
    ctx.strokeStyle = "rgba(17,17,17,0.1)";
    ctx.beginPath();
    ctx.moveTo(w / 2, gy - 5);
    ctx.lineTo(w / 2, gy + 5);
    ctx.stroke();
    // cart
    const cw = Math.min(56, w * 0.22);
    const chh = 24;
    const cyTop = gy - chh;
    roundRect(ctx, cx - cw / 2, cyTop, cw, chh, 5);
    ctx.fillStyle = "#111";
    ctx.fill();
    ctx.fillStyle = "rgba(17,17,17,0.55)";
    ctx.beginPath();
    ctx.arc(cx - cw / 2 + 12, gy, 3.5, 0, 6.283);
    ctx.fill();
    ctx.beginPath();
    ctx.arc(cx + cw / 2 - 12, gy, 3.5, 0, 6.283);
    ctx.fill();
    // pole
    const L = h * 0.42;
    const hx = cx;
    const hy = cyTop;
    const tx = hx + Math.sin(this.th) * L;
    const ty = hy - Math.cos(this.th) * L;
    ctx.strokeStyle = "#E12D1A";
    ctx.lineWidth = 6;
    ctx.lineCap = "round";
    ctx.beginPath();
    ctx.moveTo(hx, hy);
    ctx.lineTo(tx, ty);
    ctx.stroke();
    ctx.fillStyle = "#fff";
    ctx.beginPath();
    ctx.arc(hx, hy, 3, 0, 6.283);
    ctx.fill();
    ctx.strokeStyle = "#111";
    ctx.lineWidth = 1.3;
    ctx.stroke();
  }
}
