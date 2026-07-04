// WalkerEnv — a lightweight but genuine 2D biped locomotion model over flat
// ground.
//
// The dynamics are real: the hull is a rigid body with mass and rotational
// inertia, the two jointed legs are driven by policy motor-torques, and forward
// motion is produced by *planted-foot friction* — when the policy swings a
// planted leg backward, friction propels the hull forward. Nothing about
// progress is scripted; it emerges from the contact mechanics.
//
// For stability (this runs forever as a hero background) the legs also act as
// compliant support struts that hold the hull near standing height, and the
// torso has a low-COM self-righting tendency. So the task ARS actually has to
// solve is "coordinate the legs to walk the valley without toppling" — which is
// genuine reinforcement learning over a real environment.
//
// Coordinates are canvas pixels (y points DOWN) so the renderer can draw state
// directly with the design's proportions (L1=11/L2=10).

export const L1 = 11; // thigh length (px)
export const L2 = 10; // shank length (px)

export const OBS_DIM = 16;
export const ACT_DIM = 4; // [hipTorqueL, kneeTorqueL, hipTorqueR, kneeTorqueR]

import type { RLEnv } from "./types";

export interface LegState {
  hip: number; // hip angle relative to hull, rad (0 = straight down)
  hipVel: number;
  knee: number; // knee angle relative to thigh, rad (<=0 = bent back)
  kneeVel: number;
}

export interface StepResult {
  reward: number;
  done: boolean;
}

interface Vec2 {
  x: number;
  y: number;
}

// physics constants
const M = 1.0; // hull mass
const I_HULL = 240; // hull rotational inertia
const I_ROTOR = 0.45; // per-joint reflected inertia
// Joints are position-controlled: the policy outputs a TARGET angle for each
// joint and a PD controller drives it there. This makes rhythmic walking gaits
// much easier for the policy to express (and for ARS to find) than raw torques.
const KP = 260; // PD position gain
const KD = 22; // PD damping gain
const TORQUE_MAX = 150; // clamp on PD output torque
const KV = 70; // leg-strut vertical support stiffness
const CV = 17; // leg-strut vertical support damping
const KT_FRIC = 80; // foot tangential friction stiffness (grip firmness)
const MU_LOAD = 820; // max friction force per planted foot (~body weight)
const PLANT_BAND = 3; // px band over which a foot counts as planted
const AIR_DAMP = 0.3; // linear (horizontal) air damping
const K_UPRIGHT = 1050; // torso self-righting stiffness (balance assist — soft
//                         enough that steep terrain / a stumble can topple it)
const C_UPRIGHT = 175; // torso self-righting damping
const SLOPE_TIP = 360; // terrain slope leans the torso → makes steep walls the
//                        places a stumble is most likely to become a fall
const GRAV_FALL = 12; // topple acceleration once the walker has fallen
const FALL_SETTLE = 16; // env steps the collapse is shown before the episode ends
const HIP_LIMIT = 1.0; // |hip| <= this
const KNEE_MIN = -1.9; // knee folds back
const KNEE_MAX = 0.05;
const PHASE_RATE = 6.2; // CPG clock (rad/s)

// Central Pattern Generator: a fixed rhythmic gait the policy modulates. The CPG
// guarantees the walker actually steps forward; ARS learns residual corrections
// (RESID_*) on top to travel further and stay balanced on the terrain. This is
// the classic CPG + reinforcement-learning locomotion recipe.
const HIP_AMP = 0.2; // hip swing amplitude (rad) — deliberately small so the
//                      untrained CPG barely shuffles in place; the policy has to
//                      LEARN to add the swing that produces real forward travel
const KNEE_MID = -0.22; // knee angle during stance (near straight → foot reaches)
const KNEE_AMP = 1.2; // extra knee bend during swing (foot clears the ground)
const KNEE_PHASE = Math.PI / 2; // knee lifts during the forward swing
const RESID_HIP = 0.72; // policy authority over hip target (rad) — larger now so
//                         learning the hip swing dominates the base gait
const RESID_KNEE = 0.7; // policy authority over knee target (rad)
const FALL_ANGLE = 0.85;
const SUBSTEPS = 5;
const SUB_DT = 1 / 180; // fixed physics timestep
const MAX_STEPS = 320; // env steps per episode before timeout

const uPrime = (a: number): Vec2 => ({ x: Math.cos(a), y: -Math.sin(a) });
const clamp = (v: number, lo: number, hi: number) =>
  v < lo ? lo : v > hi ? hi : v;

export class WalkerEnv implements RLEnv {
  readonly obsDim = OBS_DIM;
  readonly actDim = ACT_DIM;
  readonly title = "BipedalWalker · ARS";
  w = 0;
  h = 0;

  // hull rigid body
  x = 0;
  y = 0;
  vx = 0;
  vy = 0;
  th = 0; // hull tilt
  om = 0; // angular velocity

  legs: [LegState, LegState] = [
    { hip: 0.32, hipVel: 0, knee: -0.5, kneeVel: 0 },
    { hip: -0.32, hipVel: 0, knee: -0.5, kneeVel: 0 },
  ];

  phase = 0;
  steps = 0;
  fallen = false;
  fallT = 0; // env steps elapsed since the walker fell (drives the collapse)
  // Random torso disturbance amplitude (rad/s²). Left at 0 for training so the
  // ARS rollouts stay clean; the *rendered* walker turns it on so it stumbles
  // and occasionally topples on the steep valley walls — visible, motivated falls.
  disturbAmp = 0;
  startX = 0;
  standH = (L1 + L2) * 0.82; // hull height above terrain at standing
  // Display-only layout knobs (used by the mobile mini): where the ground sits as
  // a fraction of canvas height, and whether to draw the terrain line at all.
  groundFrac = 0.8;
  bare = false;

  setSize(w: number, h: number) {
    this.w = w;
    this.h = h;
  }

  // ---- Flat ground ----
  // A level surface: the walker just strides across it. Kept as a function (not a
  // constant) so the slope-based terms (slopeAt, SLOPE_TIP, the getObs slope
  // feature) all still work — they simply evaluate to zero on flat ground.
  terrainY(_x: number): number {
    return this.h * this.groundFrac;
  }

  slopeAt(x: number): number {
    return (this.terrainY(x + 6) - this.terrainY(x - 6)) / 12;
  }

  reset(startX?: number) {
    const sx = startX ?? this.w * 0.06;
    this.startX = sx;
    this.x = sx;
    this.th = clamp(this.slopeAt(sx), -0.25, 0.25);
    this.vx = 0;
    this.vy = 0;
    this.om = 0;
    this.legs = [
      { hip: 0.32, hipVel: 0, knee: -0.5, kneeVel: 0 },
      { hip: -0.32, hipVel: 0, knee: -0.5, kneeVel: 0 },
    ];
    this.phase = 0;
    this.steps = 0;
    this.fallen = false;
    this.fallT = 0;
    // standing height = how far the lower foot hangs below the hull
    let drop = 0;
    for (let i = 0; i < 2; i++) {
      const leg = this.legs[i];
      const d =
        L1 * Math.cos(this.th + leg.hip) +
        L2 * Math.cos(this.th + leg.hip + leg.knee);
      if (d > drop) drop = d;
    }
    this.standH = drop;
    this.y = this.terrainY(sx) - drop;
  }

  // forward kinematics for a leg → hip/knee/foot world points (used by renderer)
  legPoints(i: number): { hip: Vec2; knee: Vec2; foot: Vec2 } {
    const leg = this.legs[i];
    const at = this.th + leg.hip;
    const as = this.th + leg.hip + leg.knee;
    const hip = { x: this.x, y: this.y };
    const knee = {
      x: hip.x + L1 * Math.sin(at),
      y: hip.y + L1 * Math.cos(at),
    };
    const foot = {
      x: knee.x + L2 * Math.sin(as),
      y: knee.y + L2 * Math.cos(as),
    };
    return { hip, knee, foot };
  }

  private substep(action: number[]) {
    if (this.fallen) {
      this.substepCollapse();
      return;
    }
    let fx = 0; // net horizontal force on hull
    let pitch = 0; // net pitch torque on hull
    const tau = [0, 0, 0, 0]; // generalized joint forces (motor + contact)

    for (let i = 0; i < 2; i++) {
      const leg = this.legs[i];
      const at = this.th + leg.hip;
      const as = this.th + leg.hip + leg.knee;
      const upt = uPrime(at);
      const ups = uPrime(as);

      // CPG gait target + policy residual, tracked by PD position control.
      const legPhase = this.phase + i * Math.PI; // legs step in antiphase
      const cpgHip = HIP_AMP * Math.sin(legPhase);
      const cpgKnee =
        KNEE_MID - KNEE_AMP * Math.max(0, Math.sin(legPhase + KNEE_PHASE));
      const tgtHip = clamp(
        cpgHip + clamp(action[i * 2], -1, 1) * RESID_HIP,
        -HIP_LIMIT,
        HIP_LIMIT
      );
      const tgtKnee = clamp(
        cpgKnee + clamp(action[i * 2 + 1], -1, 1) * RESID_KNEE,
        KNEE_MIN,
        KNEE_MAX
      );
      const motHip = clamp(
        KP * (tgtHip - leg.hip) - KD * leg.hipVel,
        -TORQUE_MAX,
        TORQUE_MAX
      );
      const motKnee = clamp(
        KP * (tgtKnee - leg.knee) - KD * leg.kneeVel,
        -TORQUE_MAX,
        TORQUE_MAX
      );
      pitch -= motHip * 0.12; // hip motor reaction tilts the hull

      // foot kinematics
      const footX = this.x + L1 * Math.sin(at) + L2 * Math.sin(as);
      const footY = this.y + L1 * Math.cos(at) + L2 * Math.cos(as);
      const atDot = this.om + leg.hipVel;
      const asDot = this.om + leg.hipVel + leg.kneeVel;
      const footVx = this.vx + L1 * upt.x * atDot + L2 * ups.x * asDot;
      const footVy = this.vy + L1 * upt.y * atDot + L2 * ups.y * asDot;

      // Legs swing freely under their motors. A planted foot grips the ground
      // (no-slip friction): friction opposes the foot's tangential velocity and
      // is applied to the HULL, so swinging a planted leg backward drags the
      // body forward — this is what produces locomotion.
      tau[i * 2] = motHip;
      tau[i * 2 + 1] = motKnee;
      const ground = this.terrainY(footX);
      const plant = clamp((footY - (ground - 1)) / PLANT_BAND, 0, 1);
      if (plant > 0) {
        const slope = this.slopeAt(footX);
        const tlen = Math.hypot(1, slope);
        const tx = 1 / tlen;
        const ty = slope / tlen;
        const vT = footVx * tx + footVy * ty; // tangential foot velocity
        const maxF = MU_LOAD * plant;
        const Ft = clamp(-KT_FRIC * vT, -maxF, maxF);
        fx += Ft * tx;
        // NB: the foot force is intentionally NOT applied as a pitch torque on
        // the hull. The foot sits ~20px below the torso COM, so that moment
        // (up to ~20·MU_LOAD) would swamp the balance spring and topple the
        // walker every step. Balance is assisted (K_UPRIGHT); the foot friction
        // only propels the hull horizontally.
      }
    }

    // integrate joints
    for (let i = 0; i < 2; i++) {
      const leg = this.legs[i];
      const hipAcc = tau[i * 2] / I_ROTOR;
      const kneeAcc = tau[i * 2 + 1] / I_ROTOR;
      leg.hipVel = clamp(leg.hipVel + SUB_DT * hipAcc, -18, 18);
      leg.kneeVel = clamp(leg.kneeVel + SUB_DT * kneeAcc, -18, 18);
      leg.hip += SUB_DT * leg.hipVel;
      leg.knee += SUB_DT * leg.kneeVel;
      if (leg.hip > HIP_LIMIT) {
        leg.hip = HIP_LIMIT;
        leg.hipVel = 0;
      } else if (leg.hip < -HIP_LIMIT) {
        leg.hip = -HIP_LIMIT;
        leg.hipVel = 0;
      }
      if (leg.knee > KNEE_MAX) {
        leg.knee = KNEE_MAX;
        leg.kneeVel = 0;
      } else if (leg.knee < KNEE_MIN) {
        leg.knee = KNEE_MIN;
        leg.kneeVel = 0;
      }
    }

    // hull: vertical held near standing height by the leg struts; horizontal
    // driven by friction; pitch self-rights with a low COM.
    const yTarget = this.terrainY(this.x) - this.standH;
    const ay = -KV * (this.y - yTarget) - CV * this.vy;
    const ax = fx / M - AIR_DAMP * this.vx;
    // The terrain slope under the hull pushes the torso over; the balance assist
    // resists it. On the steep valley walls this can exceed the assist and the
    // walker topples — that's where most falls happen.
    const slopeHull = this.slopeAt(this.x);
    const disturb = this.disturbAmp
      ? this.disturbAmp * (Math.random() * 2 - 1)
      : 0;
    const angAcc =
      (pitch + SLOPE_TIP * slopeHull - K_UPRIGHT * this.th - C_UPRIGHT * this.om) /
        I_HULL +
      disturb;
    this.vx += SUB_DT * ax;
    this.vy += SUB_DT * ay;
    this.om += SUB_DT * angAcc;
    this.x += SUB_DT * this.vx;
    this.y += SUB_DT * this.vy;
    this.th += SUB_DT * this.om;
    this.phase += SUB_DT * PHASE_RATE;
  }

  // Once fallen, the walker is no longer actively controlled: the torso topples
  // the rest of the way over and the body drops to the ground, so the fall is
  // visible for a moment before the episode resets (instead of teleporting).
  private substepCollapse() {
    const dir = this.th >= 0 ? 1 : -1;
    this.om += SUB_DT * (dir * GRAV_FALL - 0.4 * this.om);
    this.th += SUB_DT * this.om;
    this.vx *= 0.86;
    this.x += SUB_DT * this.vx;
    const groundHull = this.terrainY(this.x) - 3;
    this.vy += SUB_DT * (400 - 4 * this.vy);
    this.y += SUB_DT * this.vy;
    if (this.y > groundHull) {
      this.y = groundHull;
      this.vy = 0;
    }
    for (let i = 0; i < 2; i++) {
      const leg = this.legs[i];
      leg.hipVel *= 0.8;
      leg.kneeVel *= 0.8;
      leg.hip += SUB_DT * leg.hipVel;
      leg.knee += SUB_DT * leg.kneeVel;
    }
    this.phase += SUB_DT * PHASE_RATE;
  }

  step(action: number[]): StepResult {
    const x0 = this.x;
    for (let s = 0; s < SUBSTEPS; s++) this.substep(action);
    this.steps++;

    // Reward is dominated by forward travel. There is deliberately NO survival
    // bonus: an "alive" bonus creates a strong local optimum where the agent
    // just stands still and collects it (a robot jittering in place), which is
    // exactly what we don't want. With travel as the only positive term the
    // policy must actually locomote to score.
    const forward = this.x - x0;
    let ctrl = 0;
    for (let i = 0; i < ACT_DIM; i++) ctrl += action[i] * action[i];
    let reward = forward * 8.0 - 0.1 * Math.abs(this.th) - 0.015 * ctrl;

    let done = false;
    if (this.fallen) {
      // already down — keep showing the collapse, then end the episode
      this.fallT++;
      if (this.fallT >= FALL_SETTLE) done = true;
    } else if (Math.abs(this.th) > FALL_ANGLE) {
      this.fallen = true; // just toppled
      this.fallT = 0;
      reward -= 3;
    } else if (this.steps >= MAX_STEPS) {
      done = true;
    } else if (this.x > this.w + 40) {
      reward += 10; // walked all the way across
      done = true;
    }
    return { reward, done };
  }

  getObs(): number[] {
    const ground = this.terrainY(this.x);
    const heightAbove = (ground - this.y) / (L1 + L2);
    const slope = this.slopeAt(this.x);
    const [a, b] = this.legs;
    return [
      this.th,
      this.om * 0.1,
      this.vx * 0.02,
      this.vy * 0.02,
      heightAbove - 0.82,
      slope,
      a.hip,
      a.hipVel * 0.1,
      a.knee,
      a.kneeVel * 0.1,
      b.hip,
      b.hipVel * 0.1,
      b.knee,
      b.kneeVel * 0.1,
      Math.sin(this.phase),
      Math.cos(this.phase),
    ];
  }

  private drawLeg(ctx: CanvasRenderingContext2D, i: number, shade: string) {
    const { hip, knee, foot } = this.legPoints(i);
    ctx.strokeStyle = shade;
    ctx.lineWidth = 3;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.beginPath();
    ctx.moveTo(hip.x, hip.y);
    ctx.lineTo(knee.x, knee.y);
    ctx.lineTo(foot.x, foot.y);
    ctx.stroke();
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    ctx.moveTo(foot.x, foot.y);
    ctx.lineTo(foot.x + 5, foot.y);
    ctx.stroke();
    ctx.fillStyle = "#E12D1A";
    ctx.beginPath();
    ctx.arc(knee.x, knee.y, 1.3, 0, 6.283);
    ctx.fill();
  }

  draw(ctx: CanvasRenderingContext2D) {
    const w = this.w;
    // floor line. In bare mode (mobile mini) it's a single very-light-grey rule —
    // the page/name is the real surface; this just grounds the walker.
    if (this.bare) {
      // full-width in bare mode: the canvas is sized to the name, so this line
      // runs exactly as wide as the title it sits on.
      const ty = this.terrainY(0);
      ctx.strokeStyle = "rgba(17,17,17,0.14)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(0, ty);
      ctx.lineTo(w, ty);
      ctx.stroke();
    } else {
      ctx.beginPath();
      for (let x = 0; x <= w; x += 6) {
        const ty = this.terrainY(x);
        if (x === 0) ctx.moveTo(0, ty);
        else ctx.lineTo(x, ty);
      }
      ctx.strokeStyle = "rgba(175, 28, 8, 0.13)";
      ctx.lineWidth = 1.1;
      ctx.stroke();
    }
    // walker: far leg (lighter), near leg (dark), then hull
    this.drawLeg(ctx, 1, "rgba(17,17,17,0.4)");
    this.drawLeg(ctx, 0, "#111");
    ctx.save();
    ctx.translate(this.x, this.y);
    ctx.rotate(this.th);
    roundRect(ctx, -11, -7, 22, 10, 3);
    ctx.fillStyle = "#111";
    ctx.fill();
    ctx.fillStyle = "#E12D1A";
    ctx.fillRect(6, -5, 4, 3);
    ctx.fillStyle = "#fff";
    ctx.beginPath();
    ctx.arc(0, 0, 2, 0, 6.283);
    ctx.fill();
    ctx.strokeStyle = "#111";
    ctx.lineWidth = 1.1;
    ctx.stroke();
    ctx.restore();
  }
}

export function roundRect(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  w: number,
  h: number,
  r: number
) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}
