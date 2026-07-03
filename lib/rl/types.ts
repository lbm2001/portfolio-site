// A minimal environment interface so one ARS trainer can drive any of the demos
// (CartPole, Pendulum, MountainCar, BipedalWalker). Each env owns its physics
// and how it draws itself; the trainer and the UI panel are shared.

export interface StepResult {
  reward: number;
  done: boolean;
}

export interface RLEnv {
  readonly obsDim: number;
  readonly actDim: number;
  /** Short label, e.g. "CartPole-v1 · balance". */
  readonly title: string;
  w: number;
  h: number;
  setSize(w: number, h: number): void;
  reset(): void;
  step(action: number[]): StepResult;
  getObs(): number[];
  /** Draw the current state into a canvas of size (this.w × this.h). */
  draw(ctx: CanvasRenderingContext2D): void;
  /** Optional display-only disturbance so the rendered agent stumbles/varies. */
  disturbAmp?: number;
}

export type EnvFactory = () => RLEnv;
