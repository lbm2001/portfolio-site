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
  /** Optional display-only flag: let the agent topple visibly before reset. */
  showFall?: boolean;
  /** Optional display-only: the ground height as a fraction of the canvas (so a
   *  demo can sit the agent near the bottom). Defaults per env. */
  groundFrac?: number;
  /** Optional display-only: suppress the env's own ground/track line, e.g. when
   *  something else (the page) is meant to read as the floor. */
  bare?: boolean;
}

export type EnvFactory = () => RLEnv;
