import type { MLP, LossType, Sample } from "./mlp";

export interface NNTask {
  readonly title: string;
  readonly netSizes: number[]; // layer widths, e.g. [2,8,8,1] — architecture for the anatomy view
  readonly lossType: LossType;
  w: number;
  h: number;
  net: MLP; // exposed so the anatomy view can probe the live weights
  setSize(w: number, h: number): void;
  reset(): void;
  trainStep(): number; // one gradient step (may cover a mini-batch internally); returns loss
  converged(loss: number, step: number): boolean; // whether to reset and start learning over
  draw(ctx: CanvasRenderingContext2D): void;
  currentSample(): Sample; // a representative input to visualize the forward/backward pass on
}

export type NNTaskFactory = () => NNTask;
