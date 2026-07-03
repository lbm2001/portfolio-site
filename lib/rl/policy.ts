// Running mean/variance (Welford) used to whiten observations — the "V2" part of
// Augmented Random Search. Stable observation statistics make a *linear* policy
// enough to learn these control tasks.
export class RunningNorm {
  n = 0;
  mean: Float64Array;
  m2: Float64Array; // sum of squared deviations

  constructor(dim: number) {
    this.mean = new Float64Array(dim);
    this.m2 = new Float64Array(dim);
  }

  observe(x: number[]) {
    this.n++;
    for (let i = 0; i < x.length; i++) {
      const d = x[i] - this.mean[i];
      this.mean[i] += d / this.n;
      this.m2[i] += d * (x[i] - this.mean[i]);
    }
  }

  normalize(x: number[]): number[] {
    const out = new Array(x.length);
    for (let i = 0; i < x.length; i++) {
      const std = this.n > 1 ? Math.sqrt(this.m2[i] / (this.n - 1)) : 1;
      out[i] = (x[i] - this.mean[i]) / (std > 1e-6 ? std : 1);
    }
    return out;
  }
}

// Linear policy: action = clip(W · [obs, 1]). W has shape actDim × (obsDim+1),
// the trailing column being a bias. Stored as a flat Float64Array so ARS can
// perturb it cheaply.
export function policySize(actDim: number, obsDim: number): number {
  return actDim * (obsDim + 1);
}

export function act(
  weights: Float64Array,
  normObs: number[],
  actDim: number,
  obsDim: number
): number[] {
  const out = new Array(actDim);
  const cols = obsDim + 1;
  for (let a = 0; a < actDim; a++) {
    let s = weights[a * cols + obsDim]; // bias
    for (let o = 0; o < obsDim; o++) s += weights[a * cols + o] * normObs[o];
    out[a] = s < -1 ? -1 : s > 1 ? 1 : s;
  }
  return out;
}
