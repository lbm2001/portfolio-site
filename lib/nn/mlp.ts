// A small hand-rolled multilayer perceptron trained by plain backprop + SGD —
// no ML library, same "nothing pre-built" ethos as the ARS/RL side. Supports
// the three loss/output combinations the four demos need; each combination
// uses the standard simplification where the output-layer gradient collapses
// to (yhat - y): sigmoid+BCE, softmax+categorical-CE, and linear+MSE.

export type Activation = "tanh" | "sigmoid" | "linear" | "softmax";
export type LossType = "mse" | "bce" | "ce";

function applyAct(a: Activation, z: number[]): number[] {
  switch (a) {
    case "tanh":
      return z.map(Math.tanh);
    case "sigmoid":
      return z.map((v) => 1 / (1 + Math.exp(-v)));
    case "linear":
      return z.slice();
    case "softmax": {
      const m = Math.max(...z);
      const exps = z.map((v) => Math.exp(v - m));
      const s = exps.reduce((acc, v) => acc + v, 0);
      return exps.map((v) => v / s);
    }
  }
}

// Derivative of the activation w.r.t. its input, expressed in terms of the
// activation's own output (convenient closed forms for tanh/sigmoid).
function actDerivFromOutput(a: Activation, out: number[]): number[] {
  switch (a) {
    case "tanh":
      return out.map((o) => 1 - o * o);
    case "sigmoid":
      return out.map((o) => o * (1 - o));
    case "linear":
    case "softmax":
      return out.map(() => 1);
  }
}

export interface Sample {
  x: number[];
  y: number[];
}

interface BackwardResult {
  as: number[][]; // per-layer activations, as[0] = input
  deltas: number[][]; // per-layer dLoss/dz, deltas[l] belongs to layer l's output
  loss: number;
}

export class MLP {
  readonly sizes: number[];
  readonly acts: Activation[];
  W: number[][][]; // W[layer][outIdx][inIdx]
  b: number[][]; // b[layer][outIdx]

  constructor(sizes: number[], acts: Activation[]) {
    this.sizes = sizes;
    this.acts = acts;
    this.W = [];
    this.b = [];
    for (let l = 0; l < sizes.length - 1; l++) {
      const inSize = sizes[l];
      const outSize = sizes[l + 1];
      const lim = Math.sqrt(2 / inSize);
      const layer: number[][] = [];
      for (let o = 0; o < outSize; o++) {
        const row: number[] = [];
        for (let i = 0; i < inSize; i++) row.push((Math.random() * 2 - 1) * lim);
        layer.push(row);
      }
      this.W.push(layer);
      this.b.push(new Array(outSize).fill(0));
    }
  }

  forward(x: number[]): { as: number[][] } {
    const as: number[][] = [x];
    let a = x;
    for (let l = 0; l < this.W.length; l++) {
      const layer = this.W[l];
      const bl = this.b[l];
      const z = layer.map((row, o) => {
        let s = bl[o];
        for (let i = 0; i < row.length; i++) s += row[i] * a[i];
        return s;
      });
      a = applyAct(this.acts[l], z);
      as.push(a);
    }
    return { as };
  }

  predict(x: number[]): number[] {
    const { as } = this.forward(x);
    return as[as.length - 1];
  }

  // Forward + backward pass for a single sample, without applying any weight
  // update. Used both internally by trainStep (accumulated over a batch) and
  // externally by probe() for visualization.
  private backwardSingle(x: number[], y: number[], lossType: LossType): BackwardResult {
    const { as } = this.forward(x);
    const L = this.W.length;
    const yhat = as[L];

    // In all three cases the output-layer gradient dLoss/dz simplifies to
    // (yhat - y): sigmoid+BCE, softmax+categorical-CE, and linear+MSE.
    let delta: number[];
    let loss: number;
    if (lossType === "mse") {
      delta = yhat.map((v, i) => (2 * (v - y[i])) / yhat.length);
      loss = yhat.reduce((s, v, i) => s + (v - y[i]) ** 2, 0) / yhat.length;
    } else if (lossType === "bce") {
      delta = yhat.map((v, i) => v - y[i]);
      const v = yhat[0];
      const t = y[0];
      loss = -(t * Math.log(v + 1e-9) + (1 - t) * Math.log(1 - v + 1e-9));
    } else {
      delta = yhat.map((v, i) => v - y[i]);
      const idx = y.findIndex((v) => v === 1);
      loss = -Math.log(yhat[idx] + 1e-9);
    }

    const deltas: number[][] = new Array(L);
    let dNext = delta;
    deltas[L - 1] = dNext;
    for (let l = L - 1; l > 0; l--) {
      const layer = this.W[l];
      const prevSize = layer[0].length;
      const dPrevRaw = new Array(prevSize).fill(0);
      for (let o = 0; o < layer.length; o++) {
        const row = layer[o];
        for (let i = 0; i < row.length; i++) dPrevRaw[i] += row[i] * dNext[o];
      }
      const derivs = actDerivFromOutput(this.acts[l - 1], as[l]);
      dNext = dPrevRaw.map((v, i) => v * derivs[i]);
      deltas[l - 1] = dNext;
    }
    return { as, deltas, loss };
  }

  // One SGD step over a mini-batch; returns the average loss.
  trainStep(batch: Sample[], lr: number, lossType: LossType): number {
    const L = this.W.length;
    const gW = this.W.map((layer) => layer.map((row) => row.map(() => 0)));
    const gb = this.b.map((bl) => bl.map(() => 0));
    let totalLoss = 0;

    for (const { x, y } of batch) {
      const { as, deltas, loss } = this.backwardSingle(x, y, lossType);
      totalLoss += loss;
      for (let l = 0; l < L; l++) {
        const dNext = deltas[l];
        const aPrev = as[l];
        const layer = this.W[l];
        for (let o = 0; o < layer.length; o++) {
          gb[l][o] += dNext[o];
          const row = layer[o];
          for (let i = 0; i < row.length; i++) gW[l][o][i] += dNext[o] * aPrev[i];
        }
      }
    }

    const n = batch.length;
    for (let l = 0; l < L; l++) {
      for (let o = 0; o < this.W[l].length; o++) {
        this.b[l][o] -= (lr * gb[l][o]) / n;
        const row = this.W[l][o];
        for (let i = 0; i < row.length; i++) row[i] -= (lr * gW[l][o][i]) / n;
      }
    }
    return totalLoss / n;
  }

  // Forward + backward pass on a single sample for visualization only — no
  // weight update. Returns per-layer activations (for the forward-pass view)
  // and per-layer dLoss/dz (for the backward-pass view).
  probe(sample: Sample, lossType: LossType): { as: number[][]; deltas: number[][] } {
    const { as, deltas } = this.backwardSingle(sample.x, sample.y, lossType);
    return { as, deltas };
  }
}
