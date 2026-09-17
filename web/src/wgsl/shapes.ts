/** Shape and stride helpers for contiguous NCHW-style tensors. */

export function numel(shape: number[]): number {
  return shape.reduce((a, b) => a * b, 1);
}

/** Row-major (contiguous) strides. */
export function contiguousStrides(shape: number[]): number[] {
  const s = new Array(shape.length);
  let acc = 1;
  for (let i = shape.length - 1; i >= 0; i--) {
    s[i] = acc;
    acc *= shape[i];
  }
  return s;
}

/**
 * Strides that read `src` as if it had been broadcast to `dst`.
 *
 * A broadcast dimension gets stride 0, which is what lets one generic gather
 * kernel serve expand, and elementwise ops with mismatched ranks.
 */
export function broadcastStrides(srcShape: number[], dstShape: number[]): number[] {
  const out = new Array(dstShape.length).fill(0);
  const srcStrides = contiguousStrides(srcShape);
  const offset = dstShape.length - srcShape.length;
  if (offset < 0) {
    throw new Error(`cannot broadcast rank ${srcShape.length} to ${dstShape.length}`);
  }
  for (let i = 0; i < srcShape.length; i++) {
    const s = srcShape[i];
    const d = dstShape[i + offset];
    if (s === d) out[i + offset] = srcStrides[i];
    else if (s === 1) out[i + offset] = 0;
    else throw new Error(`cannot broadcast dim ${i}: ${s} vs ${d}`);
  }
  return out;
}

/** The shape resulting from broadcasting two shapes together. */
export function broadcastShape(a: number[], b: number[]): number[] {
  const rank = Math.max(a.length, b.length);
  const out = new Array(rank);
  for (let i = 0; i < rank; i++) {
    const x = a[a.length - rank + i] ?? 1;
    const y = b[b.length - rank + i] ?? 1;
    if (x !== y && x !== 1 && y !== 1) {
      throw new Error(`shapes not broadcastable: [${a}] vs [${b}]`);
    }
    out[i] = Math.max(x, y);
  }
  return out;
}

/** Normalize a possibly-negative dim index. */
export function normDim(d: number, rank: number): number {
  const n = d < 0 ? d + rank : d;
  if (n < 0 || n >= rank) throw new Error(`dim ${d} out of range for rank ${rank}`);
  return n;
}

/** Resolve a view target that may contain a single -1. */
export function resolveView(target: number[], total: number): number[] {
  const out = target.slice();
  const idx = out.indexOf(-1);
  if (idx === -1) {
    if (numel(out) !== total) {
      throw new Error(`view [${target}] does not match ${total} elements`);
    }
    return out;
  }
  const known = out.reduce((a, b, i) => (i === idx ? a : a * b), 1);
  if (known === 0 || total % known !== 0) {
    throw new Error(`view [${target}] cannot infer -1 from ${total} elements`);
  }
  out[idx] = total / known;
  return out;
}

/** Pad shape/stride arrays out to a fixed rank for the shader ABI. */
export function padTo(arr: number[], rank: number, fill = 1): number[] {
  if (arr.length > rank) throw new Error(`rank ${arr.length} exceeds shader max ${rank}`);
  return [...new Array(rank - arr.length).fill(fill), ...arr];
}

/** Same as padTo but pads with 0, for strides. */
export function padStrides(arr: number[], rank: number): number[] {
  return padTo(arr, rank, 0);
}
