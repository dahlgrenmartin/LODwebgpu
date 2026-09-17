import { Runtime, type Tensor, type ElementwiseOpName } from './runtime';
import { broadcastShape, contiguousStrides, numel } from './shapes';

interface TensorJson { shape: number[]; data: number[] }
interface Case {
  kernel: 'gather' | 'elementwise' | 'reduce' | 'conv2d' | 'groupnorm'
        | 'matmul' | 'softmax';
  name: string;
  expected: TensorJson;
  src?: TensorJson;
  a?: TensorJson;
  b?: TensorJson | null;
  op?: ElementwiseOpName;
  scalar?: number;
  perm?: number[];
  expand?: number[];
  slice?: { dim: number; start: number; end: number };
  dims?: number[];
  mean?: boolean | TensorJson;
  weight?: TensorJson | null;
  bias?: TensorJson | null;
  stride?: number[];
  padding?: number[];
  dilation?: number[];
  groups?: number;
  transposed?: boolean;
  N?: number; C?: number; HxW?: number; G?: number; eps?: number;
  B?: number; M?: number; K?: number;
  biasIsRowVector?: boolean;
  dim?: number;
  rstd?: TensorJson;
}

interface Result { name: string; kernel: string; maxAbs: number; pass: boolean; error?: string }

const TOL = 1e-5;

function maxAbs(a: Float32Array, b: number[]): number {
  if (a.length !== b.length) return Infinity;
  let m = 0;
  for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i]));
  return m;
}

function upload(rt: Runtime, t: TensorJson): Tensor {
  const buf = rt.alloc(t.data.length);
  rt.device.queue.writeBuffer(buf, 0, new Float32Array(t.data));
  return { buffer: buf, shape: t.shape };
}

async function runCase(rt: Runtime, c: Case): Promise<Result> {
  const base = { name: c.name, kernel: c.kernel };
  try {
    const outCount = c.expected.data.length;
    const out = rt.alloc(outCount);

    if (c.kernel === 'gather') {
      const src = upload(rt, c.src!);
      const srcStrides = contiguousStrides(src.shape);
      if (c.perm) {
        const outShape = c.perm.map((p) => src.shape[p]);
        const strides = c.perm.map((p) => srcStrides[p]);
        rt.gather(src, strides, outShape, out);
      } else if (c.expand) {
        const strides = src.shape.map((s, i) => (s === 1 ? 0 : srcStrides[i]));
        rt.gather(src, strides, c.expand, out);
      } else if (c.slice) {
        const { dim, start, end } = c.slice;
        const outShape = src.shape.slice();
        outShape[dim] = end - start;
        rt.gather({ ...src, offset: start * srcStrides[dim] },
                  srcStrides, outShape, out);
      } else {
        throw new Error('gather case has no transform');
      }
    } else if (c.kernel === 'elementwise') {
      const a = upload(rt, c.a!);
      const b = c.b ? upload(rt, c.b) : null;
      const outShape = b ? broadcastShape(a.shape, b.shape) : a.shape;
      rt.elementwise(c.op!, a, b, c.scalar ?? 0, outShape, out);
    } else if (c.kernel === 'reduce') {
      const src = upload(rt, c.src!);
      rt.reduce(src, c.dims!, c.mean === true, out);
    } else if (c.kernel === 'conv2d') {
      const src = upload(rt, c.src!);
      const w = upload(rt, c.weight!);
      const outShape = c.expected.shape;
      rt.conv2d(src, w, null, {
        stride: c.stride!, padding: c.padding!, dilation: c.dilation!,
        transposed: !!c.transposed, groups: c.groups!, outShape,
      }, out);
    } else if (c.kernel === 'groupnorm') {
      const src = upload(rt, c.src!);
      const w = c.weight ? upload(rt, c.weight) : null;
      const bi = c.bias ? upload(rt, c.bias) : null;
      const ng = c.N! * c.G!;
      const meanBuf = rt.alloc(ng);
      const rstdBuf = rt.alloc(ng);
      rt.groupNorm(src, w, bi, c.N!, c.C!, c.HxW!, c.G!, c.eps!,
                   out, meanBuf, rstdBuf);
      // Statistics are separate graph outputs, so verify them too.
      const gotMean = await rt.read(meanBuf, ng);
      const gotRstd = await rt.read(rstdBuf, ng);
      const statErr = Math.max(maxAbs(gotMean, (c as any).mean.data),
                               maxAbs(gotRstd, c.rstd!.data));
      if (statErr > TOL) {
        return { ...base, maxAbs: statErr, pass: false,
                 error: `mean/rstd mismatch ${statErr.toExponential(2)}` };
      }
    } else if (c.kernel === 'matmul') {
      const a = upload(rt, c.a!);
      const b = upload(rt, c.b!);
      const bi = c.bias ? upload(rt, c.bias) : null;
      rt.matmul(a, b, bi, {
        B: c.B!, M: c.M!, K: c.K!, N: (c as any).N,
        biasIsRowVector: !!c.biasIsRowVector,
      }, out);
    } else {
      const src = upload(rt, c.src!);
      rt.softmax(src, c.dim!, out);
    }

    const got = await rt.read(out, outCount);
    const err = maxAbs(got, c.expected.data);
    return { ...base, maxAbs: err, pass: err <= TOL };
  } catch (e) {
    return { ...base, maxAbs: NaN, pass: false, error: String((e as Error).message || e) };
  }
}

export async function runKernelTests(device: GPUDevice, url: string) {
  const rt = await Runtime.create(device);
  const cases: Case[] = await fetch(url).then((r) => r.json());
  const results: Result[] = [];
  for (const c of cases) results.push(await runCase(rt, c));

  // Negative control: a deliberately wrong reduction must be caught.
  const src = upload(rt, { shape: [2, 4], data: [1, 2, 3, 4, 5, 6, 7, 8] });
  const bad = rt.alloc(2);
  rt.reduce(src, [1], false, bad);                       // sums, expected means
  const got = await rt.read(bad, 2);
  const err = maxAbs(got, [2.5, 6.5]);
  results.push({
    name: 'negative control (sum vs mean)', kernel: 'reduce',
    maxAbs: err, pass: err > TOL,
  });

  const passed = results.filter((r) => r.pass).length;
  return { status: passed === results.length ? 'ALL_PASS' : 'FAIL',
           passed, total: results.length, results };
}
