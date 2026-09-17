import { Runtime, type Tensor, type ElementwiseOpName } from './runtime';
import { broadcastShape, contiguousStrides, numel } from './shapes';

interface TensorJson { shape: number[]; data: number[] }
interface Case {
  kernel: 'gather' | 'elementwise' | 'reduce';
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
  mean?: boolean;
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
    } else {
      const src = upload(rt, c.src!);
      rt.reduce(src, c.dims!, !!c.mean, out);
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
