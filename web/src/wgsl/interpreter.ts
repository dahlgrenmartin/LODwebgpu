import { Runtime, type Tensor, type ElementwiseOpName } from './runtime';
import { bindFromShape, evalExpr, evalShape, type Bindings, type Expr } from './expr';
import {
  broadcastShape, contiguousStrides, normDim, numel, resolveView,
} from './shapes';

// --- exported graph format --------------------------------------------------

export interface GraphNode {
  name: string;
  op: string;
  args: any[];
  kwargs?: Record<string, any>;
  shape?: Expr[];
  /** Multi-output ops carry one shape per tuple element. */
  shapes?: (Expr[] | null)[];
}
export interface GraphDoc {
  symbols: string[];
  /** Storage precision of the weights blob; arithmetic is always fp32. */
  weightsDtype?: 'float16' | 'float32';
  inputs: { name: 'z' | 'target'; node: string; shape: Expr[] }[];
  constants: { node: string; offset: number; numel: number; shape: Expr[] }[];
  weights: string;
  nodes: GraphNode[];
  outputs: { name: string; label: string }[];
}

type Value = Tensor | number | Value[];

/**
 * Widen an fp16 weights blob to fp32.
 *
 * Via a 65536-entry table: there are only that many distinct half values, and a
 * per-element bit decode over tens of millions of weights is noticeably slow.
 */
let halfTable: Float32Array | null = null;

function widenFloat16(src: Uint16Array): Float32Array {
  if (!halfTable) {
    halfTable = new Float32Array(65536);
    for (let h = 0; h < 65536; h++) {
      const sign = h & 0x8000 ? -1 : 1;
      const exp = (h & 0x7c00) >> 10;
      const frac = h & 0x03ff;
      halfTable[h] = exp === 0
        ? sign * 6.103515625e-5 * (frac / 1024)
        : exp === 0x1f
          ? (frac ? NaN : sign * Infinity)
          : sign * Math.pow(2, exp - 15) * (1 + frac / 1024);
    }
  }
  const out = new Float32Array(src.length);
  for (let i = 0; i < src.length; i++) out[i] = halfTable[src[i]];
  return out;
}

const isTensor = (v: Value): v is Tensor =>
  typeof v === 'object' && v !== null && !Array.isArray(v) && 'buffer' in v;

/** aten op -> elementwise opcode, for the ops that are a pure elementwise map. */
const BINARY: Record<string, ElementwiseOpName> = {
  'aten::add.Tensor': 'add', 'aten::sub.Tensor': 'sub',
  'aten::mul.Tensor': 'mul', 'aten::div.Tensor': 'div',
};
const SCALAR: Record<string, ElementwiseOpName> = {
  'aten::add.Scalar': 'add', 'aten::sub.Scalar': 'sub',
  'aten::mul.Scalar': 'mul', 'aten::div.Scalar': 'div',
  'aten::gt.Scalar': 'gt', 'aten::lt.Scalar': 'lt',
};
const UNARY: Record<string, ElementwiseOpName> = {
  'aten::abs': 'abs', 'aten::neg': 'neg',
  'aten::sigmoid': 'sigmoid', 'aten::silu': 'silu',
};
/** Ops that alias their input's buffer: shape changes, data does not move. */
const ALIASING = new Set(['aten::view_copy', 'aten::view', 'aten::unsqueeze',
                          'aten::_unsafe_view', 'aten::reshape']);

/**
 * Buffer pool with liveness-based recycling.
 *
 * The graph has ~1300 intermediates; at 512x512 a single 96-channel activation
 * is 100 MB, so allocating them all would need multiple gigabytes. Lifetime is
 * tracked per *buffer* rather than per value, because aliasing ops share one
 * buffer between several values and freeing on the first of them would corrupt
 * the rest.
 */
class Pool {
  private free = new Map<number, GPUBuffer[]>();
  peak = 0;
  private live = 0;

  constructor(private rt: Runtime) {}

  /**
   * Round an allocation up so that similar shapes share a bucket.
   *
   * The free list is keyed by exact byte size. The graph's ~1300 intermediates
   * have many near-but-not-equal sizes - a 96-channel activation and a
   * 128-channel one, a tensor and its padded twin - and exact-size keying makes
   * every one of them miss, so the pool allocates a fresh buffer instead of
   * recycling. Measured at 1024x768 that cost 19.4 GB of device memory for a
   * graph whose genuinely-live peak is a fraction of it.
   *
   * Rounding to eight steps per octave caps the waste at 12.5% while collapsing
   * the size space enough that reuse actually hits.
   */
  private static bucket(size: number): number {
    if (size <= 4096) return 4096;
    const octave = 2 ** Math.floor(Math.log2(size));
    const step = octave / 8;
    return Math.ceil(size / step) * step;
  }

  /**
   * Buffers currently sitting in the free list.
   *
   * Aliasing ops make several values share one buffer, so the same buffer can be
   * offered for release twice in a run; returning it twice would put it in the
   * free list twice and hand one piece of storage to two live values. Tracking
   * "has ever been released" instead is not equivalent - a buffer legitimately
   * cycles between free and live many times per run, and suppressing the later
   * releases collapses reuse entirely (measured: 19 GB -> 64 GB and a hung
   * device).
   */
  private inFree = new Set<GPUBuffer>();

  acquire(count: number, label?: string): GPUBuffer {
    const size = Pool.bucket(Math.max(4, count * 4));

    // Exact bucket first, then any larger free buffer up to twice the request.
    // Buckets alone still stranded memory: a run's free list ends up holding
    // several sizes that no later node asks for by name, while the node that
    // does ask wants a size one step up and allocates afresh. A kernel only ever
    // addresses the elements its shape covers, so oversized storage is safe.
    // Exact bucket only. Scanning the free list for the smallest buffer that
    // merely fits was tried and reverted: it recovered 327 MB of the 5.6 GB at
    // 512x512 and cost 29% of the step time, because the scan runs for all 1282
    // nodes while the allocation it avoids happens only on the first step.
    const reused = this.free.get(size)?.pop();
    // Charge the buffer's real size, not the request: best-fit can hand back
    // storage larger than asked for, and releasing it credits buffer.size.
    if (reused) {
      this.inFree.delete(reused);
      this.live += reused.size;
      this.peak = Math.max(this.peak, this.live);
      return reused;
    }
    const fresh = this.rt.alloc(size / 4, label);
    this.live += fresh.size;
    this.peak = Math.max(this.peak, this.live);
    return fresh;
  }

  release(buffer: GPUBuffer): void {
    if (this.inFree.has(buffer)) return;
    this.inFree.add(buffer);
    const size = buffer.size;
    const bucket = this.free.get(size) ?? [];
    bucket.push(buffer);
    this.free.set(size, bucket);
    this.live -= size;
  }

  /** Reset the live count between runs; buffers stay in the free list. */
  endRun(): void { this.live = 0; }

  /** Current live bytes, for the residency profile. */
  get liveBytes(): number { return this.live; }
}

export class Interpreter {
  private values = new Map<string, Value>();
  private pool: Pool;
  private bindings: Bindings = {};

  private constructor(
    private rt: Runtime,
    private doc: GraphDoc,
    private constants: Map<string, GPUBuffer>,
  ) {
    this.pool = new Pool(rt);
  }

  static async load(rt: Runtime, jsonUrl: string, baseUrl: string): Promise<Interpreter> {
    const doc: GraphDoc = await fetch(jsonUrl).then((r) => r.json());
    const buf = await fetch(`${baseUrl}/${doc.weights}`).then((r) => r.arrayBuffer());
    const data = doc.weightsDtype === 'float16'
      ? widenFloat16(new Uint16Array(buf))
      : new Float32Array(buf);

    // One buffer per constant rather than offsets into a single blob. Only the
    // gather kernel honours a source offset; conv, matmul, groupnorm and
    // elementwise all read from element zero, so a shared blob would silently
    // feed every one of them the wrong weights.
    const constants = new Map<string, GPUBuffer>();
    for (const c of doc.constants) {
      const slice = data.subarray(c.offset, c.offset + c.numel);
      const b = rt.device.createBuffer({
        size: Math.max(4, slice.byteLength),
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
        label: c.node,
      });
      rt.device.queue.writeBuffer(b, 0, slice);
      constants.set(c.node, b);
    }
    return new Interpreter(rt, doc, constants);
  }

  /** Symbols solved from the actual input; this is what makes any size work. */
  bind(zShape: number[], targetShape: number[]): Bindings {
    const b: Bindings = {};
    for (const inp of this.doc.inputs) {
      bindFromShape(inp.shape, inp.name === 'z' ? zShape : targetShape, b);
    }
    for (const s of this.doc.symbols) {
      if (b[s] === undefined) throw new Error(`symbol ${s} not determined by the inputs`);
    }
    this.bindings = b;
    return b;
  }

  private shapeOf(e: Expr[] | null | undefined): number[] {
    if (!e) throw new Error('node is missing a shape');
    return evalShape(e, this.bindings);
  }


  /**
   * The output shape of a node.
   *
   * Most nodes carry one from the export, but the rewrite creates nodes after
   * the trace and those have no metadata - and FakeTensorProp cannot re-derive
   * it under symbolic shapes. So the shape is inferred from the operands, the
   * way any interpreter over a shape-agnostic graph must.
   */
  private outShape(node: GraphNode): number[] {
    const declared = node.shape ?? node.shapes?.[0];
    if (declared) return evalShape(declared, this.bindings);

    const op = node.op;
    const a = node.args;
    const T = (i: number) => this.tensor(a[i]);

    if (BINARY[op]) {
      const l = this.arg(a[0]);
      const r = this.arg(a[1]);
      if (isTensor(l as Value) && isTensor(r as Value)) {
        return broadcastShape((l as Tensor).shape, (r as Tensor).shape);
      }
      return isTensor(l as Value) ? (l as Tensor).shape : (r as Tensor).shape;
    }
    if (SCALAR[op] || UNARY[op]) return T(0).shape;

    switch (op) {
      case 'aten::clone':
      case 'aten::_to_copy':
      case 'aten::contiguous':
      case 'aten::fill.Scalar':
      case 'aten::ones_like':
      case 'aten::zeros_like':
      case 'aten::empty_like':
        return T(0).shape;
      case 'aten::permute': {
        const x = T(0);
        return (this.arg(a[1]) as number[])
          .map((p) => x.shape[normDim(Number(p), x.shape.length)]);
      }
      case 'aten::expand_copy':
      case 'aten::expand':
      case 'aten::new_zeros':
      case 'aten::empty.memory_format':
        return (this.arg(a[1]) as number[]).map(Number);
      case 'aten::_softmax':
        return T(0).shape;
      case 'aten::mean.dim':
      case 'aten::sum.dim_IntList': {
        const x = T(0);
        const shape = x.shape.slice();
        for (const d of this.arg(a[1]) as number[]) {
          shape[normDim(Number(d), x.shape.length)] = 1;
        }
        return shape;
      }
      case 'aten::mean':
      case 'aten::sum':
        return T(0).shape.map(() => 1);
      case 'aten::convolution': {
        const x = T(0);
        const w = T(1);
        const pair = (v: any): number[] => {
          const arr = (this.arg(v) as number[]).map(Number);
          return arr.length === 1 ? [arr[0], arr[0]] : arr;
        };
        const st = pair(a[3]);
        const pad = pair(a[4]);
        const dil = pair(a[5]);
        const transposed = a[6] === true;
        const opad = pair(a[7]);
        const groups = Number(this.arg(a[8]));
        const [, , KH, KW] = w.shape;
        const spatial = [x.shape[2], x.shape[3]].map((inSize, i) => {
          const k = i === 0 ? KH : KW;
          return transposed
            ? (inSize - 1) * st[i] - 2 * pad[i] + dil[i] * (k - 1) + 1 + opad[i]
            : Math.floor((inSize + 2 * pad[i] - dil[i] * (k - 1) - 1) / st[i]) + 1;
        });
        const cout = transposed ? w.shape[1] * groups : w.shape[0];
        return [x.shape[0], cout, spatial[0], spatial[1]];
      }
      default:
        throw new Error(`cannot infer an output shape for ${op}`);
    }
  }

  private arg(a: any): Value | number | null {
    if (a === null || typeof a === 'boolean' || typeof a === 'string') return a as any;
    if (typeof a === 'number') return a;
    if (Array.isArray(a)) return a.map((x) => this.arg(x)) as any;
    if (typeof a === 'object') {
      if ('ref' in a) {
        const v = this.values.get(a.ref);
        if (v === undefined) throw new Error(`value ${a.ref} not produced yet`);
        return v;
      }
      if ('dtype' in a || 'device' in a || 'layout' in a || 'memory_format' in a) return null;
      return evalExpr(a as Expr, this.bindings);
    }
    return a;
  }

  private tensor(a: any): Tensor {
    const v = this.arg(a);
    if (!isTensor(v as Value)) throw new Error('expected a tensor argument');
    return v as Tensor;
  }

  private lastUse(): Map<GPUBuffer, number> {
    // Filled in during run(), once buffers exist.
    return new Map();
  }

  /**
   * Persistent destination for each graph output.
   *
   * Outputs are copied out of the pool so a later node cannot overwrite them,
   * but allocating that copy per run leaked one buffer per output per step -
   * 6 MB a step at 512x512, none of it ever freed. The shape is fixed for a
   * given input size, so one buffer per label is reused for every step and
   * replaced only if the size changes.
   */
  private outputBuffers = new Map<string, GPUBuffer>();

  private outputBuffer(label: string, count: number): GPUBuffer {
    const want = Math.max(4, count * 4);
    const hit = this.outputBuffers.get(label);
    if (hit && hit.size === want) return hit;
    if (hit) this.rt.free(hit);
    const b = this.rt.alloc(count, `out:${label}`);
    this.outputBuffers.set(label, b);
    return b;
  }

  /** Peak bytes the intermediate pool has had live, for leak diagnosis. */
  get poolPeakBytes(): number { return this.pool.peak; }

  /** When set, live pool bytes are recorded after each node (debug only). */
  profile: number[] | null = null;

  /** When set, every node's output statistics are recorded (slow; debug only). */
  trace: { name: string; op: string; shape: number[]; mean: number; absmax: number }[] | null = null;

  async run(
    z: { buffer: GPUBuffer; shape: number[] },
    target: { buffer: GPUBuffer; shape: number[] },
  ): Promise<Record<string, Tensor>> {
    this.bind(z.shape, target.shape);
    this.values.clear();

    for (const inp of this.doc.inputs) {
      this.values.set(inp.node, inp.name === 'z' ? z : target);
    }
    const protectedBuffers = new Set<GPUBuffer>([z.buffer, target.buffer]);
    for (const c of this.doc.constants) {
      const buffer = this.constants.get(c.node)!;
      this.values.set(c.node, { buffer, shape: this.shapeOf(c.shape) });
      protectedBuffers.add(buffer);
    }

    // Last consumer of each buffer, so intermediates can be recycled.
    const lastUse = new Map<string, number>();
    this.doc.nodes.forEach((n, i) => {
      const walk = (a: any): void => {
        if (Array.isArray(a)) { a.forEach(walk); return; }
        if (a && typeof a === 'object' && 'ref' in a) lastUse.set(a.ref, i);
      };
      n.args.forEach(walk);
      Object.values(n.kwargs ?? {}).forEach(walk);
    });
    for (const o of this.doc.outputs) lastUse.set(o.name, Number.MAX_SAFE_INTEGER);
    const outputNames = new Map(this.doc.outputs.map((o) => [o.name, o.label]));
    const snapshots = new Map<string, Tensor>();

    const release = (b: GPUBuffer): void => {
      if (protectedBuffers.has(b)) return;
      this.pool.release(b);
    };

    for (let i = 0; i < this.doc.nodes.length; i++) {
      const node = this.doc.nodes[i];
      try {
        this.values.set(node.name, this.exec(node));
      } catch (e) {
        // Name the node: a bare "expected a tensor" from 1284 nodes is useless.
        const argKinds = node.args.map((a) => {
          if (a && typeof a === 'object' && 'ref' in a) {
            const v = this.values.get(a.ref);
            return `${a.ref}:${Array.isArray(v) ? 'tuple' : isTensor(v as Value) ? 'tensor' : typeof v}`;
          }
          return JSON.stringify(a)?.slice(0, 24);
        });
        throw new Error(
          `node ${i} (${node.name}, ${node.op}) failed: ${(e as Error).message}` +
          ` | args = [${argKinds.join(', ')}]`);
      }

      if (this.trace) {
        const v = this.values.get(node.name);
        const t = Array.isArray(v) ? (v[0] as Tensor) : (v as Value);
        if (isTensor(t) && numel(t.shape) <= 2_000_000) {
          const d = await this.rt.read(t.buffer, numel(t.shape));
          let sum = 0;
          let amax = 0;
          for (let k = 0; k < d.length; k++) { sum += d[k]; amax = Math.max(amax, Math.abs(d[k])); }
          this.trace.push({ name: node.name, op: node.op, shape: t.shape,
                            mean: sum / Math.max(1, d.length), absmax: amax });
        }
      }

      // Copy outputs into dedicated buffers the moment they are produced. The
      // pool would otherwise be free to hand their storage to a later node,
      // since an output's own buffer can be aliased by values that do die.
      const label = outputNames.get(node.name);
      if (label) {
        const v = this.values.get(node.name);
        if (v && isTensor(v)) {
          const keep = this.outputBuffer(label, numel(v.shape));
          this.rt.gather(v, contiguousStrides(v.shape), v.shape, keep);
          snapshots.set(label, { buffer: keep, shape: v.shape });
        }
      }

      if (this.profile) this.profile.push(this.pool.liveBytes);

      // Recycle any buffer whose last consumer was this node.
      for (const [name, idx] of lastUse) {
        if (idx !== i) continue;
        const v = this.values.get(name);
        if (!v || !isTensor(v)) continue;
        if (protectedBuffers.has(v.buffer)) continue;
        const stillNeeded = [...lastUse].some(
          ([n2, i2]) => i2 > i && isTensor(this.values.get(n2) as Value) &&
            (this.values.get(n2) as Tensor).buffer === v.buffer);
        if (!stillNeeded) release(v.buffer);
      }
    }

    // The outputs are pinned above so nothing can overwrite them mid-run, which
    // also means the loop never releases them: four buffers a step that the pool
    // then had to replace with fresh device allocations. Their values are safe in
    // the snapshots by now, so hand the storage back.
    for (const v of this.values.values()) {
      if (isTensor(v as Value)) release((v as Tensor).buffer);
    }

    this.pool.endRun();

    const out: Record<string, Tensor> = {};
    for (const o of this.doc.outputs) {
      const snap = snapshots.get(o.label);
      if (!snap) throw new Error(`output ${o.label} was never produced`);
      out[o.label] = snap;
    }
    return out;
  }

  private exec(node: GraphNode): Value {
    const { op } = node;
    const a = node.args;

    // --- values that never touch the GPU ---
    if (op === 'getitem') {
      const src = this.arg(a[0]) as Value[];
      return src[a[1] as number];
    }
    if (op === 'aten::sym_size.int') {
      const t = this.tensor(a[0]);
      return t.shape[normDim(a[1] as number, t.shape.length)];
    }
    if (op === '<built-in function mul>') {
      return (this.arg(a[0]) as number) * (this.arg(a[1]) as number);
    }
    if (op === '<built-in function add>') {
      return (this.arg(a[0]) as number) + (this.arg(a[1]) as number);
    }

    // --- pure metadata: same buffer, different shape ---
    if (ALIASING.has(op)) {
      const t = this.tensor(a[0]);
      if (op === 'aten::unsqueeze') {
        const d = normDim(a[1] as number, t.shape.length + 1);
        const shape = t.shape.slice();
        shape.splice(d, 0, 1);
        return { ...t, shape };
      }
      const target = (this.arg(a[1]) as number[]).map(Number);
      return { ...t, shape: resolveView(target, numel(t.shape)) };
    }

    const shape = this.outShape(node);
    const out = this.pool.acquire(numel(shape), node.name);
    const result: Tensor = { buffer: out, shape };

    if (BINARY[op]) {
      // The .Tensor overloads accept a plain number on either side, so decide
      // per operand rather than assuming two tensors.
      const lhs = this.arg(a[0]);
      const rhs = this.arg(a[1]);
      const lhsT = isTensor(lhs as Value);
      const rhsT = isTensor(rhs as Value);
      if (lhsT && rhsT) {
        const x = lhs as Tensor;
        const y = rhs as Tensor;
        this.rt.elementwise(BINARY[op], x, y, 0, broadcastShape(x.shape, y.shape), out);
      } else if (lhsT) {
        this.rt.elementwise(BINARY[op], lhs as Tensor, null, Number(rhs), shape, out);
      } else if (rhsT) {
        // scalar OP tensor: commute, or use the reversed form where it matters.
        const y = rhs as Tensor;
        const k = Number(lhs);
        if (op === 'aten::add.Tensor' || op === 'aten::mul.Tensor') {
          this.rt.elementwise(BINARY[op], y, null, k, shape, out);
        } else if (op === 'aten::sub.Tensor') {
          this.rt.elementwise('rsub', y, null, k, shape, out);
        } else {
          throw new Error(`scalar / tensor form of ${op} is not supported`);
        }
      } else {
        throw new Error(`${op} with two non-tensor operands`);
      }
      return result;
    }
    if (SCALAR[op]) {
      const x = this.tensor(a[0]);
      this.rt.elementwise(SCALAR[op], x, null, Number(this.arg(a[1])), x.shape, out);
      return result;
    }
    if (UNARY[op]) {
      const x = this.tensor(a[0]);
      this.rt.elementwise(UNARY[op], x, null, 0, x.shape, out);
      return result;
    }

    switch (op) {
      case 'aten::clone':
      case 'aten::_to_copy':
      case 'aten::contiguous': {
        const x = this.tensor(a[0]);
        this.rt.gather(x, contiguousStrides(x.shape), shape, out);
        return result;
      }
      case 'aten::permute': {
        const x = this.tensor(a[0]);
        const perm = (this.arg(a[1]) as number[]).map((p) => normDim(p, x.shape.length));
        const st = contiguousStrides(x.shape);
        this.rt.gather(x, perm.map((p) => st[p]), perm.map((p) => x.shape[p]), out);
        return result;
      }
      case 'aten::expand_copy':
      case 'aten::expand': {
        const x = this.tensor(a[0]);
        const st = contiguousStrides(x.shape);
        const offset = shape.length - x.shape.length;
        const strides = shape.map((_, i) => {
          const si = i - offset;
          if (si < 0) return 0;
          return x.shape[si] === shape[i] ? st[si] : 0;
        });
        this.rt.gather(x, strides, shape, out);
        return result;
      }
      case 'aten::slice.Tensor': {
        const x = this.tensor(a[0]);
        const d = normDim(Number(this.arg(a[1]) ?? 0), x.shape.length);
        const st = contiguousStrides(x.shape);
        const start = Math.max(0, Number(this.arg(a[2]) ?? 0));
        this.rt.gather({ ...x, offset: (x.offset ?? 0) + start * st[d] }, st, shape, out);
        return result;
      }
      case 'aten::cat': {
        const parts = (this.arg(a[0]) as Tensor[]);
        const d = normDim(Number(this.arg(a[1]) ?? 0), shape.length);
        if (d !== 0 && parts.some((p) => numel(p.shape) === 0)) {
          throw new Error('cat with empty parts is not supported');
        }
        let written = 0;
        const outer = shape.slice(0, d).reduce((x, y) => x * y, 1);
        const innerOut = shape.slice(d).reduce((x, y) => x * y, 1);
        for (const p of parts) {
          const innerP = p.shape.slice(d).reduce((x, y) => x * y, 1);
          // Copy each outer slab into its stripe of the output.
          for (let o = 0; o < outer; o++) {
            this.rt.gather(
              { ...p, offset: (p.offset ?? 0) + o * innerP, shape: [innerP] },
              [1], [innerP], out, o * innerOut + written);
          }
          written += innerP;
        }
        return result;
      }
      case 'aten::convolution': {
        const x = this.tensor(a[0]);
        const w = this.tensor(a[1]);
        const bias = a[2] === null ? null : this.tensor(a[2]);
        const pair = (v: any): number[] => {
          const arr = (this.arg(v) as number[]).map(Number);
          return arr.length === 1 ? [arr[0], arr[0]] : arr;
        };
        this.rt.conv2d(x, w, bias, {
          stride: pair(a[3]), padding: pair(a[4]), dilation: pair(a[5]),
          transposed: a[6] === true, groups: Number(this.arg(a[8])),
          outputPadding: pair(a[7]),
          outShape: shape,
        }, out);
        return result;
      }
      case 'aten::native_group_norm': {
        const x = this.tensor(a[0]);
        const w = a[1] === null ? null : this.tensor(a[1]);
        const bias = a[2] === null ? null : this.tensor(a[2]);
        const N = Number(this.arg(a[3]));
        const C = Number(this.arg(a[4]));
        const HxW = Number(this.arg(a[5]));
        const G = Number(this.arg(a[6]));
        const eps = Number(this.arg(a[7]));
        const meanBuf = this.pool.acquire(N * G);
        const rstdBuf = this.pool.acquire(N * G);
        this.rt.groupNorm(x, w, bias, N, C, HxW, G, eps, out, meanBuf, rstdBuf);
        return [result,
                { buffer: meanBuf, shape: [N, G] },
                { buffer: rstdBuf, shape: [N, G] }];
      }
      case 'aten::mm':
      case 'aten::bmm':
      case 'aten::addmm':
      case 'aten::baddbmm': {
        const hasBias = op === 'aten::addmm' || op === 'aten::baddbmm';
        const x = this.tensor(a[hasBias ? 1 : 0]);
        const y = this.tensor(a[hasBias ? 2 : 1]);
        const bias = hasBias ? this.tensor(a[0]) : null;
        const batched = x.shape.length === 3;
        const B = batched ? x.shape[0] : 1;
        const M = x.shape[batched ? 1 : 0];
        const K = x.shape[batched ? 2 : 1];
        const N = y.shape[y.shape.length - 1];
        this.rt.matmul(x, y, bias, {
          B, M, K, N,
          biasIsRowVector: !!bias && bias.shape.length === 1,
          biasIsBatched: !!bias && bias.shape.length === 3,
          beta: hasBias ? Number(this.arg(node.kwargs?.beta) ?? 1) : 1,
          alpha: hasBias ? Number(this.arg(node.kwargs?.alpha) ?? 1) : 1,
        }, out);
        return result;
      }
      case 'aten::_softmax': {
        const x = this.tensor(a[0]);
        this.rt.softmax(x, Number(this.arg(a[1])), out);
        return result;
      }
      case 'aten::mean.dim':
      case 'aten::sum.dim_IntList': {
        const x = this.tensor(a[0]);
        const dims = (this.arg(a[1]) as number[]).map((d) => normDim(Number(d), x.shape.length));
        this.rt.reduce(x, dims, op === 'aten::mean.dim', out);
        return result;
      }
      case 'aten::mean':
      case 'aten::sum': {
        const x = this.tensor(a[0]);
        const all = x.shape.map((_, i) => i);
        this.rt.reduce(x, all, op === 'aten::mean', out);
        return result;
      }
      case 'aten::fill.Scalar': {
        this.rt.fill(Number(this.arg(a[1])), shape, out);
        return result;
      }
      case 'aten::ones_like': {
        this.rt.fill(1, shape, out);
        return result;
      }
      case 'aten::zeros_like':
      case 'aten::new_zeros':
      case 'aten::empty_like':
      case 'aten::empty.memory_format': {
        // Buffers come back from the pool dirty, so zero explicitly. This must
        // not bind `out` as an input as well: that is a synchronisation-scope
        // violation and the dispatch would be dropped, leaving the stale data.
        this.rt.fill(0, shape, out);
        return result;
      }
      default:
        throw new Error(`no kernel for op ${op}`);
    }
  }
}
