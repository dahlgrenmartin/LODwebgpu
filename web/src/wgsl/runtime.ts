import gatherSrc from './kernels/gather.wgsl?raw';
import scatterSrc from './kernels/scatter.wgsl?raw';
import elementwiseSrc from './kernels/elementwise.wgsl?raw';
import reduceSrc from './kernels/reduce.wgsl?raw';
import conv2dSrc from './kernels/conv2d.wgsl?raw';
import conv3x3Src from './kernels/conv3x3.wgsl?raw';
import gnStatsSrc from './kernels/groupnorm_stats.wgsl?raw';
import gnApplySrc from './kernels/groupnorm_apply.wgsl?raw';
import matmulSrc from './kernels/matmul.wgsl?raw';
import softmaxSrc from './kernels/softmax.wgsl?raw';
import {
  broadcastStrides, contiguousStrides, isContiguous, numel, padStrides, padTo,
} from './shapes';

export const MAX_RANK = 8;

/**
 * A device with the adapter's maximum buffer limits.
 *
 * The defaults cap a storage binding at 128 MB, which this backend exceeds at
 * realistic resolutions: the binding is then rejected, the dispatch is dropped,
 * and the output silently keeps whatever the pool left there.
 */
export async function createMaxDevice(): Promise<GPUDevice> {
  if (!navigator.gpu) throw new Error('WebGPU unavailable: navigator.gpu undefined');
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error('WebGPU unavailable: requestAdapter() returned null');
  return adapter.requestDevice({
    requiredLimits: {
      maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
      maxBufferSize: adapter.limits.maxBufferSize,
    },
  });
}

export interface Tensor {
  buffer: GPUBuffer;
  shape: number[];
  /** Offset into the buffer, in elements. */
  offset?: number;
}

export const ElementwiseOp = {
  add: 0, sub: 1, mul: 2, div: 3, abs: 4, neg: 5,
  sigmoid: 6, silu: 7, gt: 8, lt: 9, fill: 10, copy: 11, rsub: 12,
} as const;
export type ElementwiseOpName = keyof typeof ElementwiseOp;

/**
 * Compute kernels for the interpreter.
 *
 * Every pipeline is built through `compile`, which asserts on shader
 * compilation info. An invalid pipeline in WebGPU silently drops its
 * dispatches, which presents as a kernel that computed the identity - a failure
 * mode that already cost us once.
 */
export class Runtime {
  /** Matches DISPATCH_STRIDE in the kernels: 65535 * 64. */
  static readonly MAX_GROUPS = 65535;

  /** Grid for kernels that use one workgroup per output element. */
  static wgGrid(n: number): [number, number, number] {
    return [Math.min(Math.max(1, n), Runtime.MAX_GROUPS),
            Math.ceil(Math.max(1, n) / Runtime.MAX_GROUPS), 1];
  }

  private pipelines = new Map<string, GPUComputePipeline>();

  /**
   * Placeholder for optional bindings.
   *
   * Binding a real tensor as a stand-in risks binding the same buffer as both
   * read-only and read-write in one pass, which is a WebGPU synchronisation-
   * scope violation: the encoder is invalidated and the dispatch silently never
   * runs, leaving stale data behind.
   */
  private dummy!: GPUBuffer;

  private constructor(readonly device: GPUDevice) {}

  static async create(device: GPUDevice): Promise<Runtime> {
    const rt = new Runtime(device);
    rt.dummy = device.createBuffer({
      size: 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      label: 'unused-binding',
    });
    await rt.compile('gather', gatherSrc);
    await rt.compile('scatter', scatterSrc);
    await rt.compile('elementwise', elementwiseSrc);
    await rt.compile('reduce', reduceSrc);
    await rt.compile('conv2d', conv2dSrc);
    await rt.compile('conv3x3', conv3x3Src);
    await rt.compile('gnStats', gnStatsSrc);
    await rt.compile('gnApply', gnApplySrc);
    await rt.compile('matmul', matmulSrc);
    await rt.compile('softmax', softmaxSrc);
    return rt;
  }

  private async compile(name: string, code: string): Promise<void> {
    const module = this.device.createShaderModule({ code, label: name });
    const info = await module.getCompilationInfo();
    const errors = info.messages.filter((m) => m.type === 'error');
    if (errors.length) {
      throw new Error(`${name}.wgsl failed to compile: ` +
        errors.map((e) => `${e.lineNum}:${e.linePos} ${e.message}`).join('; '));
    }
    this.pipelines.set(name, this.device.createComputePipeline({
      layout: 'auto', label: name, compute: { module, entryPoint: 'main' },
    }));
  }

  /** Live device allocations made through alloc(), for leak diagnosis. */
  allocCount = 0;
  allocBytes = 0;

  /** When set, every convolution records its configuration (debug only). */
  convLog: { fast: boolean; N: number; Cin: number; Cout: number;
             Hin: number; Win: number; k: string }[] | null = null;

  alloc(count: number, label?: string): GPUBuffer {
    const size = Math.max(4, count * 4);
    this.allocCount++;
    this.allocBytes += size;
    return this.device.createBuffer({
      size,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
      label,
    });
  }

  /** Release a buffer obtained from alloc(), keeping the accounting honest. */
  free(b: GPUBuffer): void {
    this.allocCount--;
    this.allocBytes -= b.size;
    b.destroy();
  }

  /**
   * Metadata buffers, cached by content.
   *
   * These describe shapes and strides, so for a fixed input size a node's values
   * are identical on every step - and many nodes share them outright. Allocating
   * fresh ones per dispatch cost ~8000 buffer creations per step, which
   * dominated everything the kernels actually did.
   */
  private metaCache = new Map<string, GPUBuffer>();

  private cached(values: number[], kind: 'i32' | 'f32'): GPUBuffer {
    const key = `${kind}:${values.join(',')}`;
    const hit = this.metaCache.get(key);
    if (hit) return hit;
    const b = this.device.createBuffer({
      size: Math.max(4, values.length * 4),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(
      b, 0, kind === 'i32' ? new Int32Array(values) : new Float32Array(values));
    this.metaCache.set(key, b);
    return b;
  }

  private i32Buffer(values: number[]): GPUBuffer {
    return this.cached(values, 'i32');
  }

  private f32Buffer(values: number[]): GPUBuffer {
    return this.cached(values, 'f32');
  }

  /** Drop cached metadata, e.g. when the input size changes. */
  clearMetaCache(): void {
    for (const b of this.metaCache.values()) b.destroy();
    this.metaCache.clear();
  }

  /** Dispatch with an explicit workgroup grid (for tiled kernels). */
  private runGrid(name: string, entries: GPUBindGroupEntry[],
                  grid: [number, number, number]): void {
    const pipeline = this.pipelines.get(name);
    if (!pipeline) throw new Error(`pipeline ${name} not compiled`);
    const bg = this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0), entries,
    });
    const enc = this.device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(grid[0], grid[1], grid[2]);
    pass.end();
    this.device.queue.submit([enc.finish()]);
  }

  private run(name: string, entries: GPUBindGroupEntry[], threads: number): void {
    const pipeline = this.pipelines.get(name);
    if (!pipeline) throw new Error(`pipeline ${name} not compiled`);
    const bg = this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0), entries,
    });
    const enc = this.device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bg);
    // Split across a 2-D grid: a single dimension caps at 65535 workgroups, and
    // exceeding it makes the dispatch silently do nothing.
    const groups = Math.max(1, Math.ceil(threads / 64));
    const gx = Math.min(groups, Runtime.MAX_GROUPS);
    const gy = Math.ceil(groups / Runtime.MAX_GROUPS);
    pass.dispatchWorkgroups(gx, gy);
    pass.end();
    this.device.queue.submit([enc.finish()]);
  }

  /**
   * Materialise `src` (read through `srcStrides`) into a contiguous `out`.
   *
   * When the source is already contiguous this is a straight run of elements,
   * and the copy engine does it far faster than a shader can: the gather kernel
   * evaluates rank-8 index arithmetic - eight integer divisions and modulos -
   * for every element, and integer division is one of the slowest things a GPU
   * does. aten::cat copies its slabs this way, and at 512x512 its eight nodes
   * cost as much as all seventy-eight convolutions before this path existed.
   */
  gather(src: Tensor, srcStrides: number[], outShape: number[], out: GPUBuffer,
         dstOffset = 0): void {
    const total = numel(outShape);

    // copyBufferToBuffer cannot take one buffer as both source and destination.
    if (total > 0 && src.buffer !== out && isContiguous(srcStrides, outShape)) {
      const enc = this.device.createCommandEncoder();
      enc.copyBufferToBuffer(
        src.buffer, (src.offset ?? 0) * 4, out, dstOffset * 4, total * 4);
      this.device.queue.submit([enc.finish()]);
      return;
    }

    const dims = [
      outShape.length, total,
      ...padTo(outShape, MAX_RANK, 1),
      ...padStrides(srcStrides, MAX_RANK),
      src.offset ?? 0,
      dstOffset,
    ];
    this.run('gather', [
      { binding: 0, resource: { buffer: this.i32Buffer(dims) } },
      { binding: 1, resource: { buffer: src.buffer } },
      { binding: 2, resource: { buffer: out } },
    ], total);
  }

  /**
   * Write the contiguous tensor `src` into `out` through `dstStrides`.
   *
   * The inverse of gather: one dispatch places a whole block inside a larger
   * tensor, however the block is laid out in the destination.
   */
  scatter(src: Tensor, srcShape: number[], dstStrides: number[], out: GPUBuffer,
          dstOffset = 0): void {
    const total = numel(srcShape);
    if (total === 0) return;

    if (src.buffer !== out && isContiguous(dstStrides, srcShape)) {
      const enc = this.device.createCommandEncoder();
      enc.copyBufferToBuffer(
        src.buffer, (src.offset ?? 0) * 4, out, dstOffset * 4, total * 4);
      this.device.queue.submit([enc.finish()]);
      return;
    }

    const dims = [
      srcShape.length, total,
      ...padTo(srcShape, MAX_RANK, 1),
      ...padStrides(dstStrides, MAX_RANK),
      src.offset ?? 0,
      dstOffset,
    ];
    this.run('scatter', [
      { binding: 0, resource: { buffer: this.i32Buffer(dims) } },
      { binding: 1, resource: { buffer: src.buffer } },
      { binding: 2, resource: { buffer: out } },
    ], total);
  }

  /** Contiguous copy of a tensor whose logical shape is `shape`. */
  contiguous(src: Tensor, out: GPUBuffer): void {
    this.gather(src, contiguousStrides(src.shape), src.shape, out);
  }

  elementwise(
    op: ElementwiseOpName,
    a: Tensor,
    b: Tensor | null,
    scalar: number,
    outShape: number[],
    out: GPUBuffer,
  ): void {
    const total = numel(outShape);
    const aStrides = broadcastStrides(a.shape, outShape);
    const bStrides = b ? broadcastStrides(b.shape, outShape) : new Array(outShape.length).fill(0);
    const dims = [
      outShape.length, total,
      ...padTo(outShape, MAX_RANK, 1),
      ...padStrides(aStrides, MAX_RANK),
      ...padStrides(bStrides, MAX_RANK),
      ElementwiseOp[op],
      b ? 1 : 0,
    ];
    this.run('elementwise', [
      { binding: 0, resource: { buffer: this.i32Buffer(dims) } },
      { binding: 1, resource: { buffer: this.f32Buffer([scalar]) } },
      { binding: 2, resource: { buffer: a.buffer } },
      { binding: 3, resource: { buffer: b ? b.buffer : this.dummy } },
      { binding: 4, resource: { buffer: out } },
    ], total);
  }

  /** Fill a buffer with a constant without reading it back as an input. */
  fill(value: number, outShape: number[], out: GPUBuffer): void {
    const total = numel(outShape);
    const dims = [
      outShape.length, total,
      ...padTo(outShape, MAX_RANK, 1),
      ...padStrides(new Array(outShape.length).fill(0), MAX_RANK),
      ...padStrides(new Array(outShape.length).fill(0), MAX_RANK),
      ElementwiseOp.fill, 0,
    ];
    this.run('elementwise', [
      { binding: 0, resource: { buffer: this.i32Buffer(dims) } },
      { binding: 1, resource: { buffer: this.f32Buffer([value]) } },
      { binding: 2, resource: { buffer: this.dummy } },
      { binding: 3, resource: { buffer: this.dummy } },
      { binding: 4, resource: { buffer: out } },
    ], total);
  }

  /** Sum or mean over `dims`, keepdim. */
  reduce(src: Tensor, reduceDims: number[], mean: boolean, out: GPUBuffer): number[] {
    const rank = src.shape.length;
    const outShape = src.shape.slice();
    const reduced = new Array(rank).fill(1);
    let count = 1;
    for (const d of reduceDims) {
      reduced[d] = src.shape[d];
      count *= src.shape[d];
      outShape[d] = 1;
    }
    const total = numel(outShape);
    const dimsBuf = [
      rank, total,
      ...padTo(outShape, MAX_RANK, 1),
      ...padStrides(contiguousStrides(src.shape), MAX_RANK),
      ...padTo(reduced, MAX_RANK, 1),
      count,
      mean ? 1 : 0,
    ];
    // One workgroup per output element, so the grid counts outputs directly.
    this.runGrid('reduce', [
      { binding: 0, resource: { buffer: this.i32Buffer(dimsBuf) } },
      { binding: 1, resource: { buffer: src.buffer } },
      { binding: 2, resource: { buffer: out } },
    ], Runtime.wgGrid(total));
    return outShape;
  }


  /** aten::convolution, forward or transposed. */
  conv2d(
    src: Tensor, weight: Tensor, bias: Tensor | null,
    p: {
      stride: number[]; padding: number[]; dilation: number[];
      transposed: boolean; groups: number; outShape: number[];
      outputPadding?: number[];
    },
    out: GPUBuffer,
  ): void {
    const [N, Cin, Hin, Win] = src.shape;
    const [, , KH, KW] = weight.shape;
    const [, Cout, Hout, Wout] = p.outShape;
    const total = numel(p.outShape);

    // Fast path: the shape almost every decoder convolution takes. Shared-memory
    // tiling only pays off when the 3x3 window is dense and unstrided, so the
    // general gather still handles everything else.
    const fastPath = p.groups === 1
      && KH === 3 && KW === 3
      && p.stride[0] === 1 && p.stride[1] === 1
      && p.padding[0] === 1 && p.padding[1] === 1
      && p.dilation[0] === 1 && p.dilation[1] === 1
      && Hout === Hin && Wout === Win
      && (!p.transposed || (p.outputPadding?.every((v) => v === 0) ?? true))
      && Cout % 4 === 0;     // conv3x3 blocks 4 output channels per workgroup
    if (this.convLog) {
      this.convLog.push({ fast: fastPath, N, Cin, Cout, Hin, Win,
        k: `${p.stride}|${p.padding}|g${p.groups}|${p.transposed ? 'T' : 'F'}` +
           `|${KH}x${KW}` });
    }
    if (fastPath) {
      const TILE = 16;
      this.runGrid('conv3x3', [
        { binding: 0, resource: { buffer: this.i32Buffer(
            [N, Cin, Hin, Win, Cout, bias ? 1 : 0, p.transposed ? 1 : 0]) } },
        { binding: 1, resource: { buffer: src.buffer } },
        { binding: 2, resource: { buffer: weight.buffer } },
        { binding: 3, resource: { buffer: bias ? bias.buffer : this.dummy } },
        { binding: 4, resource: { buffer: out } },
      ], [Math.ceil(Win / TILE), Math.ceil(Hin / TILE),
          N * Math.ceil(Cout / 4)]);   // COUT_BLOCK = 4 in conv3x3.wgsl
      return;
    }
    const dims = [
      N, Cin, Hin, Win, Cout, Hout, Wout, KH, KW,
      p.stride[0], p.stride[1], p.padding[0], p.padding[1],
      p.dilation[0], p.dilation[1],
      p.groups, p.transposed ? 1 : 0, bias ? 1 : 0, total,
    ];
    this.run('conv2d', [
      { binding: 0, resource: { buffer: this.i32Buffer(dims) } },
      { binding: 1, resource: { buffer: src.buffer } },
      { binding: 2, resource: { buffer: weight.buffer } },
      { binding: 3, resource: { buffer: bias ? bias.buffer : this.dummy } },
      { binding: 4, resource: { buffer: out } },
    ], total);
  }

  /** aten::native_group_norm -> (out, mean, rstd). */
  groupNorm(
    src: Tensor, weight: Tensor | null, bias: Tensor | null,
    N: number, C: number, HxW: number, G: number, eps: number,
    out: GPUBuffer, mean: GPUBuffer, rstd: GPUBuffer,
  ): void {
    const ng = N * G;
    this.runGrid('gnStats', [
      { binding: 0, resource: { buffer: this.i32Buffer([N, C, HxW, G, ng]) } },
      { binding: 1, resource: { buffer: this.f32Buffer([eps]) } },
      { binding: 2, resource: { buffer: src.buffer } },
      { binding: 3, resource: { buffer: mean } },
      { binding: 4, resource: { buffer: rstd } },
    ], Runtime.wgGrid(ng));
    const total = N * C * HxW;
    this.run('gnApply', [
      { binding: 0, resource: { buffer: this.i32Buffer(
          [N, C, HxW, G, total, weight ? 1 : 0, bias ? 1 : 0]) } },
      { binding: 1, resource: { buffer: src.buffer } },
      { binding: 2, resource: { buffer: mean } },
      { binding: 3, resource: { buffer: rstd } },
      { binding: 4, resource: { buffer: weight ? weight.buffer : this.dummy } },
      { binding: 5, resource: { buffer: bias ? bias.buffer : this.dummy } },
      { binding: 6, resource: { buffer: out } },
    ], total);
  }

  /** mm / bmm / addmm / baddbmm. */
  matmul(
    a: Tensor, b: Tensor, bias: Tensor | null,
    p: { B: number; M: number; K: number; N: number;
         alpha?: number; beta?: number; biasIsRowVector?: boolean;
         biasIsBatched?: boolean },
    out: GPUBuffer,
  ): void {
    const total = p.B * p.M * p.N;
    const dims = [
      p.B, p.M, p.K, p.N, total,
      bias ? 1 : 0, p.biasIsRowVector ? 1 : 0, p.biasIsBatched ? 1 : 0,
    ];
    this.run('matmul', [
      { binding: 0, resource: { buffer: this.i32Buffer(dims) } },
      { binding: 1, resource: { buffer: this.f32Buffer([p.alpha ?? 1, p.beta ?? 1]) } },
      { binding: 2, resource: { buffer: a.buffer } },
      { binding: 3, resource: { buffer: b.buffer } },
      { binding: 4, resource: { buffer: bias ? bias.buffer : this.dummy } },
      { binding: 5, resource: { buffer: out } },
    ], total);
  }

  /** Softmax along `dim` of a contiguous tensor. */
  softmax(src: Tensor, dim: number, out: GPUBuffer): void {
    const rank = src.shape.length;
    const d = dim < 0 ? dim + rank : dim;
    const len = src.shape[d];
    let inner = 1;
    for (let i = d + 1; i < rank; i++) inner *= src.shape[i];
    let outer = 1;
    for (let i = 0; i < d; i++) outer *= src.shape[i];
    const rows = outer * inner;
    this.run('softmax', [
      { binding: 0, resource: { buffer: this.i32Buffer([rows, len, inner]) } },
      { binding: 1, resource: { buffer: src.buffer } },
      { binding: 2, resource: { buffer: out } },
    ], rows);
  }

  /** Read a buffer back to the CPU. Test-only: this stalls the pipeline. */
  async read(buffer: GPUBuffer, count: number): Promise<Float32Array> {
    const bytes = Math.max(4, count * 4);
    const staging = this.device.createBuffer({
      size: bytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const enc = this.device.createCommandEncoder();
    enc.copyBufferToBuffer(buffer, 0, staging, 0, bytes);
    this.device.queue.submit([enc.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const data = new Float32Array(staging.getMappedRange().slice(0, count * 4));
    staging.unmap();
    staging.destroy();
    return data;
  }

  upload(data: Float32Array, label?: string): Tensor & { buffer: GPUBuffer } {
    const b = this.alloc(data.length, label);
    this.device.queue.writeBuffer(b, 0, data);
    return { buffer: b, shape: [data.length] };
  }
}
