import gatherSrc from './kernels/gather.wgsl?raw';
import elementwiseSrc from './kernels/elementwise.wgsl?raw';
import reduceSrc from './kernels/reduce.wgsl?raw';
import conv2dSrc from './kernels/conv2d.wgsl?raw';
import gnStatsSrc from './kernels/groupnorm_stats.wgsl?raw';
import gnApplySrc from './kernels/groupnorm_apply.wgsl?raw';
import matmulSrc from './kernels/matmul.wgsl?raw';
import softmaxSrc from './kernels/softmax.wgsl?raw';
import {
  broadcastStrides, contiguousStrides, numel, padStrides, padTo,
} from './shapes';

export const MAX_RANK = 8;

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
  private pipelines = new Map<string, GPUComputePipeline>();

  private constructor(readonly device: GPUDevice) {}

  static async create(device: GPUDevice): Promise<Runtime> {
    const rt = new Runtime(device);
    await rt.compile('gather', gatherSrc);
    await rt.compile('elementwise', elementwiseSrc);
    await rt.compile('reduce', reduceSrc);
    await rt.compile('conv2d', conv2dSrc);
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

  alloc(count: number, label?: string): GPUBuffer {
    return this.device.createBuffer({
      size: Math.max(4, count * 4),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
      label,
    });
  }

  private i32Buffer(values: number[]): GPUBuffer {
    const b = this.device.createBuffer({
      size: Math.max(4, values.length * 4),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(b, 0, new Int32Array(values));
    return b;
  }

  private f32Buffer(values: number[]): GPUBuffer {
    const b = this.device.createBuffer({
      size: Math.max(4, values.length * 4),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(b, 0, new Float32Array(values));
    return b;
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
    pass.dispatchWorkgroups(Math.max(1, Math.ceil(threads / 64)));
    pass.end();
    this.device.queue.submit([enc.finish()]);
  }

  /** Materialise `src` (read through `srcStrides`) into a contiguous `out`. */
  gather(src: Tensor, srcStrides: number[], outShape: number[], out: GPUBuffer,
         dstOffset = 0): void {
    const total = numel(outShape);
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
      { binding: 3, resource: { buffer: (b ?? a).buffer } },
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
    this.run('reduce', [
      { binding: 0, resource: { buffer: this.i32Buffer(dimsBuf) } },
      { binding: 1, resource: { buffer: src.buffer } },
      { binding: 2, resource: { buffer: out } },
    ], total);
    return outShape;
  }


  /** aten::convolution, forward or transposed. */
  conv2d(
    src: Tensor, weight: Tensor, bias: Tensor | null,
    p: {
      stride: number[]; padding: number[]; dilation: number[];
      transposed: boolean; groups: number; outShape: number[];
    },
    out: GPUBuffer,
  ): void {
    const [N, Cin, Hin, Win] = src.shape;
    const [, , KH, KW] = weight.shape;
    const [, Cout, Hout, Wout] = p.outShape;
    const total = numel(p.outShape);
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
      { binding: 3, resource: { buffer: (bias ?? src).buffer } },
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
    this.run('gnStats', [
      { binding: 0, resource: { buffer: this.i32Buffer([N, C, HxW, G, ng]) } },
      { binding: 1, resource: { buffer: this.f32Buffer([eps]) } },
      { binding: 2, resource: { buffer: src.buffer } },
      { binding: 3, resource: { buffer: mean } },
      { binding: 4, resource: { buffer: rstd } },
    ], ng);
    const total = N * C * HxW;
    this.run('gnApply', [
      { binding: 0, resource: { buffer: this.i32Buffer(
          [N, C, HxW, G, total, weight ? 1 : 0, bias ? 1 : 0]) } },
      { binding: 1, resource: { buffer: src.buffer } },
      { binding: 2, resource: { buffer: mean } },
      { binding: 3, resource: { buffer: rstd } },
      { binding: 4, resource: { buffer: (weight ?? src).buffer } },
      { binding: 5, resource: { buffer: (bias ?? src).buffer } },
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
      { binding: 4, resource: { buffer: (bias ?? a).buffer } },
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
