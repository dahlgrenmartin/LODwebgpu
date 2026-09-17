import * as ort from 'onnxruntime-web/webgpu';
import type { Manifest } from './manifest';
import { createSession } from './session';
import { Runtime, createMaxDevice, type Tensor } from './wgsl/runtime';
import { Interpreter } from './wgsl/interpreter';

export type BackendName = 'ort' | 'wgsl';

export interface StepOutputs {
  loss: number;
  score: number;
  /** Decoder output, on `device`, contiguous NCHW fp32. */
  pred: GPUBuffer;
  /** Gradient w.r.t. the latent, on `device`. */
  grad: GPUBuffer;
}

export interface Backend {
  readonly name: BackendName;
  readonly device: GPUDevice;
  /** Non-null when only one exact square size is accepted. */
  readonly fixedSize: number | null;
  /** Upload a latent for this backend's device. */
  uploadLatent(data: Float32Array, shape: number[]): { buffer: GPUBuffer; shape: number[] };
  setTarget(data: Float32Array, shape: number[]): void;
  step(z: { buffer: GPUBuffer; shape: number[] }): Promise<StepOutputs>;
  dispose(): Promise<void>;
}

/**
 * ONNX Runtime Web on the WebGPU EP.
 *
 * Tuned kernels, but the graph is shape-static: one exported model per
 * resolution, so only that exact size is accepted.
 */
class OrtBackend implements Backend {
  readonly name = 'ort' as const;
  private target: ort.Tensor | null = null;

  constructor(
    readonly device: GPUDevice,
    readonly fixedSize: number,
    private runner: Awaited<ReturnType<typeof createSession>>,
  ) {}

  static async create(manifest: Manifest, base: string): Promise<OrtBackend> {
    const image = manifest.resolutions[0].image;
    const runner = await createSession(manifest, image, base, true);
    return new OrtBackend(runner.device, image, runner);
  }

  uploadLatent(data: Float32Array, shape: number[]) {
    const buffer = this.device.createBuffer({
      size: data.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    this.device.queue.writeBuffer(buffer, 0, data);
    return { buffer, shape };
  }

  setTarget(data: Float32Array, shape: number[]): void {
    this.target = new ort.Tensor('float32', data, shape);
  }

  async step(z: { buffer: GPUBuffer; shape: number[] }): Promise<StepOutputs> {
    if (!this.target) throw new Error('target not set');
    const zt = ort.Tensor.fromGpuBuffer(z.buffer, { dataType: 'float32', dims: z.shape });
    const out = await this.runner.run(zt, this.target);
    const scalar = async (t: ort.Tensor) => ((await t.getData(true)) as Float32Array)[0];
    return {
      loss: await scalar(out.loss as ort.Tensor),
      score: await scalar(out.score as ort.Tensor),
      pred: (out.pred as ort.Tensor).gpuBuffer as GPUBuffer,
      grad: (out.grad_z as ort.Tensor).gpuBuffer as GPUBuffer,
    };
  }

  async dispose(): Promise<void> {
    await this.runner.dispose();
  }
}

/**
 * The WGSL interpreter.
 *
 * Shapes are solved from the input tensor at run time, so any size the decoder's
 * 8x downsample allows is accepted with the pixels untouched. Its convolution is
 * an untiled gather, so it trades speed for that.
 */
class WgslBackend implements Backend {
  readonly name = 'wgsl' as const;
  readonly fixedSize = null;
  private target: { buffer: GPUBuffer; shape: number[] } | null = null;

  constructor(
    readonly device: GPUDevice,
    private rt: Runtime,
    private interp: Interpreter,
  ) {}

  static async create(base: string): Promise<WgslBackend> {
    const device = await createMaxDevice();
    const rt = await Runtime.create(device);
    const interp = await Interpreter.load(rt, `${base}/lod_graph.json`, base);
    return new WgslBackend(device, rt, interp);
  }

  uploadLatent(data: Float32Array, shape: number[]) {
    const buffer = this.rt.alloc(data.length, 'latent');
    this.device.queue.writeBuffer(buffer, 0, data);
    return { buffer, shape };
  }

  setTarget(data: Float32Array, shape: number[]): void {
    const buffer = this.rt.alloc(data.length, 'target');
    this.device.queue.writeBuffer(buffer, 0, data);
    this.target = { buffer, shape };
  }

  async step(z: { buffer: GPUBuffer; shape: number[] }): Promise<StepOutputs> {
    if (!this.target) throw new Error('target not set');
    const out = await this.interp.run(z, this.target);
    const read1 = async (t: Tensor) => (await this.rt.read(t.buffer, 1))[0];
    return {
      loss: await read1(out.loss),
      score: await read1(out.score),
      pred: out.pred.buffer,
      grad: out.grad_z.buffer,
    };
  }

  async dispose(): Promise<void> {
    this.device.destroy();
  }
}

export async function createBackend(
  name: BackendName, manifest: Manifest, base: string,
): Promise<Backend> {
  return name === 'ort'
    ? OrtBackend.create(manifest, base)
    : WgslBackend.create(base);
}
