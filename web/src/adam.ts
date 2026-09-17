import wgsl from './adam.wgsl?raw';
import type { AdamConfig } from './manifest';

/**
 * Adam over the latent, resident on the GPU.
 *
 * Owns `m`, `v` and the step counter so the caller only supplies the gradient
 * buffer; `z` is updated in place and never leaves the device.
 */
export class Adam {
  private pipeline!: GPUComputePipeline;
  private params: GPUBuffer;
  private m: GPUBuffer;
  private v: GPUBuffer;
  private module: GPUShaderModule;
  private t = 0;

  constructor(
    private device: GPUDevice,
    private numel: number,
    private cfg: AdamConfig,
  ) {
    this.module = device.createShaderModule({ code: wgsl });
    this.params = device.createBuffer({
      size: 32,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.m = this.zeroed();
    this.v = this.zeroed();
  }

  /**
   * Build the pipeline, failing loudly on a WGSL compile error.  Without this a
   * bad shader yields an invalid pipeline whose dispatches are silently dropped,
   * which reads as "the kernel computed the identity".
   */
  async init(): Promise<this> {
    const info = await this.module.getCompilationInfo();
    const errors = info.messages.filter((msg) => msg.type === 'error');
    if (errors.length) {
      throw new Error('adam.wgsl failed to compile: ' +
        errors.map((e) => `${e.lineNum}:${e.linePos} ${e.message}`).join('; '));
    }
    this.pipeline = this.device.createComputePipeline({
      layout: 'auto',
      compute: { module: this.module, entryPoint: 'main' },
    });
    return this;
  }

  private zeroed(): GPUBuffer {
    const b = this.device.createBuffer({
      size: this.numel * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(b, 0, new Float32Array(this.numel));
    return b;
  }

  reset(): void {
    this.t = 0;
    const zeros = new Float32Array(this.numel);
    this.device.queue.writeBuffer(this.m, 0, zeros);
    this.device.queue.writeBuffer(this.v, 0, zeros);
  }

  step(zBuf: GPUBuffer, gradBuf: GPUBuffer): void {
    if (!this.pipeline) throw new Error('Adam.init() must be awaited before step()');
    this.t += 1;
    const { lr, beta1, beta2, eps } = this.cfg;
    const buf = new ArrayBuffer(32);
    new Uint32Array(buf, 0, 1)[0] = this.numel;
    new Float32Array(buf, 4, 6).set([
      lr, beta1, beta2, eps,
      1 - Math.pow(beta1, this.t),
      1 - Math.pow(beta2, this.t),
    ]);
    this.device.queue.writeBuffer(this.params, 0, buf);

    const bindGroup = this.device.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.params } },
        { binding: 1, resource: { buffer: zBuf } },
        { binding: 2, resource: { buffer: gradBuf } },
        { binding: 3, resource: { buffer: this.m } },
        { binding: 4, resource: { buffer: this.v } },
      ],
    });

    const enc = this.device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(this.numel / 64));
    pass.end();
    this.device.queue.submit([enc.finish()]);
  }
}
