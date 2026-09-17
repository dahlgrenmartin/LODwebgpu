import wgsl from './display.wgsl?raw';

/**
 * Presents a decoder output buffer on a canvas without a GPU->CPU readback.
 *
 * The buffer produced by the inference session is bound directly as read-only
 * storage in the fragment stage, so displaying a frame costs one render pass
 * over the canvas rather than a pipeline stall.
 */
export class Display {
  private pipeline!: GPURenderPipeline;
  private context: GPUCanvasContext;
  private dims: GPUBuffer;
  private module: GPUShaderModule;
  private format: GPUTextureFormat;

  constructor(
    private device: GPUDevice,
    private canvas: HTMLCanvasElement,
    private width: number,
    private height: number,
  ) {
    const ctx = canvas.getContext('webgpu');
    if (!ctx) throw new Error('canvas.getContext("webgpu") returned null');
    this.context = ctx;
    this.format = navigator.gpu.getPreferredCanvasFormat();
    canvas.width = width;
    canvas.height = height;
    ctx.configure({ device, format: this.format, alphaMode: 'opaque' });

    this.module = device.createShaderModule({ code: wgsl });
    this.dims = device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(this.dims, 0, new Uint32Array([width, height, 0, 0]));
  }

  /** Build the pipeline, failing loudly on a WGSL compile error. */
  async init(): Promise<this> {
    const info = await this.module.getCompilationInfo();
    const errors = info.messages.filter((m) => m.type === 'error');
    if (errors.length) {
      throw new Error('display.wgsl failed to compile: ' +
        errors.map((e) => `${e.lineNum}:${e.linePos} ${e.message}`).join('; '));
    }
    this.pipeline = this.device.createRenderPipeline({
      layout: 'auto',
      vertex: { module: this.module, entryPoint: 'vs' },
      fragment: {
        module: this.module, entryPoint: 'fs',
        targets: [{ format: this.format }],
      },
      primitive: { topology: 'triangle-list' },
    });
    return this;
  }

  /** Draw one frame from an NCHW fp32 buffer of shape [1,3,height,width]. */
  draw(image: GPUBuffer): void {
    if (!this.pipeline) throw new Error('Display.init() must be awaited before draw()');
    const bindGroup = this.device.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.dims } },
        { binding: 1, resource: { buffer: image } },
      ],
    });

    const enc = this.device.createCommandEncoder();
    const pass = enc.beginRenderPass({
      colorAttachments: [{
        view: this.context.getCurrentTexture().createView(),
        clearValue: { r: 0, g: 0, b: 0, a: 1 },
        loadOp: 'clear',
        storeOp: 'store',
      }],
    });
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.draw(3);
    pass.end();
    this.device.queue.submit([enc.finish()]);
  }
}
