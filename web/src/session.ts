import * as ort from 'onnxruntime-web/webgpu';
import type { Manifest, Resolution } from './manifest';

export interface Runner {
  device: GPUDevice;
  session: ort.InferenceSession;
  resolution: Resolution;
  run(z: ort.Tensor, target: ort.Tensor): Promise<ort.InferenceSession.OnnxValueMapType>;
  dispose(): Promise<void>;
}

/**
 * Capability check only.  The device we actually render and compute with comes
 * from ORT (see createSession): ORT-Web creates its own GPUDevice, and buffers
 * are not shareable across devices, so adopting theirs is the only way to bind
 * session outputs in our own passes.
 */
export async function assertWebGpu(): Promise<GPUAdapterInfo | null> {
  if (!navigator.gpu) throw new Error('WebGPU unavailable: navigator.gpu undefined');
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error('WebGPU unavailable: requestAdapter() returned null');
  return adapter.info ?? null;
}

/**
 * Create the joint-graph session on a device we own, so a render pass can bind
 * ORT's output buffers directly instead of round-tripping through the CPU.
 */
export async function createSession(
  manifest: Manifest, image: number, baseUrl: string,
  outputsOnGpu = false,
): Promise<Runner> {
  const resolution = manifest.resolutions.find((r) => r.image === image);
  if (!resolution) throw new Error(`no resolution ${image} in manifest`);

  // ORT-Web cannot resolve external data from the filesystem the way native ORT
  // does; the weights file has to be handed over explicitly, keyed by the same
  // path recorded inside the model.
  const session = await ort.InferenceSession.create(
    `${baseUrl}/${resolution.graph}`,
    {
      executionProviders: ['webgpu'],
      externalData: [{ path: manifest.weights, data: `${baseUrl}/${manifest.weights}` }],
      // Keeping pred and grad_z on the device lets the render pass bind pred
      // directly and Adam consume grad_z without a round trip.
      ...(outputsOnGpu
        ? { preferredOutputLocation: { pred: 'gpu-buffer', grad_z: 'gpu-buffer' } }
        : {}),
    } as ort.InferenceSession.SessionOptions,
  );

  // ort.env.webgpu.device resolves to the device ORT created for this session.
  // It is a promise, not a plain property -- awaiting it is what makes buffer
  // interop work at all, since buffers cannot cross GPUDevice boundaries.
  const device = await ort.env.webgpu.device;
  if (!device) {
    throw new Error('ORT-Web did not expose a WebGPU device after session creation');
  }

  return {
    device,
    session,
    resolution,
    run(z, target) {
      return session.run({ z, target });
    },
    async dispose() {
      await session.release();
    },
  };
}
