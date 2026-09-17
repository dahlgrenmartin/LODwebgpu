import * as ort from 'onnxruntime-web/webgpu';
import type { Manifest, Resolution } from './manifest';

export interface Runner {
  device: GPUDevice;
  session: ort.InferenceSession;
  resolution: Resolution;
  run(z: ort.Tensor, target: ort.Tensor): Promise<ort.InferenceSession.OnnxValueMapType>;
  dispose(): Promise<void>;
}

export async function createDevice(): Promise<GPUDevice> {
  if (!navigator.gpu) throw new Error('WebGPU unavailable: navigator.gpu undefined');
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error('WebGPU unavailable: requestAdapter() returned null');
  return adapter.requestDevice();
}

/**
 * Create the joint-graph session on a device we own, so a render pass can bind
 * ORT's output buffers directly instead of round-tripping through the CPU.
 */
export async function createSession(
  manifest: Manifest, image: number, baseUrl: string, device: GPUDevice,
): Promise<Runner> {
  const resolution = manifest.resolutions.find((r) => r.image === image);
  if (!resolution) throw new Error(`no resolution ${image} in manifest`);

  ort.env.webgpu.device = device;
  // ORT-Web cannot resolve external data from the filesystem the way native ORT
  // does; the weights file has to be handed over explicitly, keyed by the same
  // path recorded inside the model.
  const session = await ort.InferenceSession.create(
    `${baseUrl}/${resolution.graph}`,
    {
      executionProviders: ['webgpu'],
      externalData: [{ path: manifest.weights, data: `${baseUrl}/${manifest.weights}` }],
    },
  );

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
