import * as ort from 'onnxruntime-web/webgpu';

export interface Scaling { factor: number; shift: number; }

/**
 * Encode an image to its posterior mean, the initial latent.
 *
 * Deterministic: takes the mean rather than sampling, so repeated runs on the
 * same image produce identical trajectories.  The encoder session is released
 * before returning - it is dead weight for the optimization loop, and freeing it
 * returns GPU memory exactly when the joint graph needs headroom.
 */
export async function encodeLatent(
  encoderUrl: string,
  image: Float32Array,
  shape: number[],
  scaling: Scaling,
  externalData?: { path: string; data: string }[],
): Promise<{ data: Float32Array; shape: number[] }> {
  const session = await ort.InferenceSession.create(encoderUrl, {
    executionProviders: ['webgpu'],
    ...(externalData ? { externalData } : {}),
  });

  let latent: Float32Array;
  let latentShape: number[];
  try {
    const out = await session.run({
      image: new ort.Tensor('float32', image, shape),
    });
    const tensor = out.latent_mean as ort.Tensor;
    latentShape = tensor.dims as number[];
    const raw = (await tensor.getData(true)) as Float32Array;
    latent = new Float32Array(raw.length);
    for (let i = 0; i < raw.length; i++) {
      latent[i] = (raw[i] - scaling.shift) * scaling.factor;
    }
  } finally {
    await session.release();
  }

  // Returned on the CPU: the caller decides which device it belongs on, and the
  // two backends do not share a GPUDevice.
  return { data: latent, shape: latentShape };
}
