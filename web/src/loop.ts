import * as ort from 'onnxruntime-web/webgpu';
import { Adam } from './adam';
import type { AdamConfig } from './manifest';
import type { Runner } from './session';

export interface LoopResult {
  losses: number[];
  scores: number[];
  msPerStep: number;
}

/**
 * Run the latent optimization headlessly.
 *
 * `z` is read back to the CPU each step because the session takes a CPU tensor
 * here; Milestone 2 replaces that with a gpu-buffer input binding so the latent
 * never leaves the device.  Adam's moments already stay resident.
 */
export async function runLoop(
  runner: Runner,
  cfg: AdamConfig,
  z0: Float32Array,
  target: Float32Array,
  zShape: number[],
  targetShape: number[],
  device: GPUDevice,
): Promise<LoopResult> {
  const numel = z0.length;
  const zBuf = device.createBuffer({
    size: numel * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
  });
  device.queue.writeBuffer(zBuf, 0, z0);

  const gradBuf = device.createBuffer({
    size: numel * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  const readBuf = device.createBuffer({
    size: numel * 4,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });

  const adam = await new Adam(device, numel, cfg).init();
  const targetTensor = new ort.Tensor('float32', target, targetShape);
  const losses: number[] = [];
  const scores: number[] = [];

  let z = z0.slice();
  const t0 = performance.now();
  for (let step = 0; step < cfg.steps; step++) {
    const out = await runner.run(new ort.Tensor('float32', z, zShape), targetTensor);
    const loss = ((await (out.loss as ort.Tensor).getData(true)) as Float32Array)[0];
    const score = ((await (out.score as ort.Tensor).getData(true)) as Float32Array)[0];
    losses.push(loss);
    scores.push(score);
    if (!Number.isFinite(loss)) {
      throw new Error(`loss diverged to NaN at step ${step}`);
    }

    const grad = (await (out.grad_z as ort.Tensor).getData(true)) as Float32Array;
    device.queue.writeBuffer(gradBuf, 0, grad);
    adam.step(zBuf, gradBuf);

    const enc = device.createCommandEncoder();
    enc.copyBufferToBuffer(zBuf, 0, readBuf, 0, numel * 4);
    device.queue.submit([enc.finish()]);
    await readBuf.mapAsync(GPUMapMode.READ);
    z = new Float32Array(readBuf.getMappedRange().slice(0));
    readBuf.unmap();
  }
  const msPerStep = (performance.now() - t0) / cfg.steps;

  return { losses, scores, msPerStep };
}
