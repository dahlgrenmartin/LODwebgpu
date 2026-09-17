import * as ort from 'onnxruntime-web/webgpu';
import { loadManifest, type Manifest } from './manifest';
import { assertWebGpu, createSession, type Runner } from './session';
import { initLatent } from './latentInit';
import { prepareImage, sampleImage, type PreparedImage } from './imageInput';
import { Display } from './display';
import { Adam } from './adam';

const BASE = '/models';

type State = 'boot' | 'loading' | 'ready' | 'encoding' | 'running' | 'done' | 'error';

const els = {
  status: document.getElementById('status') as HTMLElement,
  file: document.getElementById('file') as HTMLInputElement,
  sample: document.getElementById('sample') as HTMLButtonElement,
  drop: document.getElementById('drop') as HTMLElement,
  before: document.getElementById('before') as HTMLCanvasElement,
  after: document.getElementById('after') as HTMLCanvasElement,
  step: document.getElementById('step') as HTMLElement,
  loss: document.getElementById('loss') as HTMLElement,
  score: document.getElementById('score') as HTMLElement,
  rate: document.getElementById('rate') as HTMLElement,
};

let manifest: Manifest;
let device: GPUDevice;
let runner: Runner;
let display: Display;
let imageSize = 128;
let busy = false;

function setState(state: State, message: string): void {
  els.status.textContent = message;
  els.status.dataset.state = state;
  (window as any).__STATE__ = state;
}

function setControlsEnabled(enabled: boolean): void {
  els.file.disabled = !enabled;
  els.sample.disabled = !enabled;
}

/** Read a scalar output regardless of whether it landed on CPU or GPU. */
async function scalar(t: ort.Tensor): Promise<number> {
  return ((await t.getData(true)) as Float32Array)[0];
}

async function boot(): Promise<void> {
  setState('boot', 'Checking WebGPU…');
  try {
    await assertWebGpu();
  } catch {
    throw new Error(
      'This demo needs WebGPU. Chrome or Edge 113+ on desktop supports it; ' +
      'Safari 18+ and Firefox need it enabled. There is deliberately no CPU ' +
      'fallback — 30 optimization steps through a 28M-parameter decoder would ' +
      'take minutes rather than seconds.');
  }

  setState('loading', 'Loading model (~56 MB, cached after the first visit)…');
  manifest = await loadManifest(`${BASE}/manifest.json`);
  imageSize = manifest.resolutions[0].image;

  // ORT creates the device; we adopt it so our render pass and Adam kernel can
  // bind the session's own output buffers.
  runner = await createSession(manifest, imageSize, BASE, true);
  device = runner.device;

  device.lost.then((info) => {
    setState('error', `GPU device lost: ${info.reason}. Reload to restart.`);
    setControlsEnabled(false);
  });

  display = await new Display(device, els.after, imageSize).init();

  els.before.width = imageSize;
  els.before.height = imageSize;

  setState('ready', 'Ready — drop an image, choose a file, or use the sample.');
  setControlsEnabled(true);
}

async function run(source: Blob): Promise<void> {
  if (busy) return;
  busy = true;
  setControlsEnabled(false);
  try {
    setState('encoding', 'Preparing image…');
    let image: PreparedImage;
    try {
      image = await prepareImage(source, imageSize);
    } catch (e) {
      setState('error', (e as Error).message);
      return;
    }
    els.before.getContext('2d')!.putImageData(image.preview, 0, 0);

    setState('encoding', 'Encoding to latent…');
    const { buffer: zBuf, numel, shape: zShape } = await initLatent(
      device, `${BASE}/encoder.onnx`, image.data, image.shape,
      { factor: 1.0, shift: 0.0 },
      [{ path: 'encoder.onnx.data', data: `${BASE}/encoder.onnx.data` }]);

    const target = new ort.Tensor('float32', image.data, image.shape);
    const adam = await new Adam(device, numel, manifest.adam).init();
    const steps = manifest.adam.steps;

    let last = performance.now();
    let rising = 0;
    let previousLoss = Infinity;

    for (let step = 0; step < steps; step++) {
      // One step per frame keeps the page responsive and self-paces to the GPU.
      await new Promise((r) => requestAnimationFrame(r));

      const z = ort.Tensor.fromGpuBuffer(zBuf, {
        dataType: 'float32', dims: zShape,
      });
      const out = await runner.run(z, target);

      const predBuf = (out.pred as ort.Tensor).gpuBuffer as GPUBuffer;
      display.draw(predBuf);

      const loss = await scalar(out.loss as ort.Tensor);
      const score = await scalar(out.score as ort.Tensor);

      if (!Number.isFinite(loss)) {
        setState('error', `Diverged to NaN at step ${step + 1}. Stopped.`);
        return;
      }
      rising = loss > previousLoss ? rising + 1 : 0;
      previousLoss = loss;
      if (rising >= 3) {
        setState('error',
          `Loss rose for 3 consecutive steps (step ${step + 1}). Stopped — the ` +
          `Adam hyperparameters in manifest.json are placeholders.`);
        return;
      }

      const now = performance.now();
      els.step.textContent = `${step + 1} / ${steps}`;
      els.loss.textContent = loss.toFixed(5);
      els.score.textContent = score.toFixed(5);
      els.rate.textContent = `${Math.round(now - last)} ms`;
      last = now;
      setState('running', `Optimizing… step ${step + 1} of ${steps}`);

      const gradBuf = (out.grad_z as ort.Tensor).gpuBuffer as GPUBuffer;
      adam.step(zBuf, gradBuf);
    }

    setState('done', `Done — ${steps} steps. Drop another image to run again.`);
  } catch (e) {
    setState('error', `${(e as Error).message}`);
    throw e;
  } finally {
    busy = false;
    setControlsEnabled(true);
  }
}

function wireInputs(): void {
  els.file.addEventListener('change', () => {
    const f = els.file.files?.[0];
    if (f) void run(f);
  });
  els.sample.addEventListener('click', () => void run(sampleImage(imageSize)));

  for (const evt of ['dragenter', 'dragover'] as const) {
    els.drop.addEventListener(evt, (e) => {
      e.preventDefault();
      els.drop.classList.add('over');
    });
  }
  for (const evt of ['dragleave', 'drop'] as const) {
    els.drop.addEventListener(evt, (e) => {
      e.preventDefault();
      els.drop.classList.remove('over');
    });
  }
  els.drop.addEventListener('drop', (e) => {
    const f = (e as DragEvent).dataTransfer?.files?.[0];
    if (f) void run(f);
  });
}

// Exposed so the browser harness can drive the same path a user would.
(window as any).__runDemo = (blob: Blob) => run(blob);
(window as any).__sampleImage = () => sampleImage(imageSize);

wireInputs();
boot().catch((e) => {
  setState('error', (e as Error).message);
  setControlsEnabled(false);
});
