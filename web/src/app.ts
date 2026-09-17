import { loadManifest, type Manifest } from './manifest';
import { encodeLatent } from './latentInit';
import { prepareImage, sampleImage, type PreparedImage } from './imageInput';
import { Display } from './display';
import { Adam } from './adam';
import { createBackend, type Backend, type BackendName } from './backends';
import { bandPsnr, linearSlope } from './detectorMath';
import { parseOptimizerConfig } from './optimizerConfig';

// Resolve against the deploy prefix: a GitHub project page serves the
// site from /<repo>/, where an absolute '/models' 404s.
const BASE = `${import.meta.env.BASE_URL}models`.replace(/\/{2,}/g, '/');
const DETECTOR_WINDOW = 10;

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
  crop: document.getElementById('crop') as HTMLElement,
  backend: document.getElementById('backend') as HTMLSelectElement,
  optLr: document.getElementById('opt-lr') as HTMLInputElement,
  optBeta1: document.getElementById('opt-beta1') as HTMLInputElement,
  optBeta2: document.getElementById('opt-beta2') as HTMLInputElement,
  optEps: document.getElementById('opt-eps') as HTMLInputElement,
  optSteps: document.getElementById('opt-steps') as HTMLInputElement,
};

const optimizerInputs = [
  els.optLr, els.optBeta1, els.optBeta2, els.optEps, els.optSteps,
];

let manifest: Manifest;
let backend: Backend | null = null;
let display: Display | null = null;
let busy = false;
let watchedDevice: GPUDevice | null = null;

function setState(state: State, message: string): void {
  els.status.textContent = message;
  els.status.dataset.state = state;
  (window as any).__STATE__ = state;
}

function setControlsEnabled(enabled: boolean): void {
  els.file.disabled = !enabled;
  els.sample.disabled = !enabled;
  els.backend.disabled = !enabled;
  for (const input of optimizerInputs) input.disabled = !enabled;
}

function populateOptimizerControls(): void {
  els.optLr.value = String(manifest.adam.lr);
  els.optBeta1.value = String(manifest.adam.beta1);
  els.optBeta2.value = String(manifest.adam.beta2);
  els.optEps.value = String(manifest.adam.eps);
  els.optSteps.value = String(manifest.adam.steps);
}

function readOptimizerConfig() {
  return parseOptimizerConfig({
    lr: els.optLr.value,
    beta1: els.optBeta1.value,
    beta2: els.optBeta2.value,
    eps: els.optEps.value,
    steps: els.optSteps.value,
  });
}

function describeModel(): string {
  const id = manifest?.model;
  const ch = manifest?.latentChannels ?? manifest?.resolutions?.[0]?.latent?.[1];
  if (!id) return ch ? `unknown model, ${ch}-channel latent` : 'unknown model';
  return `${id.split('/').pop()} (${ch}-channel latent)`;
}

function describeBackend(b: Backend): string {
  if (b.supportedSizes) {
    const sizes = b.supportedSizes.map((s) => `${s.width}x${s.height}`).join(', ');
    return `ONNX Runtime Web — static graphs: ${sizes}`;
  }
  return 'WGSL interpreter — any size divisible by 8';
}

function watchDevice(device: GPUDevice): void {
  if (device === watchedDevice) return;
  watchedDevice = device;
  device.lost.then((info) => {
    if (watchedDevice !== device) return;
    setState('error', `GPU device lost: ${info.reason}. Reload to restart.`);
    setControlsEnabled(false);
  });
}

function formatSlope(s: number): string {
  if (!Number.isFinite(s)) return '—';
  return `${s >= 0 ? '+' : ''}${s.toFixed(4)}`;
}

async function useBackend(name: BackendName): Promise<void> {
  if (backend) {
    await backend.dispose();
    backend = null;
    display = null;
    watchedDevice = null;
  }
  setState('loading',
    `Loading ${name === 'ort' ? 'ONNX Runtime Web' : 'the WGSL interpreter'}…`);
  backend = await createBackend(name, manifest, BASE);

  // WGSL owns a device immediately. ORT stays lazy until an exact image shape is
  // known, so its device is watched after prepare(width, height) in run().
  if (!backend.supportedSizes) watchDevice(backend.device);

  els.crop.textContent = `${describeModel()} · ${describeBackend(backend)}`;
  setState('ready', 'Ready — drop an image, choose a file, or use the sample.');
  setControlsEnabled(true);
}

async function boot(): Promise<void> {
  setState('boot', 'Checking WebGPU…');
  if (!navigator.gpu) {
    throw new Error(
      'This demo needs WebGPU. Chrome or Edge 113+ on desktop supports it; ' +
      'Safari 18+ and Firefox need it enabled. There is deliberately no CPU ' +
      'fallback — latent refinement through a 28M-parameter decoder would ' +
      'take minutes rather than seconds.');
  }
  manifest = await loadManifest(`${BASE}/manifest.json`);
  populateOptimizerControls();
  els.backend.addEventListener('change', () => {
    void useBackend(els.backend.value as BackendName).catch((e) => {
      setState('error', (e as Error).message);
    });
  });
  await useBackend(els.backend.value as BackendName);
}

async function run(source: Blob): Promise<void> {
  if (busy || !backend) return;

  let optimizer;
  try {
    optimizer = readOptimizerConfig();
  } catch (e) {
    setState('error', `Optimizer: ${(e as Error).message}`);
    return;
  }

  busy = true;
  setControlsEnabled(false);
  try {
    setState('encoding', 'Preparing image…');
    let image: PreparedImage;
    try {
      image = await prepareImage(source, backend.supportedSizes
        ? { allowed: backend.supportedSizes }
        : { multipleOf: 8 });
    } catch (e) {
      setState('error', (e as Error).message);
      return;
    }
    const { w, h } = image.sourceSize;
    els.before.width = w;
    els.before.height = h;
    els.before.getContext('2d')!.putImageData(image.preview, 0, 0);
    els.crop.textContent =
      `${w}x${h} in, unaltered — no resize, no crop · ` +
      `${describeModel()} · ${describeBackend(backend)}`;

    setState('encoding', 'Encoding to latent…');
    const latent = await encodeLatent(
      `${BASE}/${manifest.encoder}`, image.data, image.shape,
      { factor: 1.0, shift: 0.0 },
      [{ path: manifest.encoderWeights,
         data: `${BASE}/${manifest.encoderWeights}` }]);

    if (backend.supportedSizes) {
      setState('loading', `Loading ONNX graph for ${w}x${h}…`);
    }
    await backend.prepare(w, h);
    watchDevice(backend.device);

    // The two backends do not share a GPUDevice, so the latent is uploaded to
    // whichever one is active after shape-specific preparation has completed.
    const z = backend.uploadLatent(latent.data, latent.shape);
    backend.setTarget(image.data, image.shape);

    display = await new Display(backend.device, els.after, w, h).init();

    const adam = await new Adam(backend.device, latent.data.length, optimizer).init();
    const steps = optimizer.steps;

    let last = performance.now();
    let rising = 0;
    let previousLoss = Infinity;
    const bandPsnrHistory: number[] = [];
    let detectorSlope = Number.NaN;

    for (let step = 0; step < steps; step++) {
      // Yield to the event loop, but do NOT wait for a compositor frame.
      // requestAnimationFrame self-paces to the display, and the display cannot
      // present while our own submitted work saturates the GPU: the callback is
      // deferred until the queue drains. Measured at 256x256 that cost ~780 ms
      // per step on top of ~220 ms of actual compute - more than 4x the work.
      await new Promise((r) => setTimeout(r, 0));
      const out = await backend.step(z);
      display.draw(out.pred);

      if (!Number.isFinite(out.loss)) {
        setState('error', `Diverged to NaN at step ${step + 1}. Stopped.`);
        return;
      }
      rising = out.loss > previousLoss ? rising + 1 : 0;
      previousLoss = out.loss;
      if (rising >= 3) {
        setState('error',
          `Loss rose for 3 consecutive steps (step ${step + 1}). Stopped.`);
        return;
      }

      // The graph returns the reference L3 off-diagonal residual energy.
      // Convert it to band-PSNR on the CPU, then fit the same OLS slope over the
      // latest 10 iterations as dahlgrenmartin/LOD/src/LOD.py.
      const psnr = bandPsnr(out.score);
      bandPsnrHistory.push(psnr);
      detectorSlope = linearSlope(bandPsnrHistory, DETECTOR_WINDOW);

      const now = performance.now();
      els.step.textContent = `${step + 1} / ${steps}`;
      els.loss.textContent = out.loss.toFixed(5);
      els.score.textContent = `${psnr.toFixed(3)} dB · slope ${formatSlope(detectorSlope)}`;
      els.rate.textContent = `${Math.round(now - last)} ms`;
      last = now;
      setState('running', `Optimizing… step ${step + 1} of ${steps}`);

      adam.step(z.buffer, out.grad);
    }

    const verdict = detectorSlope > 0 ? 'SYNTHETIC' : 'REAL';
    (window as any).__DETECTOR__ = {
      bandPsnr: bandPsnrHistory.slice(), slope: detectorSlope, verdict,
    };
    setState('done',
      `Done — ${verdict} · slope ${formatSlope(detectorSlope)} over the last ` +
      `${Math.min(DETECTOR_WINDOW, bandPsnrHistory.length)} steps · ${backend.name} backend.`);
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
  els.sample.addEventListener('click', () => {
    const size = backend?.supportedSizes?.[0];
    void run(sampleImage(size?.width ?? 256, size?.height ?? 256));
  });

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
(window as any).__sampleImage = (width?: number, height?: number) => {
  if (width != null) return sampleImage(width, height ?? width);
  const size = backend?.supportedSizes?.[0];
  return sampleImage(size?.width ?? 256, size?.height ?? 256);
};
(window as any).__useBackend = (n: BackendName) => {
  els.backend.value = n;
  return useBackend(n);
};

wireInputs();
boot().catch((e) => {
  setState('error', (e as Error).message);
  setControlsEnabled(false);
});
