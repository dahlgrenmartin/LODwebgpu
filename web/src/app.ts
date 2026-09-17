import { loadManifest, type Manifest } from './manifest';
import { encodeLatent } from './latentInit';
import { prepareImage, sampleImage, type PreparedImage } from './imageInput';
import { Display } from './display';
import { Adam } from './adam';
import { createBackend, type Backend, type BackendName } from './backends';
import { bandPsnr, linearSlope } from './detectorMath';

const BASE = '/models';
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
};

let manifest: Manifest;
let backend: Backend | null = null;
let display: Display | null = null;
let busy = false;

function setState(state: State, message: string): void {
  els.status.textContent = message;
  els.status.dataset.state = state;
  (window as any).__STATE__ = state;
}

function setControlsEnabled(enabled: boolean): void {
  els.file.disabled = !enabled;
  els.sample.disabled = !enabled;
  els.backend.disabled = !enabled;
}

function describeBackend(b: Backend): string {
  return b.fixedSize
    ? `ONNX Runtime Web — fixed ${b.fixedSize}x${b.fixedSize}`
    : 'WGSL interpreter — any size divisible by 8';
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
  }
  setState('loading',
    `Loading ${name === 'ort' ? 'ONNX Runtime Web' : 'the WGSL interpreter'}…`);
  backend = await createBackend(name, manifest, BASE);

  backend.device.lost.then((info) => {
    setState('error', `GPU device lost: ${info.reason}. Reload to restart.`);
    setControlsEnabled(false);
  });

  els.crop.textContent = describeBackend(backend);
  setState('ready', 'Ready — drop an image, choose a file, or use the sample.');
  setControlsEnabled(true);
}

async function boot(): Promise<void> {
  setState('boot', 'Checking WebGPU…');
  if (!navigator.gpu) {
    throw new Error(
      'This demo needs WebGPU. Chrome or Edge 113+ on desktop supports it; ' +
      'Safari 18+ and Firefox need it enabled. There is deliberately no CPU ' +
      'fallback — 30 optimization steps through a 28M-parameter decoder would ' +
      'take minutes rather than seconds.');
  }
  manifest = await loadManifest(`${BASE}/manifest.json`);
  els.backend.addEventListener('change', () => {
    void useBackend(els.backend.value as BackendName).catch((e) => {
      setState('error', (e as Error).message);
    });
  });
  await useBackend(els.backend.value as BackendName);
}

async function run(source: Blob): Promise<void> {
  if (busy || !backend) return;
  busy = true;
  setControlsEnabled(false);
  try {
    setState('encoding', 'Preparing image…');
    let image: PreparedImage;
    try {
      image = await prepareImage(source, { exact: backend.fixedSize, multipleOf: 8 });
    } catch (e) {
      setState('error', (e as Error).message);
      return;
    }
    const { w, h } = image.sourceSize;
    els.before.width = w;
    els.before.height = h;
    els.before.getContext('2d')!.putImageData(image.preview, 0, 0);
    els.crop.textContent =
      `${w}x${h} in, unaltered — no resize, no crop · ${describeBackend(backend)}`;

    setState('encoding', 'Encoding to latent…');
    const latent = await encodeLatent(
      `${BASE}/${manifest.encoder}`, image.data, image.shape,
      { factor: 1.0, shift: 0.0 },
      [{ path: manifest.encoderWeights,
         data: `${BASE}/${manifest.encoderWeights}` }]);

    // The two backends do not share a GPUDevice, so the latent is uploaded to
    // whichever one is active.
    const z = backend.uploadLatent(latent.data, latent.shape);
    backend.setTarget(image.data, image.shape);

    display = await new Display(backend.device, els.after, w, h).init();

    const adam = await new Adam(backend.device, latent.data.length, manifest.adam).init();
    const steps = manifest.adam.steps;

    let last = performance.now();
    let rising = 0;
    let previousLoss = Infinity;
    const bandPsnrHistory: number[] = [];
    let detectorSlope = Number.NaN;

    for (let step = 0; step < steps; step++) {
      // One step per frame keeps the page responsive and self-paces to the GPU.
      await new Promise((r) => requestAnimationFrame(r));
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
    void run(sampleImage(backend?.fixedSize ?? 256));
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
(window as any).__sampleImage = (size?: number) =>
  sampleImage(size ?? backend?.fixedSize ?? 256);
(window as any).__useBackend = (n: BackendName) => {
  els.backend.value = n;
  return useBackend(n);
};

wireInputs();
boot().catch((e) => {
  setState('error', (e as Error).message);
  setControlsEnabled(false);
});