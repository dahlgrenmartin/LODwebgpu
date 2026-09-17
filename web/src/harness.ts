import * as ort from 'onnxruntime-web/webgpu';
import { loadManifest } from './manifest';
import { createSession } from './session';

// Resolve against the deploy prefix: a GitHub project page serves the
// site from /<repo>/, where an absolute '/models' 404s.
const BASE = `${import.meta.env.BASE_URL}models`.replace(/\/{2,}/g, '/');
// Relative, to match the Python suite, and sized to include the accepted fp16
// weight-storage cost. The golden data comes from the fp32 PyTorch wrapper, so
// this budget covers quantization + WebGPU fp32 reassociation together.
const TOL = 6e-2;

interface RefEntry { offset: number; count: number; shape: number[]; }
interface Case { name: string; maxAbs: number; pass: boolean; error?: string; }

async function loadGolden(image: number) {
  const [meta, buf] = await Promise.all([
    fetch(`${BASE}/reference_${image}.json`).then((r) => r.json()),
    fetch(`${BASE}/reference_${image}.bin`).then((r) => r.arrayBuffer()),
  ]);
  const all = new Float32Array(buf);
  return (name: string) => {
    const e = meta[name] as RefEntry;
    if (!e) throw new Error(`golden tensor ${name} missing`);
    return { data: all.subarray(e.offset, e.offset + e.count), shape: e.shape };
  };
}

function relErr(got: Float32Array, want: Float32Array): number {
  if (got.length !== want.length) return Infinity;
  let diff = 0;
  let scale = 0;
  for (let i = 0; i < got.length; i++) {
    diff = Math.max(diff, Math.abs(got[i] - want[i]));
    scale = Math.max(scale, Math.abs(want[i]));
  }
  return diff / Math.max(1e-6, scale);
}

function render(results: { status: string; cases: Case[] }) {
  const rows = results.cases.map((c) =>
    `${c.pass ? 'PASS' : 'FAIL'}  ${c.name.padEnd(34)} ` +
    `${Number.isFinite(c.maxAbs) ? c.maxAbs.toExponential(2) : '-'}` +
    (c.error ? `\n        ${c.error}` : '')).join('\n');
  document.body.textContent = `${results.status}\n\n${rows}\n`;
}

async function main() {
  const cases: Case[] = [];
  try {
    const manifest = await loadManifest(`${BASE}/manifest.json`);
    // Golden data is generated only for square resolutions, named by side.
    const resolution = manifest.resolutions.find((r) => r.width === r.height);
    if (!resolution) {
      throw new Error('no square resolution available to verify against');
    }
    const image = resolution.width;
    const golden = await loadGolden(image);
    const runner = await createSession(
      manifest, resolution.width, resolution.height, BASE);
    const device = runner.device;

    const z = golden('z');
    const target = golden('target');
    const mk = (t: { data: Float32Array; shape: number[] }) =>
      new ort.Tensor('float32', t.data, t.shape);

    const out = await runner.run(mk(z), mk(target));
    for (const name of ['loss', 'pred', 'score', 'grad_z']) {
      const want = golden(name);
      const got = (await (out[name] as ort.Tensor).getData(true)) as Float32Array;
      const err = relErr(got, want.data);
      cases.push({ name, maxAbs: err, pass: err <= TOL });
    }

    // Negative control: a perturbed latent MUST diverge from the golden grad.
    const bad = new Float32Array(z.data);
    for (let i = 0; i < bad.length; i++) bad[i] += 0.5;
    const outBad = await runner.run(
      new ort.Tensor('float32', bad, z.shape), mk(target));
    const gradBad = (await (outBad.grad_z as ort.Tensor).getData(true)) as Float32Array;
    const errBad = relErr(gradBad, golden('grad_z').data);
    cases.push({
      name: 'negative control (perturbed z)',
      maxAbs: errBad,
      pass: errBad > TOL,
    });

    // --- Adam kernel vs the numpy reference ---
    {
      const g = await fetch(`${BASE}/adam_golden.json`).then((r) => r.json());
      const { Adam } = await import('./adam');
      const n = g.z0.length;
      const zBuf = device.createBuffer({
        size: n * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
      });
      device.queue.writeBuffer(zBuf, 0, new Float32Array(g.z0));
      const gradBuf = device.createBuffer({
        size: n * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      const adam = await new Adam(device, n, { ...g.cfg, steps: g.grads.length }).init();
      for (const grad of g.grads) {
        device.queue.writeBuffer(gradBuf, 0, new Float32Array(grad));
        adam.step(zBuf, gradBuf);
      }
      const read = device.createBuffer({
        size: n * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });
      const enc = device.createCommandEncoder();
      enc.copyBufferToBuffer(zBuf, 0, read, 0, n * 4);
      device.queue.submit([enc.finish()]);
      await read.mapAsync(GPUMapMode.READ);
      const got = new Float32Array(read.getMappedRange().slice(0));
      read.unmap();

      const want = new Float32Array(g.expected);
      const err = relErr(got, want);
      cases.push({ name: `adam ${g.grads.length} steps`, maxAbs: err, pass: err <= 1e-4 });

      const perturbed = new Float32Array(want);
      for (let i = 0; i < perturbed.length; i++) perturbed[i] += 0.5;
      const errCtl = relErr(got, perturbed);
      cases.push({
        name: 'negative control (adam)', maxAbs: errCtl, pass: errCtl > 1e-4,
      });
    }

    // --- encoder-seeded latent init ---
    {
      const { encodeLatent } = await import('./latentInit');
      const img = new Float32Array(1 * 3 * image * image).fill(0.25);
      const latent = await encodeLatent(
        `${BASE}/encoder.onnx`, img, [1, 3, image, image],
        { factor: 1.0, shift: 0.0 },
        [{ path: manifest.encoderWeights,
           data: `${BASE}/${manifest.encoderWeights}` }]);
      const expected = 32 * (image / 8) * (image / 8);
      cases.push({
        name: `latent init shape ${JSON.stringify(latent.shape)}`,
        maxAbs: Math.abs(latent.data.length - expected),
        pass: latent.data.length === expected,
      });

      const finite = latent.data.every((x) => Number.isFinite(x));
      const nonzero = latent.data.some((x) => x !== 0);
      cases.push({
        name: 'latent init finite and non-degenerate',
        maxAbs: finite && nonzero ? 0 : NaN, pass: finite && nonzero,
      });
    }

    // --- 30-step headless optimization ---
    {
      const { runLoop } = await import('./loop');
      const { losses, scores, msPerStep } = await runLoop(
        runner, manifest.adam, golden('z').data, golden('target').data,
        golden('z').shape, golden('target').shape, device);
      const finite = losses.every((l) => Number.isFinite(l));
      const improved = losses[losses.length - 1] < losses[0];
      const right = losses.length === manifest.adam.steps;
      cases.push({
        name: `loop ${losses.length} steps, loss ${losses[0].toFixed(5)} -> ` +
              `${losses[losses.length - 1].toFixed(5)}, ${msPerStep.toFixed(0)} ms/step`,
        maxAbs: losses[0] - losses[losses.length - 1],
        pass: finite && improved && right,
      });
      (window as any).__LOOP__ = { losses, scores, msPerStep };
    }

    await runner.dispose();
  } catch (e) {
    cases.push({
      name: 'harness', maxAbs: NaN, pass: false,
      error: String((e as Error)?.stack || e),
    });
  }

  const passed = cases.filter((c) => c.pass).length;
  const results = {
    status: passed === cases.length ? 'ALL_PASS' : 'FAIL',
    passed, total: cases.length, cases,
  };
  (window as any).__RESULTS__ = results;
  document.title = `ONNX parity: ${results.status}`;
  render(results);
  (window as any).__DONE__ = true;
}

main();
