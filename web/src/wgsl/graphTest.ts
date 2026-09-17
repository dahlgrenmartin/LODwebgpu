import { Runtime } from './runtime';
import { Interpreter } from './interpreter';

interface RefEntry { offset: number; count: number; shape: number[] }

/**
 * Run the whole rewritten graph through the WGSL interpreter and compare against
 * the PyTorch golden tensors, the same ones the ORT-Web harness uses.
 */
export async function runGraphTest(device: GPUDevice, base: string, image: number,
                                   trace = false) {
  const rt = await Runtime.create(device);
  const [meta, buf] = await Promise.all([
    fetch(`${base}/reference_${image}.json`).then((r) => r.json()),
    fetch(`${base}/reference_${image}.bin`).then((r) => r.arrayBuffer()),
  ]);
  const all = new Float32Array(buf);
  const golden = (name: string) => {
    const e = meta[name] as RefEntry;
    return { data: all.subarray(e.offset, e.offset + e.count), shape: e.shape };
  };

  const interp = await Interpreter.load(rt, `${base}/lod_graph.json`, base);

  const upload = (t: { data: Float32Array; shape: number[] }) => {
    const b = rt.alloc(t.data.length);
    rt.device.queue.writeBuffer(b, 0, t.data);
    return { buffer: b, shape: t.shape };
  };

  if (trace) interp.trace = [];
  const t0 = performance.now();
  const out = await interp.run(upload(golden('z')), upload(golden('target')));
  const elapsed = performance.now() - t0;

  const results = [];
  for (const label of ['loss', 'pred', 'score', 'grad_z']) {
    const want = golden(label);
    const t = out[label];
    const got = await rt.read(t.buffer, want.data.length);
    // RMS-relative is the meaningful aggregate here. grad_z's largest element is
    // ~7e-6, so a max-over-max ratio is dominated by a few near-zero entries
    // where fp32 cancellation decides the digits, and says little about whether
    // the gradient is right.
    let se = 0;
    let sw = 0;
    let maxAbs = 0;
    let maxVal = 0;
    for (let i = 0; i < want.data.length; i++) {
      const d = got[i] - want.data[i];
      se += d * d;
      sw += want.data[i] * want.data[i];
      maxAbs = Math.max(maxAbs, Math.abs(d));
      maxVal = Math.max(maxVal, Math.abs(want.data[i]));
    }
    const rms = Math.sqrt(se) / Math.max(1e-20, Math.sqrt(sw));
    results.push({
      name: label, shape: t.shape.join('x'),
      relErr: rms, maxRel: maxAbs / Math.max(1e-20, maxVal),
      pass: rms <= 5e-3,
    });
  }
  const passed = results.filter((r) => r.pass).length;
  // First node that collapses to all-zero while something upstream was not.
  let firstZero = null;
  if (interp.trace) {
    for (let i = 1; i < interp.trace.length; i++) {
      const t = interp.trace[i];
      if (t.absmax === 0 && interp.trace[i - 1].absmax !== 0) { firstZero = { index: i, ...t }; break; }
    }
  }
  return {
    status: passed === results.length ? 'ALL_PASS' : 'FAIL',
    passed, total: results.length, ms: Math.round(elapsed), results,
    firstZero, traced: interp.trace ? interp.trace.length : 0,
    zeros: interp.trace ? interp.trace.filter((t) => t.absmax === 0).length : 0,
  };
}
