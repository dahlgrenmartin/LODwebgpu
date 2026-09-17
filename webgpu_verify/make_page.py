#!/usr/bin/env python3
"""Generate a self-contained WebGPU verification page for the LOD rewrites.

For each rewrite in flux2_targeted_rewrites.py this exports:
  * the inputs,
  * the ground-truth output computed by the ORIGINAL aten *_backward op,

and emits an HTML page that recomputes the same quantity in WGSL using only
forward-style operations, then reports the fp32 error against PyTorch.

    python webgpu_verify/make_page.py
    python -m http.server 8731 --directory webgpu_verify
    # open http://localhost:8731/
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F

OUT = Path(__file__).resolve().parent
torch.manual_seed(7)


def t2l(x: torch.Tensor):
    return x.detach().to(torch.float32).flatten().tolist()


CASES = []


def conv_backward_case(name, x_shape, w_shape, stride, padding, dilation, groups):
    """dInput of a forward conv == transposed convolution (the conv_bwd rewrite)."""
    x = torch.randn(*x_shape)
    w = torch.randn(*w_shape)
    y = F.conv2d(x, w, None, stride, padding, dilation, groups)
    grad = torch.randn_like(y)
    dx, _, _ = torch.ops.aten.convolution_backward(
        grad, x, w, None,
        list(stride), list(padding), list(dilation),
        False, [0, 0], groups, [True, False, False],
    )
    CASES.append({
        "name": name,
        "kind": "conv_transpose",
        "meta": {
            "N": x_shape[0], "Cin": x_shape[1], "Hin": x_shape[2], "Win": x_shape[3],
            "Cout": y.shape[1], "Hout": y.shape[2], "Wout": y.shape[3],
            "KH": w_shape[2], "KW": w_shape[3],
            "sh": stride[0], "sw": stride[1],
            "ph": padding[0], "pw": padding[1],
            "dh": dilation[0], "dw": dilation[1],
            "groups": groups,
        },
        "inputs": {"grad": t2l(grad), "weight": t2l(w)},
        "expected": t2l(dx),
        "out_numel": dx.numel(),
        "desc": (f"x{list(x_shape)} w{list(w_shape)} stride={list(stride)} "
                 f"pad={list(padding)} groups={groups}"),
    })


def group_norm_backward_case(name, shape, G):
    """dInput of GroupNorm == the explicit mean/rstd formula (the gn_bwd rewrite)."""
    N, C = shape[0], shape[1]
    HxW = 1
    for d in shape[2:]:
        HxW *= d
    x = torch.randn(*shape)
    weight = torch.randn(C)
    bias = torch.randn(C)
    eps = 1e-6
    out, mean, rstd = torch.ops.aten.native_group_norm(x, weight, bias, N, C, HxW, G, eps)
    grad = torch.randn_like(out)
    dx, _, _ = torch.ops.aten.native_group_norm_backward(
        grad, x, mean, rstd, weight, N, C, HxW, G, [True, False, False],
    )
    CASES.append({
        "name": name,
        "kind": "group_norm_backward",
        "meta": {"N": N, "C": C, "HxW": HxW, "G": G},
        "inputs": {"grad": t2l(grad), "x": t2l(x), "mean": t2l(mean),
                   "rstd": t2l(rstd), "weight": t2l(weight)},
        "expected": t2l(dx),
        "out_numel": dx.numel(),
        "desc": f"x{list(shape)} groups={G} (rank {len(shape)})",
    })


def softmax_backward_case(name, shape, dim):
    """softmax backward == (dy - sum(dy*y)) * y (the softmax_bwd rewrite)."""
    x = torch.randn(*shape)
    y = torch.softmax(x, dim=dim)
    grad = torch.randn_like(y)
    dx = torch.ops.aten._softmax_backward_data(grad, y, dim, y.dtype)
    rows = 1
    for d in shape[:-1]:
        rows *= d
    CASES.append({
        "name": name, "kind": "softmax_backward",
        "meta": {"rows": rows, "cols": shape[-1]},
        "inputs": {"grad": t2l(grad), "y": t2l(y)},
        "expected": t2l(dx), "out_numel": dx.numel(),
        "desc": f"{list(shape)} dim={dim}",
    })


def sigmoid_backward_case(name, numel):
    """sigmoid backward == dy*y*(1-y) (the sigmoid_bwd rewrite)."""
    x = torch.randn(numel)
    y = torch.sigmoid(x)
    grad = torch.randn_like(y)
    dx = torch.ops.aten.sigmoid_backward(grad, y)
    CASES.append({
        "name": name, "kind": "sigmoid_backward",
        "meta": {"n": numel},
        "inputs": {"grad": t2l(grad), "y": t2l(y)},
        "expected": t2l(dx), "out_numel": numel,
        "desc": f"[{numel}]",
    })


# --- the cases that actually occur in the FLUX.2 joint graph ----------------
conv_backward_case("conv3x3 dInput (decoder body)",
                   (1, 8, 7, 7), (12, 8, 3, 3), (1, 1), (1, 1), (1, 1), 1)
conv_backward_case("conv1x1 dInput (post_quant / shortcut)",
                   (1, 8, 5, 5), (16, 8, 1, 1), (1, 1), (0, 0), (1, 1), 1)
conv_backward_case("sym4 grouped stride-2 dInput (wavelet loss)",
                   (1, 3, 28, 28), (9, 1, 8, 8), (2, 2), (0, 0), (1, 1), 3)
group_norm_backward_case("GroupNorm dInput rank-4 (conv path)", (1, 8, 6, 6), 4)
group_norm_backward_case("GroupNorm dInput rank-3 (attn path)", (1, 8, 9), 4)
softmax_backward_case("softmax dInput (VAE attention)", (1, 16, 16), -1)
sigmoid_backward_case("sigmoid dInput (SiLU chain)", 257)

payload = json.dumps(CASES)

HTML = """<!DOCTYPE html>
<html lang="en" data-theme="dark">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LOD Backward Rewrites on WebGPU</title>
<style>
  :root {
    --bg:#0f1115; --panel:#161922; --ink:#e6e9ef; --muted:#9aa4b2;
    --ok:#3fb950; --bad:#f85149; --accent:#58a6ff; --line:#262b36;
    --mono: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  }
  :root:not([data-theme="dark"]) {
    --bg:#ffffff; --panel:#f6f8fa; --ink:#1f2328; --muted:#636c76;
    --ok:#1a7f37; --bad:#cf222e; --accent:#0969da; --line:#d0d7de;
  }
  * { box-sizing:border-box; }
  body { margin:0; padding:32px 16px; background:var(--bg); color:var(--ink);
         font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }
  .wrap { max-width:1000px; margin:0 auto; }
  h1 { font-size:22px; margin:0 0 4px; letter-spacing:-0.01em; }
  .sub { color:var(--muted); font-size:13px; margin-bottom:24px; }
  #banner { padding:14px 18px; border-radius:10px; font-weight:600;
            margin-bottom:20px; border:1px solid var(--line); background:var(--panel); }
  #banner.ok  { border-color:var(--ok);  color:var(--ok); }
  #banner.bad { border-color:var(--bad); color:var(--bad); }
  table { width:100%; border-collapse:collapse; font-size:13px; }
  th,td { text-align:left; padding:9px 10px; border-bottom:1px solid var(--line); }
  th { color:var(--muted); font-weight:600; font-size:11px; text-transform:uppercase;
       letter-spacing:0.06em; }
  td.num { font-family:var(--mono); text-align:right; white-space:nowrap; }
  .pass { color:var(--ok); font-weight:600; }
  .fail { color:var(--bad); font-weight:600; }
  .desc { color:var(--muted); font-family:var(--mono); font-size:11px; }
  #env { margin-top:22px; padding:14px 16px; background:var(--panel);
         border:1px solid var(--line); border-radius:10px;
         font-family:var(--mono); font-size:12px; color:var(--muted);
         white-space:pre-wrap; word-break:break-word; }
  @media (max-width:640px){ body{padding:20px 16px;} .desc{display:none;} }
</style>
</head>
<body>
<div class="wrap">
  <h1>LOD backward rewrites on WebGPU</h1>
  <div class="sub">Each row recomputes a PyTorch <code>*_backward</code> result in WGSL
  using only forward-style ops, then compares against the reference computed by the
  original aten op.</div>
  <div id="banner">Initializing WebGPU&hellip;</div>
  <table>
    <thead><tr>
      <th>Rewrite</th><th>Case</th>
      <th style="text-align:right">max abs err</th>
      <th style="text-align:right">max rel err</th>
      <th style="text-align:right">n</th><th>Result</th>
    </tr></thead>
    <tbody id="rows"></tbody>
  </table>
  <div id="env">collecting adapter info&hellip;</div>
</div>

<script id="cases" type="application/json">__PAYLOAD__</script>
<script>
const CASES = JSON.parse(document.getElementById('cases').textContent);
const TOL = 2e-5;

const WGSL = {
conv_transpose: `
struct Meta { N:u32, Cin:u32, Hin:u32, Win:u32, Cout:u32, Hout:u32, Wout:u32,
              KH:u32, KW:u32, sh:u32, sw:u32, ph:u32, pw:u32, dh:u32, dw:u32, groups:u32 };
@group(0) @binding(0) var<uniform> m : Meta;
@group(0) @binding(1) var<storage, read> grad : array<f32>;
@group(0) @binding(2) var<storage, read> w : array<f32>;
@group(0) @binding(3) var<storage, read_write> out : array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let idx = gid.x;
  let total = m.N * m.Cin * m.Hin * m.Win;
  if (idx >= total) { return; }
  let iw = idx % m.Win;
  let ih = (idx / m.Win) % m.Hin;
  let ci = (idx / (m.Win * m.Hin)) % m.Cin;
  let n  =  idx / (m.Win * m.Hin * m.Cin);

  let cinPerG = m.Cin / m.groups;
  let coutPerG = m.Cout / m.groups;
  let g   = ci / cinPerG;
  let cig = ci % cinPerG;

  var acc : f32 = 0.0;
  for (var kh : u32 = 0u; kh < m.KH; kh = kh + 1u) {
    let th = i32(ih) + i32(m.ph) - i32(kh * m.dh);
    if (th < 0) { continue; }
    if (th % i32(m.sh) != 0) { continue; }
    let oh = th / i32(m.sh);
    if (oh >= i32(m.Hout)) { continue; }
    for (var kw : u32 = 0u; kw < m.KW; kw = kw + 1u) {
      let tw = i32(iw) + i32(m.pw) - i32(kw * m.dw);
      if (tw < 0) { continue; }
      if (tw % i32(m.sw) != 0) { continue; }
      let ow = tw / i32(m.sw);
      if (ow >= i32(m.Wout)) { continue; }
      for (var j : u32 = 0u; j < coutPerG; j = j + 1u) {
        let co = g * coutPerG + j;
        let gi = ((n * m.Cout + co) * m.Hout + u32(oh)) * m.Wout + u32(ow);
        let wi = ((co * cinPerG + cig) * m.KH + kh) * m.KW + kw;
        acc = acc + grad[gi] * w[wi];
      }
    }
  }
  out[idx] = acc;
}`,

group_norm_backward: `
struct Meta { N:u32, C:u32, HxW:u32, G:u32 };
@group(0) @binding(0) var<uniform> m : Meta;
@group(0) @binding(1) var<storage, read> grad : array<f32>;
@group(0) @binding(2) var<storage, read> x : array<f32>;
@group(0) @binding(3) var<storage, read> mean : array<f32>;
@group(0) @binding(4) var<storage, read> rstd : array<f32>;
@group(0) @binding(5) var<storage, read> weight : array<f32>;
@group(0) @binding(6) var<storage, read_write> out : array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let ng = gid.x;
  if (ng >= m.N * m.G) { return; }
  let n = ng / m.G;
  let g = ng % m.G;
  let cpg = m.C / m.G;
  let M = cpg * m.HxW;
  let mu = mean[ng];
  let rs = rstd[ng];
  let base = (n * m.C + g * cpg) * m.HxW;

  var s1 : f32 = 0.0;
  var s2 : f32 = 0.0;
  for (var k : u32 = 0u; k < M; k = k + 1u) {
    let c = g * cpg + k / m.HxW;
    let dyg = grad[base + k] * weight[c];
    let xhat = (x[base + k] - mu) * rs;
    s1 = s1 + dyg;
    s2 = s2 + dyg * xhat;
  }
  let invM = 1.0 / f32(M);
  for (var k : u32 = 0u; k < M; k = k + 1u) {
    let c = g * cpg + k / m.HxW;
    let dyg = grad[base + k] * weight[c];
    let xhat = (x[base + k] - mu) * rs;
    out[base + k] = (rs * invM) * (f32(M) * dyg - s1 - xhat * s2);
  }
}`,

softmax_backward: `
struct Meta { rows:u32, cols:u32 };
@group(0) @binding(0) var<uniform> m : Meta;
@group(0) @binding(1) var<storage, read> grad : array<f32>;
@group(0) @binding(2) var<storage, read> y : array<f32>;
@group(0) @binding(3) var<storage, read_write> out : array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let r = gid.x;
  if (r >= m.rows) { return; }
  let base = r * m.cols;
  var s : f32 = 0.0;
  for (var c : u32 = 0u; c < m.cols; c = c + 1u) { s = s + grad[base+c] * y[base+c]; }
  for (var c : u32 = 0u; c < m.cols; c = c + 1u) {
    out[base+c] = (grad[base+c] - s) * y[base+c];
  }
}`,

sigmoid_backward: `
struct Meta { n:u32 };
@group(0) @binding(0) var<uniform> m : Meta;
@group(0) @binding(1) var<storage, read> grad : array<f32>;
@group(0) @binding(2) var<storage, read> y : array<f32>;
@group(0) @binding(3) var<storage, read_write> out : array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  if (i >= m.n) { return; }
  out[i] = grad[i] * y[i] * (1.0 - y[i]);
}`
};

const META_ORDER = {
  conv_transpose: ["N","Cin","Hin","Win","Cout","Hout","Wout","KH","KW",
                   "sh","sw","ph","pw","dh","dw","groups"],
  group_norm_backward: ["N","C","HxW","G"],
  softmax_backward: ["rows","cols"],
  sigmoid_backward: ["n"],
};
const INPUT_ORDER = {
  conv_transpose: ["grad","weight"],
  group_norm_backward: ["grad","x","mean","rstd","weight"],
  softmax_backward: ["grad","y"],
  sigmoid_backward: ["grad","y"],
};
const THREADS = {
  conv_transpose: c => c.out_numel,
  group_norm_backward: c => c.meta.N * c.meta.G,
  softmax_backward: c => c.meta.rows,
  sigmoid_backward: c => c.meta.n,
};

function banner(txt, cls) {
  const b = document.getElementById('banner');
  b.textContent = txt;
  b.className = cls || '';
}

async function runCase(device, c) {
  const module = device.createShaderModule({ code: WGSL[c.kind] });
  const info = await module.getCompilationInfo();
  const errs = info.messages.filter(m => m.type === 'error');
  if (errs.length) throw new Error('WGSL: ' + errs.map(e => e.message).join('; '));

  const pipeline = device.createComputePipeline({
    layout: 'auto', compute: { module, entryPoint: 'main' },
  });

  const metaVals = META_ORDER[c.kind].map(k => c.meta[k]);
  while (metaVals.length % 4 !== 0) metaVals.push(0);
  const metaBuf = device.createBuffer({
    size: metaVals.length * 4,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(metaBuf, 0, new Uint32Array(metaVals));

  const entries = [{ binding: 0, resource: { buffer: metaBuf } }];
  let binding = 1;
  for (const key of INPUT_ORDER[c.kind]) {
    const arr = new Float32Array(c.inputs[key]);
    const buf = device.createBuffer({
      size: Math.max(4, arr.byteLength),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(buf, 0, arr);
    entries.push({ binding: binding++, resource: { buffer: buf } });
  }
  const outBytes = Math.max(4, c.out_numel * 4);
  const outBuf = device.createBuffer({
    size: outBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
  entries.push({ binding: binding, resource: { buffer: outBuf } });

  const bg = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries });
  const enc = device.createCommandEncoder();
  const pass = enc.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bg);
  pass.dispatchWorkgroups(Math.ceil(THREADS[c.kind](c) / 64));
  pass.end();

  const read = device.createBuffer({
    size: outBytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  enc.copyBufferToBuffer(outBuf, 0, read, 0, outBytes);
  device.queue.submit([enc.finish()]);

  await read.mapAsync(GPUMapMode.READ);
  const got = new Float32Array(read.getMappedRange().slice(0));
  read.unmap();

  const exp = c.expected;
  let maxAbs = 0, maxRel = 0;
  for (let i = 0; i < exp.length; i++) {
    const a = Math.abs(got[i] - exp[i]);
    if (a > maxAbs) maxAbs = a;
    const denom = Math.max(1e-6, Math.abs(exp[i]));
    const r = a / denom;
    if (r > maxRel) maxRel = r;
  }
  return { maxAbs, maxRel, n: exp.length };
}

(async () => {
  const results = [];
  const tbody = document.getElementById('rows');
  const envEl = document.getElementById('env');
  try {
    if (!navigator.gpu) throw new Error('navigator.gpu is undefined - WebGPU unavailable');
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error('requestAdapter() returned null');
    const device = await adapter.requestDevice();
    let adapterDesc = 'unknown adapter';
    try {
      const ai = adapter.info || (adapter.requestAdapterInfo
        ? await adapter.requestAdapterInfo() : null);
      if (ai) adapterDesc = [ai.vendor, ai.architecture, ai.device, ai.description]
        .filter(Boolean).join(' / ') || 'unnamed adapter';
    } catch (e) { /* adapter info optional */ }
    envEl.textContent = 'adapter: ' + adapterDesc
      + '\\nmaxComputeInvocationsPerWorkgroup: ' + device.limits.maxComputeInvocationsPerWorkgroup
      + '\\nmaxStorageBufferBindingSize: ' + device.limits.maxStorageBufferBindingSize
      + '\\ntolerance: ' + TOL + ' (max abs error)'
      + '\\nuserAgent: ' + navigator.userAgent;

    for (const c of CASES) {
      let row;
      try {
        const r = await runCase(device, c);
        const ok = r.maxAbs <= TOL && Number.isFinite(r.maxAbs);
        row = { name: c.name, desc: c.desc, ...r, pass: ok };
      } catch (e) {
        row = { name: c.name, desc: c.desc, maxAbs: NaN, maxRel: NaN,
                n: c.out_numel, pass: false, error: String(e.message || e) };
      }
      results.push(row);
      const tr = document.createElement('tr');
      tr.innerHTML =
        '<td>' + row.name + '</td>' +
        '<td class="desc">' + (row.error ? row.error : row.desc) + '</td>' +
        '<td class="num">' + (Number.isFinite(row.maxAbs) ? row.maxAbs.toExponential(2) : '-') + '</td>' +
        '<td class="num">' + (Number.isFinite(row.maxRel) ? row.maxRel.toExponential(2) : '-') + '</td>' +
        '<td class="num">' + row.n + '</td>' +
        '<td class="' + (row.pass ? 'pass' : 'fail') + '">' +
          (row.pass ? 'PASS' : 'FAIL') + '</td>';
      tbody.appendChild(tr);
    }

    const passed = results.filter(r => r.pass).length;
    const allOk = passed === results.length;
    const worst = Math.max(...results.map(r => Number.isFinite(r.maxAbs) ? r.maxAbs : Infinity));
    banner(
      (allOk ? 'ALL PASS' : 'FAILURES PRESENT') +
      ' - ' + passed + '/' + results.length +
      ' rewrites verified on WebGPU, worst max-abs-error ' +
      (Number.isFinite(worst) ? worst.toExponential(2) : 'n/a'),
      allOk ? 'ok' : 'bad');
    window.__RESULTS__ = { status: allOk ? 'ALL_PASS' : 'FAIL',
                           passed, total: results.length, worst, results,
                           adapter: adapterDesc };
  } catch (e) {
    banner('WebGPU ERROR - ' + (e.message || e), 'bad');
    envEl.textContent = 'userAgent: ' + navigator.userAgent;
    window.__RESULTS__ = { status: 'ERROR', error: String(e.message || e) };
  }
  document.title = 'WebGPU verify: ' + (window.__RESULTS__.status || 'unknown');
  window.__DONE__ = true;
})();
</script>
</body>
</html>
"""

page = HTML.replace("__PAYLOAD__", payload)
(OUT / "index.html").write_text(page, encoding="utf-8")
print(f"wrote {OUT / 'index.html'}  ({len(page)/1024:.1f} KB, {len(CASES)} cases)")
for c in CASES:
    print(f"  {c['kind']:22s} {c['name']}  -> {c['out_numel']} values")
