// Per-(batch, group) mean and reciprocal standard deviation.
//
// One WORKGROUP per (n, g). The previous version used one thread per group, so
// a [1,256,256,256] GroupNorm reduced 262144 elements on 32 threads. Threads
// stride the group and combine through workgroup memory; two passes, because
// variance needs the mean and E[x^2]-E[x]^2 loses too much precision here.
//
// dims layout (i32): [0] N  [1] C  [2] HxW  [3] G  [4] total=N*G
// scalars: [0] eps

const GROUP : u32 = 256u;
const WG_STRIDE : u32 = 65535u;

@group(0) @binding(0) var<storage, read> dims : array<i32>;
@group(0) @binding(1) var<storage, read> scalars : array<f32>;
@group(0) @binding(2) var<storage, read> src : array<f32>;
@group(0) @binding(3) var<storage, read_write> mean : array<f32>;
@group(0) @binding(4) var<storage, read_write> rstd : array<f32>;

var<workgroup> partial : array<f32, 256>;

fn combine(lid : u32) {
  workgroupBarrier();
  var stride : u32 = GROUP / 2u;
  loop {
    if (stride == 0u) { break; }
    if (lid < stride) { partial[lid] = partial[lid] + partial[lid + stride]; }
    workgroupBarrier();
    stride = stride / 2u;
  }
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid : vec3<u32>,
        @builtin(local_invocation_id) lid : vec3<u32>) {
  let ng = wid.x + wid.y * WG_STRIDE;
  if (ng >= u32(dims[4])) { return; }

  let C = u32(dims[1]);
  let HxW = u32(dims[2]);
  let G = u32(dims[3]);
  let n = ng / G;
  let g = ng % G;
  let cpg = C / G;
  let M = cpg * HxW;
  let base = (n * C + g * cpg) * HxW;

  // Pass 1: mean.
  var acc : f32 = 0.0;
  var comp : f32 = 0.0;
  var k : u32 = lid.x;
  loop {
    if (k >= M) { break; }
    let y = src[base + k] - comp;
    let t = acc + y;
    comp = (t - acc) - y;
    acc = t;
    k = k + GROUP;
  }
  partial[lid.x] = acc;
  combine(lid.x);
  let mu = partial[0] / f32(M);
  workgroupBarrier();

  // Pass 2: variance about that mean.
  acc = 0.0;
  comp = 0.0;
  k = lid.x;
  loop {
    if (k >= M) { break; }
    let d = src[base + k] - mu;
    let y = d * d - comp;
    let t = acc + y;
    comp = (t - acc) - y;
    acc = t;
    k = k + GROUP;
  }
  partial[lid.x] = acc;
  combine(lid.x);

  if (lid.x == 0u) {
    mean[ng] = mu;
    rstd[ng] = 1.0 / sqrt(partial[0] / f32(M) + scalars[0]);
  }
}
