// Per-(batch, group) mean and reciprocal standard deviation.
// One thread per (n, g); the group spans C/G channels x HxW spatial elements.
//
// dims layout (i32): [0] N  [1] C  [2] HxW  [3] G  [4] total=N*G
// scalars: [0] eps

// Linear thread index across a 2-D dispatch grid.
//
// maxComputeWorkgroupsPerDimension is 65535, and a 512x512 activation needs
// ~98k workgroups. Exceeding the limit makes the dispatch invalid, and an
// invalid dispatch silently does nothing - the output buffer simply stays zero.
// The x extent is pinned to 65535 whenever a second row is needed, so this
// stride is a constant.
const DISPATCH_STRIDE : u32 = 4194240u;   // 65535 * 64

@group(0) @binding(0) var<storage, read> dims : array<i32>;
@group(0) @binding(1) var<storage, read> scalars : array<f32>;
@group(0) @binding(2) var<storage, read> src : array<f32>;
@group(0) @binding(3) var<storage, read_write> mean : array<f32>;
@group(0) @binding(4) var<storage, read_write> rstd : array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let ng = i32(gid.x + gid.y * DISPATCH_STRIDE);
  if (ng >= dims[4]) { return; }
  let C = dims[1]; let HxW = dims[2]; let G = dims[3];
  let n = ng / G;
  let g = ng % G;
  let cpg = C / G;
  let M = cpg * HxW;
  let base = (n * C + g * cpg) * HxW;

  // Kahan compensated summation: a group spans up to ~10^6 elements and plain
  // fp32 accumulation in one thread is not accurate enough for the gradient.
  var sum : f32 = 0.0;
  var c1 : f32 = 0.0;
  for (var k = 0; k < M; k = k + 1) {
    let y = src[base + k] - c1;
    let t = sum + y;
    c1 = (t - sum) - y;
    sum = t;
  }
  let mu = sum / f32(M);

  var sq : f32 = 0.0;
  var c2 : f32 = 0.0;
  for (var k = 0; k < M; k = k + 1) {
    let d = src[base + k] - mu;
    let y = d * d - c2;
    let t = sq + y;
    c2 = (t - sq) - y;
    sq = t;
  }
  mean[ng] = mu;
  rstd[ng] = 1.0 / sqrt(sq / f32(M) + scalars[0]);
}
