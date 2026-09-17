// Normalise with the per-group statistics, then apply the affine terms.
// One thread per element.
//
// dims layout (i32): [0] N [1] C [2] HxW [3] G [4] total [5] hasWeight [6] hasBias

// Linear thread index across a 2-D dispatch grid.
//
// maxComputeWorkgroupsPerDimension is 65535, and a 512x512 activation needs
// ~98k workgroups. Exceeding the limit makes the dispatch invalid, and an
// invalid dispatch silently does nothing - the output buffer simply stays zero.
// The x extent is pinned to 65535 whenever a second row is needed, so this
// stride is a constant.
const DISPATCH_STRIDE : u32 = 4194240u;   // 65535 * 64

@group(0) @binding(0) var<storage, read> dims : array<i32>;
@group(0) @binding(1) var<storage, read> src : array<f32>;
@group(0) @binding(2) var<storage, read> mean : array<f32>;
@group(0) @binding(3) var<storage, read> rstd : array<f32>;
@group(0) @binding(4) var<storage, read> weight : array<f32>;
@group(0) @binding(5) var<storage, read> bias : array<f32>;
@group(0) @binding(6) var<storage, read_write> dst : array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = i32(gid.x + gid.y * DISPATCH_STRIDE);
  if (i >= dims[4]) { return; }
  let C = dims[1]; let HxW = dims[2]; let G = dims[3];
  let cpg = C / G;
  let c = (i / HxW) % C;
  let n = i / (HxW * C);
  let g = c / cpg;
  let ng = n * G + g;

  var v = (src[i] - mean[ng]) * rstd[ng];
  if (dims[5] == 1) { v = v * weight[c]; }
  if (dims[6] == 1) { v = v + bias[c]; }
  dst[i] = v;
}
