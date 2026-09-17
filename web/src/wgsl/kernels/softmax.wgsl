// Softmax along one dimension, max-subtracted for stability.
//
// The tensor is treated as (outer, len, inner) around the softmax axis, so one
// thread handles one (outer, inner) pair and walks `len` elements `inner` apart.
//
// dims layout (i32): [0] rows = outer*inner  [1] len  [2] inner

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
@group(0) @binding(2) var<storage, read_write> dst : array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let r = i32(gid.x + gid.y * DISPATCH_STRIDE);
  if (r >= dims[0]) { return; }
  let len = dims[1];
  let inner = dims[2];
  let base = (r / inner) * len * inner + (r % inner);

  // Seed from the first element rather than an f32 sentinel: the literal for
  // -FLT_MAX is not exactly representable and the shader fails to compile.
  var mx : f32 = src[base];
  for (var i = 1; i < len; i = i + 1) {
    mx = max(mx, src[base + i * inner]);
  }
  var sum : f32 = 0.0;
  for (var i = 0; i < len; i = i + 1) {
    let e = exp(src[base + i * inner] - mx);
    dst[base + i * inner] = e;
    sum = sum + e;
  }
  for (var i = 0; i < len; i = i + 1) {
    dst[base + i * inner] = dst[base + i * inner] / sum;
  }
}
