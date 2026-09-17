// Batched matrix multiply with an optional bias term, covering
// mm / bmm / addmm / baddbmm. mm is the batch-1 case.
//
// C[b, m, n] = beta*bias[...] + alpha * sum_k A[b, m, k] * B[b, k, n]
//
// dims layout (i32): [0] B [1] M [2] K [3] N [4] total
//                    [5] hasBias [6] biasBroadcastsRows [7] batchedBias
// scalars: [0] alpha  [1] beta

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
@group(0) @binding(2) var<storage, read> a : array<f32>;
@group(0) @binding(3) var<storage, read> b : array<f32>;
@group(0) @binding(4) var<storage, read> bias : array<f32>;
@group(0) @binding(5) var<storage, read_write> dst : array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let idx = i32(gid.x + gid.y * DISPATCH_STRIDE);
  if (idx >= dims[4]) { return; }
  let M = dims[1]; let K = dims[2]; let N = dims[3];

  let n = idx % N;
  let m = (idx / N) % M;
  let bt = idx / (N * M);

  var acc : f32 = 0.0;
  let aBase = (bt * M + m) * K;
  let bBase = bt * K * N;
  for (var k = 0; k < K; k = k + 1) {
    acc = acc + a[aBase + k] * b[bBase + k * N + n];
  }
  acc = acc * scalars[0];

  if (dims[5] == 1) {
    var bi = n;                                  // row vector bias
    if (dims[6] == 0) { bi = m * N + n; }        // full matrix bias
    if (dims[7] == 1) { bi = bi + bt * M * N; }  // per-batch bias
    acc = acc + scalars[1] * bias[bi];
  }
  dst[idx] = acc;
}
