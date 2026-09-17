// Strided scatter: reads a contiguous input and writes it through arbitrary
// strides. The mirror of gather.wgsl, and what placing a block inside a larger
// tensor needs - aten::cat writes each part into its stripe of the output.
//
// Doing that with gather instead means one dispatch per contiguous row, because
// only the innermost run is contiguous on both sides. The Sym4 detector pads
// along the last two dimensions, so at 512x512 that was ~4600 dispatches for a
// single cat node and made eight of them cost as much as all 78 convolutions.
//
// dims layout (i32):
//   [0] rank
//   [1] total input elements
//   [2 .. 9]  input shape (right-aligned to MAX_RANK)
//   [10..17]  output strides, in elements
//   [18]      input base offset, in elements
//   [19]      output base offset, in elements

const MAX_RANK : u32 = 8u;

// See gather.wgsl: 65535 workgroups per dimension, 64 threads each.
const DISPATCH_STRIDE : u32 = 4194240u;

@group(0) @binding(0) var<storage, read> dims : array<i32>;
@group(0) @binding(1) var<storage, read> src : array<f32>;
@group(0) @binding(2) var<storage, read_write> dst : array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x + gid.y * DISPATCH_STRIDE;
  let total = u32(dims[1]);
  if (i >= total) { return; }

  let rank = u32(dims[0]);
  var rem = i;
  var offset = dims[19];
  for (var k : u32 = 0u; k < rank; k = k + 1u) {
    let d = MAX_RANK - 1u - k;
    let extent = u32(dims[2 + i32(d)]);
    let coord = rem % extent;
    rem = rem / extent;
    offset = offset + i32(coord) * dims[10 + i32(d)];
  }
  dst[u32(offset)] = src[u32(dims[18]) + i];
}
