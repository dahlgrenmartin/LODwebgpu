// Strided gather: writes a contiguous output by reading the input through
// arbitrary strides. One kernel serves permute, expand, slice, clone and any
// "make this contiguous" step, because all of them are just a stride pattern.
//
// dims layout (i32):
//   [0] rank
//   [1] total output elements
//   [2 .. 9]  output shape (right-aligned to MAX_RANK)
//   [10..17]  input strides, in elements (0 means broadcast)
//   [18]      input base offset, in elements

const MAX_RANK : u32 = 8u;

@group(0) @binding(0) var<storage, read> dims : array<i32>;
@group(0) @binding(1) var<storage, read> src : array<f32>;
@group(0) @binding(2) var<storage, read_write> dst : array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  let total = u32(dims[1]);
  if (i >= total) { return; }

  let rank = u32(dims[0]);
  var rem = i;
  var offset = dims[18];
  // Walk dimensions from the fastest-varying end.
  for (var k : u32 = 0u; k < rank; k = k + 1u) {
    let d = MAX_RANK - 1u - k;                 // shape/stride slot
    let extent = u32(dims[2 + i32(d)]);
    let coord = rem % extent;
    rem = rem / extent;
    offset = offset + i32(coord) * dims[10 + i32(d)];
  }
  dst[i] = src[u32(offset)];
}
