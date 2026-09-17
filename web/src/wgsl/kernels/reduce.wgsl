// Sum / mean over an arbitrary set of dimensions, keepdim semantics.
//
// One WORKGROUP per output element, not one thread. The previous version gave
// each output element a single thread that walked the whole reduction, so a
// GroupNorm mean over [1,32,262144] ran 32 threads on a 20000-core GPU and took
// 66 ms - while an elementwise pass over the same bytes takes 0.9 ms. Threads
// stride over the reduction and then combine through workgroup memory.
//
// dims layout (i32):
//   [0] rank
//   [1] total output elements
//   [2 .. 9]  output shape (right-aligned, reduced dims are 1)
//   [10..17]  input strides
//   [18..25]  reduced extents per dim (1 where not reduced)
//   [26]      number of reduced elements
//   [27]      opcode: 0 sum, 1 mean

const MAX_RANK : u32 = 8u;
const GROUP : u32 = 256u;

// One workgroup per output element, so the grid is indexed by workgroup and a
// second row is needed past the 65535 per-dimension limit.
const WG_STRIDE : u32 = 65535u;

@group(0) @binding(0) var<storage, read> dims : array<i32>;
@group(0) @binding(1) var<storage, read> src : array<f32>;
@group(0) @binding(2) var<storage, read_write> dst : array<f32>;

var<workgroup> partial : array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid : vec3<u32>,
        @builtin(local_invocation_id) lid : vec3<u32>) {
  let out_i = wid.x + wid.y * WG_STRIDE;
  let total = u32(dims[1]);
  if (out_i >= total) { return; }

  let rank = u32(dims[0]);

  // Base offset from the non-reduced coordinates.
  var rem = out_i;
  var base : i32 = 0;
  for (var k : u32 = 0u; k < rank; k = k + 1u) {
    let d = MAX_RANK - 1u - k;
    let extent = u32(dims[2 + i32(d)]);
    let coord = i32(rem % extent);
    rem = rem / extent;
    base = base + coord * dims[10 + i32(d)];
  }

  let count = u32(dims[26]);

  // Each thread accumulates a strided slice, compensated: a slice can still be
  // ~10^5 terms, and plain fp32 addition loses enough to show in the gradient.
  var acc : f32 = 0.0;
  var comp : f32 = 0.0;
  var j : u32 = lid.x;
  loop {
    if (j >= count) { break; }
    var r = j;
    var off = base;
    for (var k : u32 = 0u; k < rank; k = k + 1u) {
      let d = MAX_RANK - 1u - k;
      let red = u32(dims[18 + i32(d)]);
      if (red > 1u) {
        let coord = i32(r % red);
        r = r / red;
        off = off + coord * dims[10 + i32(d)];
      }
    }
    let y = src[u32(off)] - comp;
    let t = acc + y;
    comp = (t - acc) - y;
    acc = t;
    j = j + GROUP;
  }

  partial[lid.x] = acc;
  workgroupBarrier();

  // Tree reduction across the workgroup.
  var stride : u32 = GROUP / 2u;
  loop {
    if (stride == 0u) { break; }
    if (lid.x < stride) {
      partial[lid.x] = partial[lid.x] + partial[lid.x + stride];
    }
    workgroupBarrier();
    stride = stride / 2u;
  }

  if (lid.x == 0u) {
    var v = partial[0];
    if (dims[27] == 1 && count > 0u) { v = v / f32(count); }
    dst[out_i] = v;
  }
}
