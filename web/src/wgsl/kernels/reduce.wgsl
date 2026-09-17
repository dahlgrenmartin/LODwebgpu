// Sum / mean over an arbitrary set of dimensions, keepdim semantics.
//
// One thread per output element; it walks the cartesian product of the reduced
// extents. The reduced dims keep extent 1 in the output shape, so the output is
// always contiguous and downstream ops need no special case.
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
  let i = gid.x + gid.y * DISPATCH_STRIDE;
  let total = u32(dims[1]);
  if (i >= total) { return; }

  let rank = u32(dims[0]);

  // Base offset from the non-reduced coordinates.
  var rem = i;
  var base : i32 = 0;
  for (var k : u32 = 0u; k < rank; k = k + 1u) {
    let d = MAX_RANK - 1u - k;
    let extent = u32(dims[2 + i32(d)]);
    let coord = i32(rem % extent);
    rem = rem / extent;
    base = base + coord * dims[10 + i32(d)];
  }

  let count = u32(dims[26]);
  // Kahan compensated summation: one thread accumulates the whole reduction,
  // and plain fp32 addition over ~10^5-10^6 terms loses enough precision to
  // show up in the gradient.
  var acc : f32 = 0.0;
  var comp : f32 = 0.0;
  for (var j : u32 = 0u; j < count; j = j + 1u) {
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
  }

  if (dims[27] == 1 && count > 0u) {
    acc = acc / f32(count);
  }
  dst[i] = acc;
}
