// Elementwise unary and binary ops with broadcasting.
//
// Both operands are read through strides, so a stride of 0 broadcasts and a
// rank-padded shape needs no materialisation. Scalar variants (add.Scalar,
// mul.Scalar, fill, comparisons) pass their constant in `scalars`.
//
// dims layout (i32):
//   [0] rank
//   [1] total output elements
//   [2 .. 9]  output shape (right-aligned to MAX_RANK)
//   [10..17]  a strides
//   [18..25]  b strides
//   [26]      opcode
//   [27]      1 if b is an actual tensor, 0 if the scalar operand is used

const MAX_RANK : u32 = 8u;

const OP_ADD : i32 = 0;
const OP_SUB : i32 = 1;
const OP_MUL : i32 = 2;
const OP_DIV : i32 = 3;
const OP_ABS : i32 = 4;
const OP_NEG : i32 = 5;
const OP_SIGMOID : i32 = 6;
const OP_SILU : i32 = 7;
const OP_GT : i32 = 8;
const OP_LT : i32 = 9;
const OP_FILL : i32 = 10;
const OP_COPY : i32 = 11;
const OP_RSUB : i32 = 12;

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
@group(0) @binding(4) var<storage, read_write> dst : array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x + gid.y * DISPATCH_STRIDE;
  let total = u32(dims[1]);
  if (i >= total) { return; }

  let rank = u32(dims[0]);
  var rem = i;
  var offA : i32 = 0;
  var offB : i32 = 0;
  for (var k : u32 = 0u; k < rank; k = k + 1u) {
    let d = MAX_RANK - 1u - k;
    let extent = u32(dims[2 + i32(d)]);
    let coord = i32(rem % extent);
    rem = rem / extent;
    offA = offA + coord * dims[10 + i32(d)];
    offB = offB + coord * dims[18 + i32(d)];
  }

  let op = dims[26];
  let x = a[u32(offA)];
  var y : f32 = scalars[0];
  if (dims[27] == 1) { y = b[u32(offB)]; }

  var r : f32 = 0.0;
  switch (op) {
    case 0:  { r = x + y; }
    case 1:  { r = x - y; }
    case 2:  { r = x * y; }
    case 3:  { r = x / y; }
    case 4:  { r = abs(x); }
    case 5:  { r = -x; }
    case 6:  { r = 1.0 / (1.0 + exp(-x)); }
    case 7:  { r = x / (1.0 + exp(-x)); }          // silu
    case 8:  { r = select(0.0, 1.0, x > y); }
    case 9:  { r = select(0.0, 1.0, x < y); }
    case 10: { r = y; }                            // fill
    case 11: { r = x; }                            // copy
    case 12: { r = y - x; }                        // reverse subtract
    default: { r = x; }
  }
  dst[i] = r;
}
