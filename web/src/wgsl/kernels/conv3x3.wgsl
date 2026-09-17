// Tiled 3x3 / stride-1 / pad-1 / groups-1 convolution.
//
// The general conv2d kernel is an output-stationary gather: every multiply-add
// issues two global loads and reuses nothing, which measures ~2-3% of the card's
// fp32 peak. Here a workgroup cooperatively stages a halo'd input tile in
// workgroup memory, so each input element crosses the memory bus once instead of
// nine times, and the 3x3 window is read from shared memory.
//
// One workgroup produces a TILE x TILE patch of COUT_BLOCK output channels:
//   dispatch(ceil(W/TILE), ceil(H/TILE), N*ceil(Cout/COUT_BLOCK))
// Blocking over output channels is what makes the staging pay: the tile is read
// from workgroup memory once and reused for every channel in the block, so its
// global-memory cost is amortised COUT_BLOCK ways.
//
// The transposed case is the same traversal: for stride 1, pad 1 and a 3x3
// kernel, grad_in[ci] = sum over co,kh,kw of grad_out[co, ih+1-kh, iw+1-kw] *
// w[co,ci,kh,kw], which is this convolution with the window flipped and the
// weight's channel roles swapped. Only the weight index differs, and the branch
// is uniform across the workgroup.
//
// dims layout (i32):
//   [0] N  [1] Cin  [2] H  [3] W  [4] Cout  [5] hasBias  [6] transposed

const TILE : u32 = 16u;
const HALO : u32 = TILE + 2u;          // 3x3 with pad 1 needs one element either side
const CH_CHUNK : u32 = 8u;             // 8 * 18 * 18 * 4B = 10.4 KB, within the 16 KB floor
const COUT_BLOCK : u32 = 4u;           // output channels per workgroup

@group(0) @binding(0) var<storage, read> dims : array<i32>;
@group(0) @binding(1) var<storage, read> src : array<f32>;
@group(0) @binding(2) var<storage, read> wgt : array<f32>;
@group(0) @binding(3) var<storage, read> bias : array<f32>;
@group(0) @binding(4) var<storage, read_write> dst : array<f32>;

var<workgroup> tile : array<f32, 2592>;   // CH_CHUNK * HALO * HALO

@compute @workgroup_size(16, 16)
fn main(@builtin(workgroup_id) wid : vec3<u32>,
        @builtin(local_invocation_id) lid : vec3<u32>) {
  let N = u32(dims[0]);
  let Cin = u32(dims[1]);
  let H = u32(dims[2]);
  let W = u32(dims[3]);
  let Cout = u32(dims[4]);

  // Cout is a multiple of COUT_BLOCK for every decoder convolution, so the
  // block loop needs no per-channel bounds test; the store below still guards.
  let blocks = (Cout + COUT_BLOCK - 1u) / COUT_BLOCK;
  let co0 = (wid.z % blocks) * COUT_BLOCK;
  let n = wid.z / blocks;
  let oh0 = wid.y * TILE;
  let ow0 = wid.x * TILE;
  let flat = lid.y * TILE + lid.x;          // 0..255

  var acc = array<f32, 4>(0.0, 0.0, 0.0, 0.0);

  var c0 : u32 = 0u;
  loop {
    if (c0 >= Cin) { break; }

    // Cooperative load: HALO x HALO for CH_CHUNK input channels.
    workgroupBarrier();
    var idx : u32 = flat;
    loop {
      if (idx >= CH_CHUNK * HALO * HALO) { break; }
      let ci = idx / (HALO * HALO);
      let rem = idx % (HALO * HALO);
      let ty = rem / HALO;
      let tx = rem % HALO;
      let ih = i32(oh0 + ty) - 1;
      let iw = i32(ow0 + tx) - 1;
      var v : f32 = 0.0;
      if (c0 + ci < Cin && ih >= 0 && iw >= 0 && u32(ih) < H && u32(iw) < W) {
        v = src[((n * Cin + c0 + ci) * H + u32(ih)) * W + u32(iw)];
      }
      tile[idx] = v;
      idx = idx + 256u;
    }
    workgroupBarrier();

    var ci : u32 = 0u;
    loop {
      if (ci >= CH_CHUNK || c0 + ci >= Cin) { break; }
      let tbase = ci * HALO * HALO;
      let cin_i = c0 + ci;
      // Hoist the layout branch: it is uniform for the whole dispatch, and
      // testing it inside the kh/block loops evaluated it 12x per channel.
      if (dims[6] == 0) {
        // forward: weight is [Cout, Cin, 3, 3]
        for (var kh : u32 = 0u; kh < 3u; kh = kh + 1u) {
          let row = tbase + (lid.y + kh) * HALO + lid.x;
          let t0 = tile[row];
          let t1 = tile[row + 1u];
          let t2 = tile[row + 2u];
          for (var b : u32 = 0u; b < COUT_BLOCK; b = b + 1u) {
            let w = (((co0 + b) * Cin + cin_i) * 3u + kh) * 3u;
            acc[b] = acc[b] + t0 * wgt[w] + t1 * wgt[w + 1u] + t2 * wgt[w + 2u];
          }
        }
      } else {
        // transposed: weight is [Cin, Cout, 3, 3], window flipped
        for (var kh : u32 = 0u; kh < 3u; kh = kh + 1u) {
          let row = tbase + (lid.y + kh) * HALO + lid.x;
          let t0 = tile[row];
          let t1 = tile[row + 1u];
          let t2 = tile[row + 2u];
          for (var b : u32 = 0u; b < COUT_BLOCK; b = b + 1u) {
            let w = ((cin_i * Cout + co0 + b) * 3u + (2u - kh)) * 3u;
            acc[b] = acc[b] + t0 * wgt[w + 2u] + t1 * wgt[w + 1u] + t2 * wgt[w];
          }
        }
      }
      ci = ci + 1u;
    }
    c0 = c0 + CH_CHUNK;
  }

  let oh = oh0 + lid.y;
  let ow = ow0 + lid.x;
  if (oh < H && ow < W) {
    for (var b : u32 = 0u; b < COUT_BLOCK; b = b + 1u) {
      let co = co0 + b;
      if (co >= Cout) { continue; }
      var v = acc[b];
      if (dims[5] == 1) { v = v + bias[co]; }
      dst[((n * Cout + co) * H + oh) * W + ow] = v;
    }
  }
}
