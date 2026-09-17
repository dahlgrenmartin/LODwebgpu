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
//
// Each thread owns a 2x2 block of that patch rather than a single pixel. With
// one pixel per thread every weight the thread loads is used exactly once, so
// the inner loop spends more instructions fetching weights than multiplying:
// 36 weight loads against 36 MACs. A 2x2 block reuses each weight four times
// and needs four input rows instead of three, which lifts the ratio of MACs to
// loads from 0.8 to 2.8.
//
// The transposed case is the same traversal: for stride 1, pad 1 and a 3x3
// kernel, grad_in[ci] = sum over co,kh,kw of grad_out[co, ih+1-kh, iw+1-kw] *
// w[co,ci,kh,kw], which is this convolution with the window flipped and the
// weight's channel roles swapped. Only the weight index differs, and the branch
// is uniform across the workgroup.
//
// dims layout (i32):
//   [0] N  [1] Cin  [2] H  [3] W  [4] Cout  [5] hasBias  [6] transposed

const TILE : u32 = 16u;                // output patch edge
const HALO : u32 = TILE + 2u;          // 3x3 with pad 1 needs one element either side
const CH_CHUNK : u32 = 8u;             // 8 * 18 * 18 * 4B = 10.4 KB, within the 16 KB floor
const COUT_BLOCK : u32 = 4u;           // output channels per workgroup
const THREADS : u32 = 64u;             // 8x8 threads, each owning a 2x2 block

@group(0) @binding(0) var<storage, read> dims : array<i32>;
@group(0) @binding(1) var<storage, read> src : array<f32>;
@group(0) @binding(2) var<storage, read> wgt : array<f32>;
@group(0) @binding(3) var<storage, read> bias : array<f32>;
@group(0) @binding(4) var<storage, read_write> dst : array<f32>;

var<workgroup> tile : array<f32, 2592>;   // CH_CHUNK * HALO * HALO

@compute @workgroup_size(8, 8)
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
  let flat = lid.y * 8u + lid.x;            // 0..63
  let tx = lid.x * 2u;                      // this thread's corner within the patch
  let ty = lid.y * 2u;

  // acc[b * 4 + dy * 2 + dx]
  var acc = array<f32, 16>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                           0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);

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
      let ry = rem / HALO;
      let rx = rem % HALO;
      let ih = i32(oh0 + ry) - 1;
      let iw = i32(ow0 + rx) - 1;
      var v : f32 = 0.0;
      if (c0 + ci < Cin && ih >= 0 && iw >= 0 && u32(ih) < H && u32(iw) < W) {
        v = src[((n * Cin + c0 + ci) * H + u32(ih)) * W + u32(iw)];
      }
      tile[idx] = v;
      idx = idx + THREADS;
    }
    workgroupBarrier();

    var ci : u32 = 0u;
    loop {
      if (ci >= CH_CHUNK || c0 + ci >= Cin) { break; }
      let tbase = ci * HALO * HALO;
      let cin_i = c0 + ci;

      // Four input rows cover both output rows: row kh serves output row 0 and
      // row kh+1 serves output row 1, so consecutive kh share three of them.
      var r0 = array<f32, 4>(0.0, 0.0, 0.0, 0.0);
      var r1 = array<f32, 4>(0.0, 0.0, 0.0, 0.0);
      var r2 = array<f32, 4>(0.0, 0.0, 0.0, 0.0);
      var r3 = array<f32, 4>(0.0, 0.0, 0.0, 0.0);
      for (var j : u32 = 0u; j < 4u; j = j + 1u) {
        let base = tbase + ty * HALO + tx + j;
        r0[j] = tile[base];
        r1[j] = tile[base + HALO];
        r2[j] = tile[base + 2u * HALO];
        r3[j] = tile[base + 3u * HALO];
      }

      // Hoist the layout branch: it is uniform for the whole dispatch, and
      // testing it inside the kh/block loops evaluated it 12x per channel.
      if (dims[6] == 0) {
        // forward: weight is [Cout, Cin, 3, 3]
        for (var b : u32 = 0u; b < COUT_BLOCK; b = b + 1u) {
          let wb = ((co0 + b) * Cin + cin_i) * 9u;
          for (var kh : u32 = 0u; kh < 3u; kh = kh + 1u) {
            let w0 = wgt[wb + kh * 3u];
            let w1 = wgt[wb + kh * 3u + 1u];
            let w2 = wgt[wb + kh * 3u + 2u];
            var a0 : array<f32, 4>;
            var a1 : array<f32, 4>;
            if (kh == 0u) { a0 = r0; a1 = r1; }
            else if (kh == 1u) { a0 = r1; a1 = r2; }
            else { a0 = r2; a1 = r3; }
            acc[b * 4u + 0u] = acc[b * 4u + 0u] + a0[0] * w0 + a0[1] * w1 + a0[2] * w2;
            acc[b * 4u + 1u] = acc[b * 4u + 1u] + a0[1] * w0 + a0[2] * w1 + a0[3] * w2;
            acc[b * 4u + 2u] = acc[b * 4u + 2u] + a1[0] * w0 + a1[1] * w1 + a1[2] * w2;
            acc[b * 4u + 3u] = acc[b * 4u + 3u] + a1[1] * w0 + a1[2] * w1 + a1[3] * w2;
          }
        }
      } else {
        // transposed: weight is [Cin, Cout, 3, 3], window flipped
        for (var b : u32 = 0u; b < COUT_BLOCK; b = b + 1u) {
          let wb = (cin_i * Cout + co0 + b) * 9u;
          for (var kh : u32 = 0u; kh < 3u; kh = kh + 1u) {
            let base = wb + (2u - kh) * 3u;
            let w0 = wgt[base + 2u];
            let w1 = wgt[base + 1u];
            let w2 = wgt[base];
            var a0 : array<f32, 4>;
            var a1 : array<f32, 4>;
            if (kh == 0u) { a0 = r0; a1 = r1; }
            else if (kh == 1u) { a0 = r1; a1 = r2; }
            else { a0 = r2; a1 = r3; }
            acc[b * 4u + 0u] = acc[b * 4u + 0u] + a0[0] * w0 + a0[1] * w1 + a0[2] * w2;
            acc[b * 4u + 1u] = acc[b * 4u + 1u] + a0[1] * w0 + a0[2] * w1 + a0[3] * w2;
            acc[b * 4u + 2u] = acc[b * 4u + 2u] + a1[0] * w0 + a1[1] * w1 + a1[2] * w2;
            acc[b * 4u + 3u] = acc[b * 4u + 3u] + a1[1] * w0 + a1[2] * w1 + a1[3] * w2;
          }
        }
      }
      ci = ci + 1u;
    }
    c0 = c0 + CH_CHUNK;
  }

  for (var b : u32 = 0u; b < COUT_BLOCK; b = b + 1u) {
    let co = co0 + b;
    if (co >= Cout) { continue; }
    for (var dy : u32 = 0u; dy < 2u; dy = dy + 1u) {
      let oh = oh0 + ty + dy;
      if (oh >= H) { continue; }
      for (var dx : u32 = 0u; dx < 2u; dx = dx + 1u) {
        let ow = ow0 + tx + dx;
        if (ow >= W) { continue; }
        var v = acc[b * 4u + dy * 2u + dx];
        if (dims[5] == 1) { v = v + bias[co]; }
        dst[((n * Cout + co) * H + oh) * W + ow] = v;
      }
    }
  }
}
