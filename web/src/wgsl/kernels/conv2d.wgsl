// aten::convolution, forward and transposed, with groups/stride/padding/dilation.
//
// Output-stationary: one thread per output element, accumulating over the input
// channels of its group and the kernel window. The transposed case is the same
// traversal with the stride relation inverted, which is why the lowered backward
// pass needs no separate kernel.
//
// Weight layout follows aten:
//   forward    [Cout, Cin/groups, KH, KW]
//   transposed [Cin,  Cout/groups, KH, KW]
//
// dims layout (i32):
//   [0] N   [1] Cin  [2] Hin  [3] Win
//   [4] Cout [5] Hout [6] Wout
//   [7] KH  [8] KW
//   [9] sh  [10] sw  [11] ph  [12] pw  [13] dh  [14] dw
//   [15] groups  [16] transposed  [17] hasBias  [18] total

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
@group(0) @binding(2) var<storage, read> wgt : array<f32>;
@group(0) @binding(3) var<storage, read> bias : array<f32>;
@group(0) @binding(4) var<storage, read_write> dst : array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let idx = i32(gid.x + gid.y * DISPATCH_STRIDE);
  if (idx >= dims[18]) { return; }

  let Cin = dims[1];  let Hin = dims[2];  let Win = dims[3];
  let Cout = dims[4]; let Hout = dims[5]; let Wout = dims[6];
  let KH = dims[7];   let KW = dims[8];
  let sh = dims[9];   let sw = dims[10];
  let ph = dims[11];  let pw = dims[12];
  let dh = dims[13];  let dw = dims[14];
  let groups = dims[15];
  let transposed = dims[16];

  let ow = idx % Wout;
  let oh = (idx / Wout) % Hout;
  let co = (idx / (Wout * Hout)) % Cout;
  let n  = idx / (Wout * Hout * Cout);

  let cinPerG = Cin / groups;
  let coutPerG = Cout / groups;
  let g = co / coutPerG;
  let coLocal = co % coutPerG;

  var acc : f32 = 0.0;

  if (transposed == 0) {
    for (var kh = 0; kh < KH; kh = kh + 1) {
      let ih = oh * sh - ph + kh * dh;
      if (ih < 0 || ih >= Hin) { continue; }
      for (var kw = 0; kw < KW; kw = kw + 1) {
        let iw = ow * sw - pw + kw * dw;
        if (iw < 0 || iw >= Win) { continue; }
        for (var c = 0; c < cinPerG; c = c + 1) {
          let ci = g * cinPerG + c;
          let si = ((n * Cin + ci) * Hin + ih) * Win + iw;
          let wi = ((co * cinPerG + c) * KH + kh) * KW + kw;
          acc = acc + src[si] * wgt[wi];
        }
      }
    }
  } else {
    // Transposed: an input pixel contributes to oh when oh + ph - kh*dh == ih*sh.
    for (var kh = 0; kh < KH; kh = kh + 1) {
      let th = oh + ph - kh * dh;
      if (th < 0 || th % sh != 0) { continue; }
      let ih = th / sh;
      if (ih >= Hin) { continue; }
      for (var kw = 0; kw < KW; kw = kw + 1) {
        let tw = ow + pw - kw * dw;
        if (tw < 0 || tw % sw != 0) { continue; }
        let iw = tw / sw;
        if (iw >= Win) { continue; }
        for (var c = 0; c < cinPerG; c = c + 1) {
          let ci = g * cinPerG + c;
          let si = ((n * Cin + ci) * Hin + ih) * Win + iw;
          let wi = ((ci * coutPerG + coLocal) * KH + kh) * KW + kw;
          acc = acc + src[si] * wgt[wi];
        }
      }
    }
  }

  if (dims[17] == 1) { acc = acc + bias[co]; }
  dst[idx] = acc;
}
