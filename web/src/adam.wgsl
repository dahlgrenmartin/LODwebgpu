// NOTE: `meta` is a reserved keyword in WGSL - naming this struct binding `meta`
// makes the module fail to compile, and an unchecked pipeline then silently
// does nothing. The Adam constructor asserts on compilation info for that reason.
struct Params { n: u32, lr: f32, beta1: f32, beta2: f32, eps: f32,
                bc1: f32, bc2: f32, pad: u32 };
@group(0) @binding(0) var<uniform> params : Params;
@group(0) @binding(1) var<storage, read_write> z : array<f32>;
@group(0) @binding(2) var<storage, read> grad : array<f32>;
@group(0) @binding(3) var<storage, read_write> m : array<f32>;
@group(0) @binding(4) var<storage, read_write> v : array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  if (i >= params.n) { return; }
  let g = grad[i];
  let mi = params.beta1 * m[i] + (1.0 - params.beta1) * g;
  let vi = params.beta2 * v[i] + (1.0 - params.beta2) * g * g;
  m[i] = mi;
  v[i] = vi;
  let mhat = mi / params.bc1;
  let vhat = vi / params.bc2;
  z[i] = z[i] - params.lr * mhat / (sqrt(vhat) + params.eps);
}
