// Renders an NCHW fp32 decoder output straight from its GPU buffer.
// No readback: the buffer ORT wrote is bound read-only to the fragment stage.
struct Dims { w: u32, h: u32, pad0: u32, pad1: u32 };
@group(0) @binding(0) var<uniform> dims : Dims;
@group(0) @binding(1) var<storage, read> img : array<f32>;

@vertex
fn vs(@builtin(vertex_index) i : u32) -> @builtin(position) vec4<f32> {
  // Fullscreen triangle; no vertex buffer.
  var p = array<vec2<f32>, 3>(
    vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0));
  return vec4<f32>(p[i], 0.0, 1.0);
}

@fragment
fn fs(@builtin(position) pos : vec4<f32>) -> @location(0) vec4<f32> {
  let x = u32(pos.x);
  let y = u32(pos.y);
  if (x >= dims.w || y >= dims.h) {
    return vec4<f32>(0.0, 0.0, 0.0, 1.0);
  }
  let n = dims.w * dims.h;
  let i = y * dims.w + x;
  // Decoder emits roughly [-1, 1]; map to display range.
  let rgb = vec3<f32>(img[i], img[n + i], img[2u * n + i]) * 0.5 + 0.5;
  return vec4<f32>(clamp(rgb, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
}
