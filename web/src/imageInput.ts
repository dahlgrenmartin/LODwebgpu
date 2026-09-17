export interface PreparedImage {
  /** NCHW fp32 in [-1, 1], shape [1, 3, size, size]. */
  data: Float32Array;
  shape: number[];
  /** sRGB bytes for the "before" preview. */
  preview: ImageData;
  /** Dimensions of the decoded source, before cropping. */
  sourceSize: { w: number; h: number };
  /** Top-left of the native-resolution crop within the source. */
  crop: { x: number; y: number };
}

/**
 * Decode a user-supplied image and hand its pixels to the VAE unaltered.
 *
 * Nothing is resampled and nothing is cropped. What varies between backends is
 * only which sizes are *accepted*, never what happens to the pixels.
 * `imageOrientation: 'from-image'` applies EXIF rotation, which phone photos rely
 * on; without it portrait shots arrive sideways.  Alpha is dropped.
 */
export interface SizePolicy {
  /** Require exactly this square size (the fixed-shape ONNX backend). */
  exact?: number | null;
  /** Otherwise accept the native size if both sides divide by this. */
  multipleOf?: number;
}

export async function prepareImage(
  source: Blob,
  policy: SizePolicy,
): Promise<PreparedImage> {
  let bitmap: ImageBitmap;
  try {
    bitmap = await createImageBitmap(source, { imageOrientation: 'from-image' });
  } catch (e) {
    throw new Error(`could not decode that file as an image (${(e as Error).message})`);
  }

  const w = bitmap.width;
  const h = bitmap.height;
  // The image must reach the VAE byte-for-byte. Resizing rewrites the
  // high-frequency detail the wavelet detector reads; cropping removes evidence
  // from the frame. Neither is acceptable, so anything that does not fit is
  // refused rather than altered.
  if (policy.exact != null) {
    if (w !== policy.exact || h !== policy.exact) {
      bitmap.close();
      throw new Error(
        `The fixed-shape backend accepts exactly ${policy.exact}x${policy.exact}; ` +
        `yours is ${w}x${h}. Switch to the WGSL backend for native sizes.`);
    }
  } else {
    const m = policy.multipleOf ?? 8;
    if (w % m !== 0 || h % m !== 0) {
      bitmap.close();
      throw new Error(
        `Image is ${w}x${h}. The decoder downsamples by ${m}, so both sides must ` +
        `be a multiple of ${m}. It is not cropped or resized to fit, because ` +
        `either would alter the detail the detector measures.`);
    }
  }

  const size = { w, h };
  const canvas = document.createElement('canvas');
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  if (!ctx) throw new Error('2D canvas context unavailable');
  ctx.drawImage(bitmap, 0, 0);   // 1:1, no scaling, no offset
  const sourceSize = { w, h };
  bitmap.close();

  const preview = ctx.getImageData(0, 0, size.w, size.h);
  const px = preview.data;
  const n = size.w * size.h;
  const data = new Float32Array(3 * n);
  for (let i = 0; i < n; i++) {
    data[i] = (px[i * 4] / 255) * 2 - 1;              // R
    data[n + i] = (px[i * 4 + 1] / 255) * 2 - 1;      // G
    data[2 * n + i] = (px[i * 4 + 2] / 255) * 2 - 1;  // B
  }
  return { data, shape: [1, 3, size.h, size.w], preview, sourceSize, crop: { x: 0, y: 0 } };
}

/**
 * A deterministic stand-in image, used when no file has been chosen so the demo
 * is usable without an upload and so the pipeline is testable headlessly.
 */
export function sampleImage(size: number): Blob {
  const canvas = document.createElement('canvas');
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext('2d')!;
  const g = ctx.createLinearGradient(0, 0, size, size);
  g.addColorStop(0, '#1f3b73');
  g.addColorStop(0.5, '#c94f7c');
  g.addColorStop(1, '#f2c14e');
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, size, size);
  ctx.fillStyle = 'rgba(255,255,255,0.85)';
  for (let i = 0; i < 6; i++) {
    ctx.beginPath();
    ctx.arc(size * (0.2 + 0.12 * i), size * (0.3 + 0.08 * (i % 3)),
            size * 0.06, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.strokeStyle = 'rgba(0,0,0,0.6)';
  ctx.lineWidth = Math.max(2, size / 64);
  ctx.strokeRect(size * 0.1, size * 0.1, size * 0.8, size * 0.8);

  const url = canvas.toDataURL('image/png');
  const bin = atob(url.split(',')[1]);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return new Blob([bytes], { type: 'image/png' });
}
