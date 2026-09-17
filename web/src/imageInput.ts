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
 * Decode a user-supplied image, which must already be exactly `size` x `size`.
 *
 * Nothing is resampled and nothing is cropped: the pixels reach the VAE as given.
 * `imageOrientation: 'from-image'` applies EXIF rotation, which phone photos rely
 * on; without it portrait shots arrive sideways.  Alpha is dropped.
 */
export async function prepareImage(
  source: Blob,
  size: number,
): Promise<PreparedImage> {
  let bitmap: ImageBitmap;
  try {
    bitmap = await createImageBitmap(source, { imageOrientation: 'from-image' });
  } catch (e) {
    throw new Error(`could not decode that file as an image (${(e as Error).message})`);
  }

  // The image must reach the VAE byte-for-byte. Resizing rewrites the
  // high-frequency detail the wavelet detector reads; cropping removes evidence
  // from the frame. Neither is acceptable for detection, so an image that is not
  // exactly the model's size is refused rather than altered.
  if (bitmap.width !== size || bitmap.height !== size) {
    const w = bitmap.width;
    const h = bitmap.height;
    bitmap.close();
    throw new Error(
      `This build accepts exactly ${size}x${size} images; yours is ${w}x${h}. ` +
      `It is not resized or cropped, because either would alter the detail the ` +
      `detector measures. Arbitrary sizes need the shape-agnostic runtime.`);
  }

  const canvas = document.createElement('canvas');
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  if (!ctx) throw new Error('2D canvas context unavailable');
  ctx.drawImage(bitmap, 0, 0);   // 1:1, no scaling, no offset
  const sourceSize = { w: bitmap.width, h: bitmap.height };
  const sx = 0;
  const sy = 0;
  bitmap.close();

  const preview = ctx.getImageData(0, 0, size, size);
  const px = preview.data;
  const n = size * size;
  const data = new Float32Array(3 * n);
  for (let i = 0; i < n; i++) {
    data[i] = (px[i * 4] / 255) * 2 - 1;              // R
    data[n + i] = (px[i * 4 + 1] / 255) * 2 - 1;      // G
    data[2 * n + i] = (px[i * 4 + 2] / 255) * 2 - 1;  // B
  }
  return { data, shape: [1, 3, size, size], preview, sourceSize, crop: { x: sx, y: sy } };
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
