export type DecodedImage = {
  bitmap: ImageBitmap;
  width: number;
  height: number;
};

export async function decodeImageBytes(
  bytes: ArrayBuffer,
  mimeType = "image/jpeg",
): Promise<DecodedImage> {
  if (bytes.byteLength === 0) {
    throw new Error("The source image could not be decoded. (empty buffer)");
  }
  // Copy first — callers may pass a buffer that structured-clone will detach.
  const copy = bytes.slice(0);
  const view = new Uint8Array(copy);
  const type = mimeType.startsWith("image/")
    ? mimeType
    : guessMimeType(view);

  // ImageDecoder handles progressive JPEG reliably in workers/offscreen.
  if (typeof ImageDecoder !== "undefined") {
    try {
      const decoder = new ImageDecoder({ data: view, type });
      const { image } = await decoder.decode({ frameIndex: 0 });
      try {
        const bitmap = await createImageBitmap(image);
        return {
          bitmap,
          width: bitmap.width,
          height: bitmap.height,
        };
      } finally {
        image.close();
        decoder.close();
      }
    } catch {
      // fall through to createImageBitmap(Blob)
    }
  }

  const blob = new Blob([copy], { type });
  try {
    const bitmap = await createImageBitmap(blob);
    return {
      bitmap,
      width: bitmap.width,
      height: bitmap.height,
    };
  } catch (error) {
    const detail = error instanceof Error ? error.message : String(error);
    throw new Error(
      `The source image could not be decoded. (${type}, ${copy.byteLength} bytes: ${detail})`,
    );
  }
}

export async function rasterizeForModel(
  bitmap: ImageBitmap,
  size: number,
): Promise<ImageData> {
  const canvas = new OffscreenCanvas(size, size);
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  if (!ctx) {
    throw new Error("OffscreenCanvas 2d context unavailable");
  }

  // Match ViTFeatureExtractor: stretch-resize to HxW (no center-crop).
  ctx.drawImage(bitmap, 0, 0, size, size);
  return ctx.getImageData(0, 0, size, size);
}

/**
 * Area-average downsample so spectral features stay stable across browsers
 * and the Node canvas polyfill (nearest-neighbor drawImage skews Laplacian).
 */
export function boxDownsampleImageData(
  source: ImageData,
  maxSide = 256,
): ImageData {
  const scale = Math.min(1, maxSide / Math.max(source.width, source.height));
  const width = Math.max(32, Math.round(source.width * scale));
  const height = Math.max(32, Math.round(source.height * scale));
  if (width === source.width && height === source.height) {
    return source;
  }

  const out = new Uint8ClampedArray(width * height * 4);
  for (let y = 0; y < height; y += 1) {
    const y0 = Math.floor((y * source.height) / height);
    const y1 = Math.max(y0 + 1, Math.floor(((y + 1) * source.height) / height));
    for (let x = 0; x < width; x += 1) {
      const x0 = Math.floor((x * source.width) / width);
      const x1 = Math.max(x0 + 1, Math.floor(((x + 1) * source.width) / width));
      let r = 0;
      let g = 0;
      let b = 0;
      let n = 0;
      for (let sy = y0; sy < y1; sy += 1) {
        for (let sx = x0; sx < x1; sx += 1) {
          const si = (sy * source.width + sx) * 4;
          r += source.data[si] ?? 0;
          g += source.data[si + 1] ?? 0;
          b += source.data[si + 2] ?? 0;
          n += 1;
        }
      }
      const di = (y * width + x) * 4;
      out[di] = n === 0 ? 0 : Math.round(r / n);
      out[di + 1] = n === 0 ? 0 : Math.round(g / n);
      out[di + 2] = n === 0 ? 0 : Math.round(b / n);
      out[di + 3] = 255;
    }
  }
  return { data: out, width, height, colorSpace: "srgb" } as ImageData;
}

export async function rasterizeForSpectral(
  bitmap: ImageBitmap,
  maxSide = 256,
): Promise<ImageData> {
  // Prefer direct pixel access when available (Node test polyfill) so we can
  // box-downsample without nearest-neighbor drawImage artifacts.
  const readable = bitmap as ImageBitmap & { data?: Uint8ClampedArray };
  if (
    readable.data &&
    readable.data.byteLength >= bitmap.width * bitmap.height * 4
  ) {
    return boxDownsampleImageData(
      {
        data: readable.data,
        width: bitmap.width,
        height: bitmap.height,
        colorSpace: "srgb",
      } as ImageData,
      maxSide,
    );
  }

  // Browser ImageBitmap: read native pixels then box-downsample.
  // Do NOT pre-shrink (old 768 cap depressed laplacianVariance and skipped
  // Community Forensics on Riverflow cases → 97.4% BA vs host 100%).
  const maxNative = 4096;
  const nativeScale = Math.min(
    1,
    maxNative / Math.max(bitmap.width, bitmap.height),
  );
  const readW = Math.max(32, Math.round(bitmap.width * nativeScale));
  const readH = Math.max(32, Math.round(bitmap.height * nativeScale));
  const canvas = new OffscreenCanvas(readW, readH);
  const ctx = canvas.getContext("2d", {
    willReadFrequently: true,
    colorSpace: "srgb",
  });
  if (!ctx) {
    throw new Error("OffscreenCanvas 2d context unavailable");
  }
  ctx.drawImage(bitmap, 0, 0, readW, readH);
  const mid = ctx.getImageData(0, 0, readW, readH);
  return boxDownsampleImageData(mid, maxSide);
}

/** ImageNet-style NCHW float32 tensor for ViT classifiers. */
export function imageDataToNchwFloat32(
  imageData: ImageData,
  mean: readonly [number, number, number] = [0.5, 0.5, 0.5],
  std: readonly [number, number, number] = [0.5, 0.5, 0.5],
): Float32Array {
  const { data, width, height } = imageData;
  const plane = width * height;
  const out = new Float32Array(3 * plane);
  for (let i = 0, p = 0; i < plane; i += 1, p += 4) {
    const r = ((data[p] ?? 0) / 255 - mean[0]) / std[0];
    const g = ((data[p + 1] ?? 0) / 255 - mean[1]) / std[1];
    const b = ((data[p + 2] ?? 0) / 255 - mean[2]) / std[2];
    out[i] = r;
    out[plane + i] = g;
    out[2 * plane + i] = b;
  }
  return out;
}

export function guessMimeType(bytes: Uint8Array): string {
  if (bytes.length >= 3 && bytes[0] === 0xff && bytes[1] === 0xd8 && bytes[2] === 0xff) {
    return "image/jpeg";
  }
  if (
    bytes.length >= 8 &&
    bytes[0] === 0x89 &&
    bytes[1] === 0x50 &&
    bytes[2] === 0x4e &&
    bytes[3] === 0x47
  ) {
    return "image/png";
  }
  if (
    bytes.length >= 12 &&
    bytes[0] === 0x52 &&
    bytes[1] === 0x49 &&
    bytes[2] === 0x46 &&
    bytes[3] === 0x46 &&
    bytes[8] === 0x57 &&
    bytes[9] === 0x45 &&
    bytes[10] === 0x42 &&
    bytes[11] === 0x50
  ) {
    return "image/webp";
  }
  if (
    bytes.length >= 6 &&
    bytes[0] === 0x47 &&
    bytes[1] === 0x49 &&
    bytes[2] === 0x46
  ) {
    return "image/gif";
  }
  return "application/octet-stream";
}
