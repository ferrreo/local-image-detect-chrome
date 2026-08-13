export type DecodedImage = {
  bitmap: ImageBitmap;
  width: number;
  height: number;
};

export async function decodeImageBytes(
  bytes: ArrayBuffer,
  mimeType = "image/jpeg",
): Promise<DecodedImage> {
  const blob = new Blob([bytes], { type: mimeType });
  const bitmap = await createImageBitmap(blob);
  return {
    bitmap,
    width: bitmap.width,
    height: bitmap.height,
  };
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

export async function rasterizeForSpectral(
  bitmap: ImageBitmap,
  maxSide = 256,
): Promise<ImageData> {
  const scale = Math.min(1, maxSide / Math.max(bitmap.width, bitmap.height));
  const width = Math.max(32, Math.round(bitmap.width * scale));
  const height = Math.max(32, Math.round(bitmap.height * scale));
  const canvas = new OffscreenCanvas(width, height);
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  if (!ctx) {
    throw new Error("OffscreenCanvas 2d context unavailable");
  }
  ctx.drawImage(bitmap, 0, 0, width, height);
  return ctx.getImageData(0, 0, width, height);
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
