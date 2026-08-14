/**
 * Chrome extension messaging does not reliably preserve ArrayBuffer across
 * service-worker ↔ offscreen boundaries (often arrives as a non-buffer).
 * Base64 is boring and correct for SW→offscreen.
 */

export function arrayBufferToBase64(bytes: ArrayBuffer): string {
  const view = new Uint8Array(bytes);
  const chunk = 0x8000;
  let binary = "";
  for (let i = 0; i < view.length; i += chunk) {
    binary += String.fromCharCode(...view.subarray(i, i + chunk));
  }
  return btoa(binary);
}

export function base64ToArrayBuffer(b64: string): ArrayBuffer {
  const binary = atob(b64);
  const out = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) {
    out[i] = binary.charCodeAt(i);
  }
  return out.buffer;
}

type BufferLike = {
  byteLength: number;
  slice: (start?: number, end?: number) => ArrayBuffer;
};

function isBufferLike(value: unknown): value is BufferLike {
  return (
    typeof value === "object" &&
    value !== null &&
    typeof (value as BufferLike).byteLength === "number" &&
    typeof (value as BufferLike).slice === "function"
  );
}

/** Accept ArrayBuffer / TypedArray / cross-realm buffers from structured-clone. */
export function coerceArrayBuffer(value: unknown): ArrayBuffer {
  if (value instanceof ArrayBuffer) return value.slice(0);
  // Cross-realm ArrayBuffer: instanceof fails across extension contexts.
  if (isBufferLike(value) && !ArrayBuffer.isView(value)) {
    return value.slice(0);
  }
  if (ArrayBuffer.isView(value)) {
    const view = value;
    const raw = view.buffer;
    const start = view.byteOffset;
    const end = view.byteOffset + view.byteLength;
    if (raw instanceof ArrayBuffer) return raw.slice(start, end);
    // SharedArrayBuffer / cross-realm backing store — copy via view.
    const out = new Uint8Array(view.byteLength);
    out.set(new Uint8Array(raw, start, view.byteLength));
    return out.buffer;
  }
  if (Array.isArray(value)) {
    return new Uint8Array(value as number[]).buffer;
  }
  // Chrome sometimes JSON-shapes a buffer as { "0": n, "1": n, ... }.
  if (value && typeof value === "object") {
    const rec = value as Record<string, unknown>;
    const keys = Object.keys(rec);
    if (
      keys.length > 16 &&
      keys.every((k) => /^\d+$/.test(k)) &&
      typeof rec["0"] === "number"
    ) {
      const out = new Uint8Array(keys.length);
      for (let i = 0; i < keys.length; i += 1) {
        out[i] = Number(rec[String(i)] ?? 0);
      }
      return out.buffer;
    }
  }
  throw new Error(
    `Cannot coerce image bytes from ${Object.prototype.toString.call(value)}`,
  );
}
