import {
  ALL_MODELS,
  MODEL_CACHE_NAME,
  MODEL_VERSION,
  type ModelArtifact,
} from "./model-manifest";
import type { ModelStatus } from "../shared/types";

/**
 * Cache Storage only accepts http(s) request schemes. Relative keys resolve to
 * chrome-extension:// in extension pages/SW and fail with
 * "Request scheme 'chrome-extension' is unsupported".
 */
function cacheUrl(model: ModelArtifact): string {
  return `https://truepixel.local/${model.cacheKey}`;
}

async function sha256Hex(buffer: ArrayBuffer): Promise<string> {
  const digest = await crypto.subtle.digest("SHA-256", buffer);
  return [...new Uint8Array(digest)]
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("");
}

export async function getModelStatus(): Promise<ModelStatus> {
  if (typeof caches === "undefined") {
    return { kind: "error", message: "Cache Storage unavailable" };
  }

  const cache = await caches.open(MODEL_CACHE_NAME);
  let total = 0;
  for (const model of ALL_MODELS) {
    const match =
      (await cache.match(cacheUrl(model))) ??
      (await cache.match(model.cacheKey));
    if (!match) return { kind: "missing" };
    const buf = await match.arrayBuffer();
    total += buf.byteLength;
  }
  return { kind: "ready", version: MODEL_VERSION, bytes: total };
}

export async function readCachedModel(
  model: ModelArtifact,
): Promise<ArrayBuffer | undefined> {
  if (typeof caches === "undefined") return undefined;
  const cache = await caches.open(MODEL_CACHE_NAME);
  const match =
    (await cache.match(cacheUrl(model))) ??
    (await cache.match(model.cacheKey));
  if (!match) return undefined;
  return match.arrayBuffer();
}

export type DownloadProgress = {
  modelId: string;
  received: number;
  total: number;
  progress: number;
};

/**
 * One-time download of public model weights into Cache Storage.
 * Subsequent runs reuse the cache and never hit the network for models.
 */
export async function downloadModels(args?: {
  onProgress?: (p: DownloadProgress) => void;
  fetchImpl?: typeof fetch;
}): Promise<ModelStatus> {
  const fetchImpl = args?.fetchImpl ?? fetch;
  if (typeof caches === "undefined") {
    return { kind: "error", message: "Cache Storage unavailable" };
  }

  try {
    const cache = await caches.open(MODEL_CACHE_NAME);
    let totalBytes = 0;

    for (const model of ALL_MODELS) {
      const key = cacheUrl(model);
      const existing =
        (await cache.match(key)) ?? (await cache.match(model.cacheKey));
      if (existing) {
        const buf = await existing.arrayBuffer();
        const digest = await sha256Hex(buf);
        if (digest === model.sha256) {
          totalBytes += buf.byteLength;
          // Migrate legacy relative keys to https://truepixel.local/…
          if (!(await cache.match(key))) {
            await cache.put(key, existing.clone());
          }
          continue;
        }
        await cache.delete(key);
        await cache.delete(model.cacheKey);
      }

      if (!model.url) {
        return {
          kind: "error",
          message: `${model.id} has no public URL — package models/${model.localPath} (npm run setup:models)`,
        };
      }
      const response = await fetchImpl(model.url, { redirect: "follow" });
      if (!response.ok || !response.body) {
        return {
          kind: "error",
          message: `Failed to download ${model.id}: HTTP ${response.status}`,
        };
      }

      const reader = response.body.getReader();
      const chunks: Uint8Array[] = [];
      let received = 0;
      const totalHeader = Number(response.headers.get("content-length") ?? model.bytes);

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        if (value) {
          chunks.push(value);
          received += value.byteLength;
          args?.onProgress?.({
            modelId: model.id,
            received,
            total: totalHeader,
            progress: totalHeader > 0 ? received / totalHeader : 0,
          });
        }
      }

      const merged = concatChunks(chunks, received);
      const buffer = new ArrayBuffer(merged.byteLength);
      new Uint8Array(buffer).set(merged);
      const digest = await sha256Hex(buffer);
      if (digest !== model.sha256) {
        return {
          kind: "error",
          message: `Checksum mismatch for ${model.id}: got ${digest}`,
        };
      }

      await cache.put(
        cacheUrl(model),
        new Response(buffer, {
          headers: {
            "Content-Type": "application/octet-stream",
            "X-TruePixel-SHA256": digest,
            "X-TruePixel-Model-Id": model.id,
          },
        }),
      );
      totalBytes += merged.byteLength;
    }

    return { kind: "ready", version: MODEL_VERSION, bytes: totalBytes };
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    return { kind: "error", message };
  }
}

function concatChunks(chunks: readonly Uint8Array[], total: number): Uint8Array {
  const out = new Uint8Array(total);
  let offset = 0;
  for (const chunk of chunks) {
    out.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return out;
}

/** Install models from local ArrayBuffers (used by build/setup scripts & tests). */
export async function installModelBytes(
  model: ModelArtifact,
  bytes: ArrayBuffer,
): Promise<void> {
  if (typeof caches === "undefined") {
    throw new Error("Cache Storage unavailable");
  }
  const digest = await sha256Hex(bytes);
  if (digest !== model.sha256) {
    throw new Error(`Checksum mismatch for ${model.id}: got ${digest}`);
  }
  const cache = await caches.open(MODEL_CACHE_NAME);
  await cache.put(
    cacheUrl(model),
    new Response(bytes, {
      headers: {
        "Content-Type": "application/octet-stream",
        "X-TruePixel-SHA256": digest,
        "X-TruePixel-Model-Id": model.id,
      },
    }),
  );
}

/**
 * Seed Cache Storage from packaged `dist/models/...` (offline CI / local suite).
 * Falls back to network downloadModels when packaged files are absent.
 */
export async function seedPackagedModels(): Promise<ModelStatus> {
  if (typeof caches === "undefined" || typeof chrome === "undefined") {
    return { kind: "error", message: "Packaged seed unavailable" };
  }
  try {
    let total = 0;
    for (const model of ALL_MODELS) {
      const url = chrome.runtime.getURL(model.localPath);
      const response = await fetch(url);
      if (!response.ok) {
        return { kind: "missing" };
      }
      const bytes = await response.arrayBuffer();
      await installModelBytes(model, bytes);
      total += bytes.byteLength;
    }
    return { kind: "ready", version: MODEL_VERSION, bytes: total };
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    return { kind: "error", message };
  }
}

/** Prefer packaged models; otherwise one-time public download. */
export async function ensureModelsReady(args?: {
  onProgress?: (p: DownloadProgress) => void;
  fetchImpl?: typeof fetch;
}): Promise<ModelStatus> {
  const status = await getModelStatus();
  if (status.kind === "ready") return status;
  const seeded = await seedPackagedModels();
  if (seeded.kind === "ready") return seeded;
  return downloadModels(args);
}
