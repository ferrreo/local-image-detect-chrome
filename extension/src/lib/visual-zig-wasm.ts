/**
 * Browser Zig+ORT WASM visual backend.
 *
 * Requires `npm run build:zig-wasm` (links libonnxruntime_webassembly.a).
 * Loads emscripten MODULARIZE glue from `wasm/truepixel_infer.js`.
 */
import {
  DISTILLED_MODEL,
  FORENSICS_MODEL,
  VISUAL_MODEL,
  type ModelArtifact,
} from "./model-manifest";
import { readCachedModel } from "./model-cache";
import { rasterizeForModel } from "./image-decode";
import { asAiConfidence, type InferenceBackend } from "../shared/types";
import type { ClassifyVisualOptions } from "./visual-classifier";
import {
  stubVisualClassify,
  type VisualClassification,
} from "./visual-stub";
import { needsForensicsCascade } from "./forensics-cascade";

type ZigOrtModule = {
  HEAPU8: Uint8Array;
  HEAPF32: Float32Array;
  _tp_abi_version: () => number;
  _tp_has_ort_session: () => number;
  _tp_malloc: (n: number) => number;
  _tp_free: (ptr: number, n: number) => void;
  _tp_session_create: (
    modelPtr: number,
    modelLen: number,
    graphOptDisabled: number,
  ) => number;
  _tp_session_free: (id: number) => void;
  _tp_session_run: (
    id: number,
    nchwPtr: number,
    size: number,
    aiLabel: number,
  ) => number;
  _tp_rgb_to_nchw_half: (rgbPtr: number, size: number, outPtr: number) => void;
};

type CreateModule = (opts?: {
  locateFile?: (path: string) => string;
  wasmBinary?: Uint8Array;
}) => Promise<ZigOrtModule>;

let modulePromise: Promise<ZigOrtModule> | undefined;
let distilledId = -1;
let forensicsId = -1;
let warmPromise: Promise<InferenceBackend> | undefined;

function wasmBaseUrl(): string {
  if (typeof chrome !== "undefined" && chrome.runtime?.getURL) {
    return chrome.runtime.getURL("wasm/");
  }
  return "/wasm/";
}

async function loadModule(): Promise<ZigOrtModule> {
  if (!modulePromise) {
    modulePromise = (async () => {
      const base = wasmBaseUrl();
      const glueUrl = `${base}truepixel_infer.js`;
      const wasmUrl = `${base}truepixel_infer.wasm`;
      // Fetch .wasm ourselves — Chrome extension pages are more reliable with
      // wasmBinary than emscripten streaming fetch of chrome-extension:// URLs.
      const [mod, wasmBinary] = await Promise.all([
        import(/* @vite-ignore */ glueUrl) as Promise<{
          default: CreateModule;
          createTruepixelInfer?: CreateModule;
        }>,
        fetch(wasmUrl).then(async (r) => {
          if (!r.ok) throw new Error(`Failed to fetch ${wasmUrl}: ${r.status}`);
          return new Uint8Array(await r.arrayBuffer());
        }),
      ]);
      const create = mod.default ?? mod.createTruepixelInfer;
      if (!create) {
        throw new Error("Zig+ORT WASM glue missing createTruepixelInfer");
      }
      const instance = await create({ wasmBinary });
      if (instance._tp_has_ort_session() !== 1) {
        throw new Error("Zig WASM built without ORT (tp_has_ort_session!=1)");
      }
      return instance;
    })();
  }
  return modulePromise;
}

function writeBytes(m: ZigOrtModule, bytes: ArrayBuffer): { ptr: number; len: number } {
  const len = bytes.byteLength;
  const ptr = m._tp_malloc(len);
  if (!ptr) throw new Error("tp_malloc failed");
  m.HEAPU8.set(new Uint8Array(bytes), ptr);
  return { ptr, len };
}

function createSession(
  m: ZigOrtModule,
  model: ModelArtifact,
  bytes: ArrayBuffer,
): number {
  const { ptr, len } = writeBytes(m, bytes);
  try {
    const id = m._tp_session_create(
      ptr,
      len,
      model.graphOptimizationLevel === "disabled" ? 1 : 0,
    );
    if (id < 0) {
      throw new Error(`tp_session_create failed for ${model.id}`);
    }
    return id;
  } finally {
    m._tp_free(ptr, len);
  }
}

let lastLoadError: string | undefined;

export function getZigWasmLoadError(): string | undefined {
  return lastLoadError;
}

export async function isZigWasmOrtReady(): Promise<boolean> {
  try {
    // Fast path: packaged artifacts must exist before we claim ready.
    if (typeof chrome !== "undefined" && chrome.runtime?.getURL) {
      const probe = await fetch(chrome.runtime.getURL("wasm/truepixel_infer.wasm"), {
        method: "HEAD",
      }).catch(() => null);
      if (!probe?.ok) {
        lastLoadError = "dist/wasm/truepixel_infer.wasm missing";
        return false;
      }
    }
    const m = await loadModule();
    const ok = m._tp_has_ort_session() === 1;
    if (!ok) lastLoadError = "tp_has_ort_session!=1";
    else lastLoadError = undefined;
    return ok;
  } catch (error) {
    lastLoadError = error instanceof Error ? error.message : String(error);
    modulePromise = undefined;
    return false;
  }
}

export async function warmVisualZigWasm(): Promise<InferenceBackend> {
  if (!warmPromise) {
    warmPromise = (async () => {
      const m = await loadModule();
      const distilledBytes = await readCachedModel(DISTILLED_MODEL);
      const forensicsBytes = await readCachedModel(FORENSICS_MODEL);
      if (!distilledBytes || !forensicsBytes) {
        throw new Error("Zig+ORT WASM: models missing from cache. Run setup.");
      }
      if (distilledId >= 0) m._tp_session_free(distilledId);
      if (forensicsId >= 0) m._tp_session_free(forensicsId);
      distilledId = createSession(m, DISTILLED_MODEL, distilledBytes);
      forensicsId = createSession(m, FORENSICS_MODEL, forensicsBytes);
      return { kind: "wasm" as const };
    })();
  }
  return warmPromise;
}

async function runSession(
  m: ZigOrtModule,
  sessionId: number,
  bitmap: ImageBitmap,
  model: ModelArtifact,
): Promise<number> {
  const imageData = await rasterizeForModel(bitmap, model.inputSize);
  const size = model.inputSize;
  const rgbLen = size * size * 3;
  const nchwLen = 3 * size * size;
  const rgbPtr = m._tp_malloc(rgbLen);
  const nchwPtr = m._tp_malloc(nchwLen * 4);
  if (!rgbPtr || !nchwPtr) {
    if (rgbPtr) m._tp_free(rgbPtr, rgbLen);
    if (nchwPtr) m._tp_free(nchwPtr, nchwLen * 4);
    throw new Error("tp_malloc for tensors failed");
  }
  try {
    // imageData is RGBA — pack RGB for the Zig helper.
    const rgba = imageData.data;
    const rgb = new Uint8Array(rgbLen);
    for (let i = 0, j = 0; i < size * size; i += 1, j += 3) {
      const p = i * 4;
      rgb[j] = rgba[p] ?? 0;
      rgb[j + 1] = rgba[p + 1] ?? 0;
      rgb[j + 2] = rgba[p + 2] ?? 0;
    }
    m.HEAPU8.set(rgb, rgbPtr);
    // emscripten pointers are byte addresses for both [*]u8 and [*]f32 exports.
    m._tp_rgb_to_nchw_half(rgbPtr, size, nchwPtr);
    const score = m._tp_session_run(
      sessionId,
      nchwPtr,
      size,
      model.aiLabelIndex,
    );
    if (!(score >= 0) || score > 1) {
      throw new Error(`tp_session_run failed: ${score}`);
    }
    return score;
  } finally {
    m._tp_free(rgbPtr, rgbLen);
    m._tp_free(nchwPtr, nchwLen * 4);
  }
}

export async function classifyVisualZigWasm(
  bitmap: ImageBitmap,
  options?: ClassifyVisualOptions,
): Promise<VisualClassification> {
  if (options?.stub) {
    const imageData = await rasterizeForModel(bitmap, VISUAL_MODEL.inputSize);
    return stubVisualClassify(imageData);
  }

  await warmVisualZigWasm();
  const m = await loadModule();
  if (distilledId < 0 || forensicsId < 0) {
    throw new Error("Zig+ORT sessions not warmed");
  }

  const cascade = options?.cascade !== false;
  const forceDistilled = options?.runDistilled;
  const forceForensics = options?.runForensics;
  const runDistilled = forceDistilled !== false;
  let runForensics = forceForensics === true;
  if (forceForensics === undefined && forceDistilled === undefined) {
    runForensics = !cascade;
  }

  const scores = new Map<string, number>();
  if (runDistilled) {
    scores.set(
      DISTILLED_MODEL.id,
      await runSession(m, distilledId, bitmap, DISTILLED_MODEL),
    );
  }
  const distilled = scores.get(DISTILLED_MODEL.id);

  if (
    forceForensics === undefined &&
    forceDistilled === undefined &&
    cascade &&
    distilled !== undefined &&
    options?.spectral
  ) {
    runForensics = needsForensicsCascade({
      distilled,
      spectral: options.spectral.score,
      laplacianVariance: options.spectral.laplacianVariance,
      chromaFlatness: options.spectral.chromaFlatness,
    });
  }

  if (runForensics) {
    scores.set(
      FORENSICS_MODEL.id,
      await runSession(m, forensicsId, bitmap, FORENSICS_MODEL),
    );
  }

  const forensics = scores.get(FORENSICS_MODEL.id);
  const primary = distilled !== undefined ? distilled : forensics;
  if (primary === undefined) throw new Error("No Zig+ORT visual score");

  return {
    score: asAiConfidence(primary),
    ...(forensics !== undefined
      ? { secondaryScore: asAiConfidence(forensics) }
      : {}),
    backend: { kind: "wasm" },
    detail: [
      ...[...scores.entries()].map(([id, v]) => `${id}=${v.toFixed(3)}`),
      "engine=zig-ort-wasm",
      cascade ? "cascade" : "dual",
      runForensics ? "ranForensics" : "skipForensics",
    ].join(","),
  };
}

export function resetVisualZigWasm(): void {
  modulePromise = undefined;
  warmPromise = undefined;
  distilledId = -1;
  forensicsId = -1;
}
