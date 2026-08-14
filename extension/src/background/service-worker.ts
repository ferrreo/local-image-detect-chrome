import {
  DEFAULT_OPTIONS,
  parseAiConcealMode,
  type AnalyzeBytesRequest,
  type AnalyzeImageRequest,
  type AnalyzeSpeedMode,
  type ExtensionOptions,
  type ExtensionRequest,
  type ExtensionResponse,
  type InferenceBackend,
  type ModelStatus,
  type OffscreenInferRequest,
  type OffscreenInferResponse,
  type OffscreenResetRequest,
  type OffscreenResetResponse,
  type VisualEnginePreference,
  type VisualProvider,
} from "../shared/types";
import { ensureModelsReady, getModelStatus } from "../lib/model-cache";
import { analyzeLocalStub } from "../lib/analyze-local";
import { guessMimeType } from "../lib/image-decode";
import {
  arrayBufferToBase64,
  coerceArrayBuffer,
} from "../lib/bytes-codec";
import { asAiConfidence } from "../shared/types";

const OFFSCREEN_URL = "offscreen.html";
/** ORT wasm threads spawn Workers; blobs used for image/tensor paths. */
const OFFSCREEN_REASONS = [
  "DOM_SCRAPING",
  "WORKERS",
  "BLOBS",
] as chrome.offscreen.Reason[];

let optionsCache: ExtensionOptions = { ...DEFAULT_OPTIONS };
let backendCache: InferenceBackend = { kind: "none" };
let backendErrorCache = "";
let modelStatusCache: ModelStatus = { kind: "missing" };
let gpuAvailableCache = false;
/** Deduped offscreen create (Proofmark pattern — no fixed sleep race). */
let creatingOffscreen: Promise<void> | undefined;
/** LRU-ish result cache: src|WxH|speedMode → DetectionResult */
const resultCache = new Map<
  string,
  import("../shared/types").DetectionResult
>();
const RESULT_CACHE_MAX = 400;
/** Parallel offscreen infers — was 2; grids felt capped and slow. */
const MAX_INFER_IN_FLIGHT = 6;
let inferInFlight = 0;
const inferWaiters: Array<() => void> = [];

async function withInferSlot<T>(fn: () => Promise<T>): Promise<T> {
  if (inferInFlight >= MAX_INFER_IN_FLIGHT) {
    await new Promise<void>((resolve) => {
      inferWaiters.push(resolve);
    });
  }
  inferInFlight += 1;
  try {
    return await fn();
  } finally {
    inferInFlight -= 1;
    inferWaiters.shift()?.();
  }
}

function cacheKey(
  src: string,
  width: number,
  height: number,
  speedMode: AnalyzeSpeedMode | undefined,
): string {
  return `${src}|${width}x${height}|${speedMode ?? "accurate"}`;
}

function rememberResult(
  key: string,
  result: import("../shared/types").DetectionResult,
): void {
  if (result.label.kind === "error") return;
  resultCache.set(key, result);
  if (resultCache.size > RESULT_CACHE_MAX) {
    const first = resultCache.keys().next().value;
    if (first !== undefined) resultCache.delete(first);
  }
}

async function loadOptions(): Promise<ExtensionOptions> {
  const stored = await chrome.storage.local.get(["options", "stubInference"]);
  const raw = stored.options;
  const partial =
    typeof raw === "object" && raw !== null
      ? (raw as Partial<ExtensionOptions>)
      : {};
  // Migrate pre-asymmetric installs that still have the old 65% single band.
  const migrated = { ...partial };
  if (
    migrated.realThreshold === undefined &&
    (migrated.threshold === undefined || migrated.threshold === 0.65)
  ) {
    migrated.threshold = DEFAULT_OPTIONS.threshold;
    migrated.realThreshold = DEFAULT_OPTIONS.realThreshold;
  }
  const merged: ExtensionOptions = {
    ...DEFAULT_OPTIONS,
    ...migrated,
    aiConceal: parseAiConcealMode(migrated.aiConceal),
  };
  if (stored.stubInference === true) merged.stubInference = true;
  optionsCache = merged;
  return merged;
}

async function broadcastOptionsChanged(): Promise<void> {
  const tabs = await chrome.tabs.query({});
  await Promise.all(
    tabs.map(async (tab) => {
      if (tab.id === undefined) return;
      try {
        await chrome.tabs.sendMessage(tab.id, {
          kind: "neopixel-options",
        });
      } catch {
        // No content script on this tab (chrome://, PDF, etc.).
      }
    }),
  );
}

async function saveOptions(
  patch: Partial<ExtensionOptions>,
): Promise<ExtensionOptions> {
  const next = { ...(await loadOptions()), ...patch };
  await chrome.storage.local.set({
    options: next,
    stubInference: next.stubInference,
  });
  optionsCache = next;
  return next;
}

async function ensureOffscreen(): Promise<void> {
  if (!chrome.offscreen) return;
  const existing = await chrome.runtime.getContexts({
    contextTypes: ["OFFSCREEN_DOCUMENT"],
  });
  if (existing.length > 0) return;
  if (!creatingOffscreen) {
    creatingOffscreen = chrome.offscreen
      .createDocument({
        url: OFFSCREEN_URL,
        reasons: OFFSCREEN_REASONS,
        justification:
          "Run ONNX Runtime WebGPU/WASM inference (incl. threaded wasm workers) off the service worker.",
      })
      .then(async () => {
        // Brief yield so the module listener registers after createDocument.
        await new Promise((r) => setTimeout(r, 0));
      })
      .finally(() => {
        creatingOffscreen = undefined;
      });
  }
  await creatingOffscreen;
}

function fetchCacheMode(src: string): RequestCache {
  if (/thispersondoesnotexist|random|uuid=|timestamp=|nocache/i.test(src)) {
    return "no-cache";
  }
  return "force-cache";
}

async function fetchImageBytes(
  src: string,
): Promise<{ bytes: ArrayBuffer; mimeType: string }> {
  const response = await fetch(src, {
    credentials: "omit",
    cache: fetchCacheMode(src),
  });
  if (!response.ok) {
    throw new Error(`Image fetch failed (${response.status})`);
  }
  const bytes = await response.arrayBuffer();
  const headerType = response.headers.get("content-type") ?? "";
  const mimeType = headerType.startsWith("image/")
    ? (headerType.split(";")[0] ?? "application/octet-stream")
    : guessMimeType(new Uint8Array(bytes));
  return { bytes, mimeType };
}

async function inferViaOffscreen(args: {
  requestId: string;
  imageId: string;
  bytes?: ArrayBuffer;
  src?: string;
  mimeType: string;
  speedMode?: AnalyzeSpeedMode;
  visualEngine?: VisualEnginePreference;
}): Promise<OffscreenInferResponse | undefined> {
  if (!chrome.offscreen) return undefined;
  try {
    await ensureOffscreen();
    const message: OffscreenInferRequest = {
      kind: "offscreen-infer",
      requestId: args.requestId,
      imageId: args.imageId,
      mimeType: args.mimeType,
      ...(args.speedMode ? { speedMode: args.speedMode } : {}),
      ...(args.visualEngine ? { visualEngine: args.visualEngine } : {}),
      ...(args.src
        ? { src: args.src }
        : {
            bytesBase64: arrayBufferToBase64(
              args.bytes ?? new ArrayBuffer(0),
            ),
          }),
    };
    const response: unknown = await withInferSlot(() =>
      chrome.runtime.sendMessage(message),
    );
    if (
      typeof response === "object" &&
      response !== null &&
      "kind" in response &&
      response.kind === "offscreen-infer-result"
    ) {
      return response as OffscreenInferResponse;
    }
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    return {
      kind: "offscreen-infer-result",
      requestId: args.requestId,
      result: {
        imageId: args.imageId,
        label: { kind: "error", message: `offscreen send failed: ${message}` },
        confidence: asAiConfidence(0.5),
        tiers: [],
        backend: { kind: "none" },
        elapsedMs: 0,
      },
    };
  }
  return undefined;
}

async function resetViaOffscreen(args: {
  requestId: string;
  warm?: boolean;
  visualProvider?: VisualProvider["kind"];
  visualEngine?: VisualEnginePreference;
}): Promise<OffscreenResetResponse | undefined> {
  if (!chrome.offscreen) return undefined;
  try {
    await ensureOffscreen();
    const message: OffscreenResetRequest = {
      kind: "offscreen-reset",
      requestId: args.requestId,
      ...(args.warm !== undefined ? { warm: args.warm } : {}),
      ...(args.visualProvider !== undefined
        ? { visualProvider: args.visualProvider }
        : {}),
      ...(args.visualEngine !== undefined
        ? { visualEngine: args.visualEngine }
        : {}),
    };
    const response: unknown = await chrome.runtime.sendMessage(message);
    if (
      typeof response === "object" &&
      response !== null &&
      "kind" in response
    ) {
      if (response.kind === "offscreen-reset-result") {
        return response as OffscreenResetResponse;
      }
      if (response.kind === "error" && "message" in response) {
        throw new Error(String((response as { message: unknown }).message));
      }
    }
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    throw new Error(`offscreen reset failed: ${message}`);
  }
  return undefined;
}

function relabel(
  confidence: ReturnType<typeof asAiConfidence>,
  aiThreshold: number,
  realThreshold: number,
) {
  if (confidence >= aiThreshold) {
    return { kind: "ai" as const, confidence };
  }
  if (confidence <= realThreshold) {
    return {
      kind: "real" as const,
      confidence: asAiConfidence(1 - confidence),
    };
  }
  return { kind: "uncertain" as const, confidence };
}

async function detectFromBytes(args: {
  imageId: string;
  bytes: ArrayBuffer;
  mimeType: string;
  speedMode?: AnalyzeSpeedMode;
}): Promise<import("../shared/types").DetectionResult> {
  const options = await loadOptions();
  if (options.stubInference) {
    const result = await analyzeLocalStub({
      imageId: args.imageId,
      bytes: args.bytes,
      mimeType: args.mimeType,
      threshold: options.threshold,
      realThreshold: options.realThreshold,
    });
    backendCache = result.backend;
    return result;
  }

  const offscreen = await inferViaOffscreen({
    requestId: crypto.randomUUID(),
    imageId: args.imageId,
    bytes: args.bytes,
    mimeType: args.mimeType,
    ...(args.speedMode ? { speedMode: args.speedMode } : {}),
  });
  if (offscreen && offscreen.result.label.kind !== "error") {
    backendCache = offscreen.result.backend;
    backendErrorCache = "";
    return {
      ...offscreen.result,
      label: relabel(
        offscreen.result.confidence,
        options.threshold,
        options.realThreshold,
      ),
    };
  }

  // Do not silently fall back to the heuristic stub — that reported as
  // "webgpu/wasm stub" in eval and tanked BA. Surface the offscreen failure.
  const message =
    offscreen?.result.label.kind === "error"
      ? offscreen.result.label.message
      : "Offscreen visual inference unavailable";
  backendCache = { kind: "none" };
  return {
    imageId: args.imageId,
    label: { kind: "error", message },
    confidence: asAiConfidence(0.5),
    tiers: [],
    backend: { kind: "none" },
    elapsedMs: 0,
  };
}

async function analyzeImage(
  request: AnalyzeImageRequest,
): Promise<ExtensionResponse> {
  try {
    const options = await loadOptions();
    const key = cacheKey(
      request.src,
      request.width,
      request.height,
      request.speedMode,
    );
    if (!request.bypassCache) {
      const cached = resultCache.get(key);
      if (cached) {
        return {
          kind: "analyze-image-result",
          requestId: request.requestId,
          result: {
            ...cached,
            imageId: request.imageId,
            label: relabel(
              cached.confidence,
              options.threshold,
              options.realThreshold,
            ),
          },
        };
      }
    }
    // Hot path: offscreen fetches `src` itself (no SW fetch + base64 tax).
    const offscreen = await inferViaOffscreen({
      requestId: request.requestId,
      imageId: request.imageId,
      src: request.src,
      mimeType: "application/octet-stream",
      ...(request.speedMode ? { speedMode: request.speedMode } : {}),
    });
    if (offscreen && offscreen.result.label.kind !== "error") {
      backendCache = offscreen.result.backend;
      backendErrorCache = "";
      const result = {
        ...offscreen.result,
        label: relabel(
          offscreen.result.confidence,
          options.threshold,
          options.realThreshold,
        ),
      };
      rememberResult(key, result);
      return {
        kind: "analyze-image-result",
        requestId: request.requestId,
        result,
      };
    }
    // Fallback: SW fetch + bytes path (content scripts / opaque URLs).
    const { bytes, mimeType } = await fetchImageBytes(request.src);
    const result = await detectFromBytes({
      imageId: request.imageId,
      bytes,
      mimeType,
      ...(request.speedMode ? { speedMode: request.speedMode } : {}),
    });
    return {
      kind: "analyze-image-result",
      requestId: request.requestId,
      result,
    };
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    return {
      kind: "analyze-image-result",
      requestId: request.requestId,
      result: {
        imageId: request.imageId,
        label: { kind: "error", message },
        confidence: asAiConfidence(0.5),
        tiers: [],
        backend: { kind: "none" },
        elapsedMs: 0,
      },
    };
  }
}

async function analyzeBytes(
  request: AnalyzeBytesRequest,
): Promise<ExtensionResponse> {
  try {
    const result = await detectFromBytes({
      imageId: request.imageId,
      bytes: coerceArrayBuffer(request.bytes),
      mimeType: request.mimeType,
    });
    return {
      kind: "analyze-bytes-result",
      requestId: request.requestId,
      result,
    };
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    return {
      kind: "analyze-bytes-result",
      requestId: request.requestId,
      result: {
        imageId: request.imageId,
        label: { kind: "error", message },
        confidence: asAiConfidence(0.5),
        tiers: [],
        backend: { kind: "none" },
        elapsedMs: 0,
      },
    };
  }
}

async function handleRequest(
  request: ExtensionRequest,
): Promise<ExtensionResponse> {
  switch (request.kind) {
    case "analyze-image":
      return analyzeImage(request);
    case "analyze-bytes":
      return analyzeBytes(request);
    case "setup-models": {
      if ((await loadOptions()).stubInference) {
        modelStatusCache = { kind: "ready", version: "stub", bytes: 0 };
        return {
          kind: "setup-models-result",
          requestId: request.requestId,
          status: modelStatusCache,
        };
      }
      modelStatusCache = { kind: "downloading", progress: 0 };
      modelStatusCache = await ensureModelsReady({
        onProgress: (p) => {
          modelStatusCache = { kind: "downloading", progress: p.progress };
        },
      });
      return {
        kind: "setup-models-result",
        requestId: request.requestId,
        status: modelStatusCache,
      };
    }
    case "get-status": {
      const options = await loadOptions();
      if (!options.stubInference) {
        modelStatusCache = await getModelStatus();
      } else {
        modelStatusCache = { kind: "ready", version: "stub", bytes: 0 };
      }
      return {
        kind: "get-status-result",
        requestId: request.requestId,
        models: modelStatusCache,
        backend: backendCache,
        ...(backendErrorCache ? { backendError: backendErrorCache } : {}),
        autoScan: options.autoScan,
        threshold: options.threshold,
        realThreshold: options.realThreshold,
        visualProvider: options.visualProvider,
        gpuAvailable: gpuAvailableCache,
        aiConceal: options.aiConceal,
        debug: options.debug,
      };
    }
    case "set-options": {
      await saveOptions({
        ...(request.autoScan !== undefined
          ? { autoScan: request.autoScan }
          : {}),
        ...(request.threshold !== undefined
          ? { threshold: request.threshold }
          : {}),
        ...(request.realThreshold !== undefined
          ? { realThreshold: request.realThreshold }
          : {}),
        ...(request.debug !== undefined ? { debug: request.debug } : {}),
        ...(request.visualProvider !== undefined
          ? { visualProvider: request.visualProvider }
          : {}),
        ...(request.stubInference !== undefined
          ? { stubInference: request.stubInference }
          : {}),
        ...(request.aiConceal !== undefined
          ? { aiConceal: parseAiConcealMode(request.aiConceal) }
          : {}),
      });
      void broadcastOptionsChanged();
      return {
        kind: "set-options-result",
        requestId: request.requestId,
        ok: true,
      };
    }
    case "reset-visual": {
      const options = await loadOptions();
      const visualEngine = request.visualEngine ?? "auto";
      let lastError: unknown;
      let reset: OffscreenResetResponse | undefined;
      try {
        reset = await resetViaOffscreen({
          requestId: request.requestId,
          ...(request.warm !== undefined ? { warm: request.warm } : {}),
          visualProvider: options.visualProvider,
          visualEngine,
        });
      } catch (error) {
        lastError = error;
      }
      // One retry — first warm after offscreen boot is flaky under headless.
      if (!reset || reset.backend.kind === "none") {
        await new Promise((r) => setTimeout(r, 150));
        try {
          // Recreate offscreen in case WORKERS/COEP blocked the first document.
          if (chrome.offscreen?.closeDocument) {
            try {
              await chrome.offscreen.closeDocument();
            } catch {
              // none open
            }
          }
          reset = await resetViaOffscreen({
            requestId: request.requestId,
            warm: true,
            visualProvider: options.visualProvider,
            visualEngine,
          });
        } catch (error) {
          lastError = error;
        }
      }
      if (reset && reset.backend.kind !== "none") {
        backendCache = reset.backend;
        backendErrorCache = "";
        gpuAvailableCache = reset.gpuAvailable;
        return {
          kind: "reset-visual-result",
          requestId: request.requestId,
          backend: reset.backend,
          gpuAvailable: reset.gpuAvailable,
          visualEngine: reset.visualEngine,
        };
      }
      const detail =
        reset?.error ||
        (lastError instanceof Error
          ? lastError.message
          : lastError
            ? String(lastError)
            : "offscreen warm returned no backend");
      backendErrorCache = detail;
      return {
        kind: "error",
        requestId: request.requestId,
        message: `reset-visual failed (requested ${visualEngine}): ${detail}`,
      };
    }
    default: {
      const _exhaustive: never = request;
      return {
        kind: "error",
        requestId: "unknown",
        message: `Unhandled request ${JSON.stringify(_exhaustive)}`,
      };
    }
  }
}

chrome.runtime.onInstalled.addListener(() => {
  void (async () => {
    await loadOptions();
    modelStatusCache = await getModelStatus();
  })();
});

chrome.runtime.onMessage.addListener((message: unknown, _sender, sendResponse) => {
  if (typeof message !== "object" || message === null || !("kind" in message)) {
    return false;
  }

  if (
    message.kind === "offscreen-infer" ||
    message.kind === "offscreen-infer-result" ||
    message.kind === "offscreen-reset" ||
    message.kind === "offscreen-reset-result"
  ) {
    return false;
  }

  if (
    message.kind === "analyze-image" ||
    message.kind === "analyze-bytes" ||
    message.kind === "setup-models" ||
    message.kind === "get-status" ||
    message.kind === "set-options" ||
    message.kind === "reset-visual"
  ) {
    void handleRequest(message as ExtensionRequest).then(sendResponse);
    return true;
  }

  return false;
});

void loadOptions();
