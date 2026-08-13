import {
  DEFAULT_OPTIONS,
  type AnalyzeBytesRequest,
  type AnalyzeImageRequest,
  type ExtensionOptions,
  type ExtensionRequest,
  type ExtensionResponse,
  type InferenceBackend,
  type ModelStatus,
  type OffscreenInferRequest,
  type OffscreenInferResponse,
  type OffscreenResetRequest,
  type OffscreenResetResponse,
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
const OFFSCREEN_REASON = "DOM_SCRAPING" as chrome.offscreen.Reason;

let optionsCache: ExtensionOptions = { ...DEFAULT_OPTIONS };
let backendCache: InferenceBackend = { kind: "none" };
let modelStatusCache: ModelStatus = { kind: "missing" };
let gpuAvailableCache = false;

async function loadOptions(): Promise<ExtensionOptions> {
  const stored = await chrome.storage.local.get(["options", "stubInference"]);
  const raw = stored.options;
  const merged: ExtensionOptions = {
    ...DEFAULT_OPTIONS,
    ...(typeof raw === "object" && raw !== null
      ? (raw as Partial<ExtensionOptions>)
      : {}),
  };
  if (stored.stubInference === true) merged.stubInference = true;
  optionsCache = merged;
  return merged;
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
  await chrome.offscreen.createDocument({
    url: OFFSCREEN_URL,
    reasons: [OFFSCREEN_REASON],
    justification:
      "Run ONNX Runtime WebGPU/WASM inference off the service worker.",
  });
  // First message right after createDocument often races the module listener.
  await new Promise((r) => setTimeout(r, 50));
}

async function fetchImageBytes(
  src: string,
): Promise<{ bytes: ArrayBuffer; mimeType: string }> {
  const response = await fetch(src, {
    credentials: "omit",
    cache: "force-cache",
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
  bytes: ArrayBuffer;
  mimeType: string;
}): Promise<OffscreenInferResponse | undefined> {
  if (!chrome.offscreen) return undefined;
  try {
    await ensureOffscreen();
    // Wait a tick so the offscreen listener is registered after createDocument.
    await new Promise((r) => setTimeout(r, 0));
    const message: OffscreenInferRequest = {
      kind: "offscreen-infer",
      requestId: args.requestId,
      imageId: args.imageId,
      bytesBase64: arrayBufferToBase64(args.bytes),
      mimeType: args.mimeType,
    };
    const response: unknown = await chrome.runtime.sendMessage(message);
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
    };
    const response: unknown = await chrome.runtime.sendMessage(message);
    if (
      typeof response === "object" &&
      response !== null &&
      "kind" in response &&
      response.kind === "offscreen-reset-result"
    ) {
      return response as OffscreenResetResponse;
    }
  } catch {
    return undefined;
  }
  return undefined;
}

function relabel(
  confidence: ReturnType<typeof asAiConfidence>,
  threshold: number,
) {
  if (confidence >= threshold) {
    return { kind: "ai" as const, confidence };
  }
  if (confidence <= 1 - threshold) {
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
}): Promise<import("../shared/types").DetectionResult> {
  const options = await loadOptions();
  if (options.stubInference) {
    const result = await analyzeLocalStub({
      imageId: args.imageId,
      bytes: args.bytes,
      mimeType: args.mimeType,
      threshold: options.threshold,
    });
    backendCache = result.backend;
    return result;
  }

  const offscreen = await inferViaOffscreen({
    requestId: crypto.randomUUID(),
    imageId: args.imageId,
    bytes: args.bytes,
    mimeType: args.mimeType,
  });
  if (offscreen && offscreen.result.label.kind !== "error") {
    backendCache = offscreen.result.backend;
    return {
      ...offscreen.result,
      label: relabel(offscreen.result.confidence, options.threshold),
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
    const { bytes, mimeType } = await fetchImageBytes(request.src);
    const result = await detectFromBytes({
      imageId: request.imageId,
      bytes,
      mimeType,
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
        autoScan: options.autoScan,
        threshold: options.threshold,
        visualProvider: options.visualProvider,
        gpuAvailable: gpuAvailableCache,
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
        ...(request.debug !== undefined ? { debug: request.debug } : {}),
        ...(request.visualProvider !== undefined
          ? { visualProvider: request.visualProvider }
          : {}),
        ...(request.stubInference !== undefined
          ? { stubInference: request.stubInference }
          : {}),
      });
      return {
        kind: "set-options-result",
        requestId: request.requestId,
        ok: true,
      };
    }
    case "reset-visual": {
      const options = await loadOptions();
      let reset = await resetViaOffscreen({
        requestId: request.requestId,
        ...(request.warm !== undefined ? { warm: request.warm } : {}),
        visualProvider: options.visualProvider,
      });
      // One retry — first warm after offscreen boot is flaky under headless.
      if (!reset || reset.backend.kind === "none") {
        await new Promise((r) => setTimeout(r, 100));
        reset = await resetViaOffscreen({
          requestId: request.requestId,
          warm: true,
          visualProvider: options.visualProvider,
        });
      }
      if (reset) {
        backendCache = reset.backend;
        gpuAvailableCache = reset.gpuAvailable;
        return {
          kind: "reset-visual-result",
          requestId: request.requestId,
          backend: reset.backend,
          gpuAvailable: reset.gpuAvailable,
          visualEngine: reset.visualEngine,
        };
      }
      return {
        kind: "reset-visual-result",
        requestId: request.requestId,
        backend: backendCache,
        gpuAvailable: false,
        visualEngine: "none",
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
