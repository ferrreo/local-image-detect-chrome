import { detectAiImage, type VisualEngineKind } from "../lib/pipeline";
import {
  getVisualBackend,
  probeWebGpuAvailable,
  resetVisualClassifier,
  setPreferredVisualProvider,
  warmVisualClassifier,
} from "../lib/visual-classifier";
import {
  getZigWasmLoadError,
  isZigWasmOrtReady,
  resetVisualZigWasm,
  warmVisualZigWasm,
} from "../lib/visual-zig-wasm";
import type {
  OffscreenInferRequest,
  OffscreenInferResponse,
  OffscreenResetRequest,
  OffscreenResetResponse,
  VisualEngineId,
  VisualEnginePreference,
  VisualProvider,
} from "../shared/types";
import { asAiConfidence } from "../shared/types";
import { base64ToArrayBuffer } from "../lib/bytes-codec";
import { guessMimeType } from "../lib/image-decode";

/** Last warmed engine preference — infer uses this unless overridden. */
let activeEnginePref: VisualEnginePreference = "auto";

async function loadProviderPreference(): Promise<VisualProvider["kind"]> {
  try {
    const stored = await chrome.storage.local.get(["options"]);
    const raw = stored.options;
    if (
      typeof raw === "object" &&
      raw !== null &&
      "visualProvider" in raw &&
      (raw.visualProvider === "auto" ||
        raw.visualProvider === "webgpu" ||
        raw.visualProvider === "wasm")
    ) {
      return raw.visualProvider;
    }
  } catch {
    // ignore
  }
  return "auto";
}

async function loadInferBytes(
  request: OffscreenInferRequest,
): Promise<{ bytes: ArrayBuffer; mimeType: string }> {
  if (request.src) {
    const response = await fetch(request.src, {
      credentials: "omit",
      cache: "force-cache",
    });
    if (!response.ok) {
      throw new Error(`Offscreen image fetch failed (${response.status})`);
    }
    const bytes = await response.arrayBuffer();
    const headerType = response.headers.get("content-type") ?? "";
    const mimeType = headerType.startsWith("image/")
      ? (headerType.split(";")[0] ?? request.mimeType)
      : guessMimeType(new Uint8Array(bytes));
    return { bytes, mimeType };
  }
  if (request.bytesBase64) {
    return {
      bytes: base64ToArrayBuffer(request.bytesBase64),
      mimeType: request.mimeType,
    };
  }
  throw new Error("offscreen-infer requires src or bytesBase64");
}

function formatTiming(result: Awaited<ReturnType<typeof detectAiImage>>): string {
  const t = result.timing;
  if (!t) return `total=${result.elapsedMs.toFixed(1)}ms`;
  return [
    `total=${t.totalMs.toFixed(1)}ms`,
    `decode=${t.decodeMs.toFixed(1)}`,
    `spectral=${t.spectralMs.toFixed(1)}`,
    `prep=${t.preprocessMs.toFixed(1)}`,
    `distilled=${t.distilledMs.toFixed(1)}`,
    `cf=${t.forensicsMs.toFixed(1)}${t.ranForensics ? "" : "(skip)"}`,
    `fuse=${t.fuseMs.toFixed(1)}`,
  ].join(" ");
}

function toPipelineEngine(
  pref: VisualEnginePreference,
): VisualEngineKind | "auto" {
  if (pref === "zig-ort-wasm") return "zig-wasm";
  if (pref === "onnxruntime-web") return "onnxruntime-web";
  return "auto";
}

async function resolveWarmEngine(
  pref: VisualEnginePreference,
): Promise<{ backend: ReturnType<typeof getVisualBackend>; visualEngine: VisualEngineId }> {
  if (pref === "onnxruntime-web") {
    const backend = await warmVisualClassifier();
    return { backend, visualEngine: "onnxruntime-web" };
  }
  if (pref === "zig-ort-wasm") {
    if (!(await isZigWasmOrtReady())) {
      const zigErr = getZigWasmLoadError() ?? "Zig+ORT WASM not ready";
      throw new Error(`visualEngine=zig-ort-wasm failed: ${zigErr}`);
    }
    const backend = await warmVisualZigWasm();
    return { backend, visualEngine: "zig-ort-wasm" };
  }
  // auto: prefer Zig when linked, else ort-web (product default).
  if (await isZigWasmOrtReady()) {
    const backend = await warmVisualZigWasm();
    return { backend, visualEngine: "zig-ort-wasm" };
  }
  const zigErr = getZigWasmLoadError();
  if (zigErr) {
    console.warn("Zig+ORT WASM unavailable, using onnxruntime-web:", zigErr);
  }
  const backend = await warmVisualClassifier();
  return { backend, visualEngine: "onnxruntime-web" };
}

async function handleInfer(
  request: OffscreenInferRequest,
): Promise<OffscreenInferResponse> {
  const fetchT0 = performance.now();
  const { bytes, mimeType } = await loadInferBytes(request);
  const fetchMs = performance.now() - fetchT0;
  const enginePref = request.visualEngine ?? activeEnginePref;
  const result = await detectAiImage({
    imageId: request.imageId,
    bytes,
    mimeType,
    ...(request.speedMode ? { speedMode: request.speedMode } : {}),
    visualEngine: toPipelineEngine(enginePref),
  });

  // Structured stage times — CF on single-thread ORT WASM is the 1s+ spike.
  console.info(
    `[truepixel] ${request.imageId} fetch=${fetchMs.toFixed(1)}ms ${formatTiming(result)} ` +
      `mode=${request.speedMode ?? "accurate"} engine=${enginePref}`,
  );

  return {
    kind: "offscreen-infer-result",
    requestId: request.requestId,
    result,
  };
}

async function handleReset(
  request: OffscreenResetRequest,
): Promise<OffscreenResetResponse> {
  const provider =
    request.visualProvider ?? (await loadProviderPreference());
  const enginePref = request.visualEngine ?? "auto";
  activeEnginePref = enginePref;
  setPreferredVisualProvider(provider);
  resetVisualClassifier();
  resetVisualZigWasm();
  let backend = getVisualBackend();
  let visualEngine: VisualEngineId = "none";
  if (request.warm !== false) {
    const warmed = await resolveWarmEngine(enginePref);
    backend = warmed.backend;
    visualEngine = warmed.visualEngine;
  }
  return {
    kind: "offscreen-reset-result",
    requestId: request.requestId,
    backend,
    gpuAvailable: await probeWebGpuAvailable(),
    visualEngine,
  };
}

chrome.runtime.onMessage.addListener((message: unknown, _sender, sendResponse) => {
  if (typeof message !== "object" || message === null || !("kind" in message)) {
    return false;
  }

  if (message.kind === "offscreen-infer") {
    void handleInfer(message as OffscreenInferRequest)
      .then(sendResponse)
      .catch((error: unknown) => {
        const msg = error instanceof Error ? error.message : String(error);
        const failure: OffscreenInferResponse = {
          kind: "offscreen-infer-result",
          requestId: (message as OffscreenInferRequest).requestId,
          result: {
            imageId: (message as OffscreenInferRequest).imageId,
            label: { kind: "error", message: msg },
            confidence: asAiConfidence(0.5),
            tiers: [],
            backend: { kind: "none" },
            elapsedMs: 0,
          },
        };
        sendResponse(failure);
      });
    return true;
  }

  if (message.kind === "offscreen-reset") {
    void handleReset(message as OffscreenResetRequest)
      .then(sendResponse)
      .catch((error: unknown) => {
        const msg = error instanceof Error ? error.message : String(error);
        sendResponse({
          kind: "error",
          requestId: (message as OffscreenResetRequest).requestId,
          message: msg,
        });
      });
    return true;
  }

  return false;
});

void (async () => {
  // Only set preference here — do not auto-warm. Concurrent warm with
  // reset-visual triggers onnxruntime-web "multiple calls to initWasm()".
  setPreferredVisualProvider(await loadProviderPreference());
})();
