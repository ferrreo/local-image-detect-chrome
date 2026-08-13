import { detectAiImage } from "../lib/pipeline";
import {
  getVisualBackend,
  isGpuAvailable,
  resetVisualClassifier,
  setPreferredVisualProvider,
  warmVisualClassifier,
} from "../lib/visual-classifier";
import type {
  OffscreenInferRequest,
  OffscreenInferResponse,
  OffscreenResetRequest,
  OffscreenResetResponse,
  VisualProvider,
} from "../shared/types";
import { asAiConfidence } from "../shared/types";

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

async function handleInfer(
  request: OffscreenInferRequest,
): Promise<OffscreenInferResponse> {
  const result = await detectAiImage({
    imageId: request.imageId,
    bytes: request.bytes,
    mimeType: request.mimeType,
  });

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
  setPreferredVisualProvider(provider);
  resetVisualClassifier();
  let backend = getVisualBackend();
  if (request.warm !== false) {
    backend = await warmVisualClassifier();
  }
  return {
    kind: "offscreen-reset-result",
    requestId: request.requestId,
    backend,
    gpuAvailable: isGpuAvailable(),
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
  setPreferredVisualProvider(await loadProviderPreference());
  await warmVisualClassifier().catch(() => {
    // Model may not be installed yet; popup/eval setup handles download.
  });
})();
