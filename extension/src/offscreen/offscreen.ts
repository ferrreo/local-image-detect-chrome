import { detectAiImage } from "../lib/pipeline";
import { warmVisualClassifier } from "../lib/visual-classifier";
import type {
  OffscreenInferRequest,
  OffscreenInferResponse,
} from "../shared/types";
import { asAiConfidence } from "../shared/types";

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

chrome.runtime.onMessage.addListener((message: unknown, _sender, sendResponse) => {
  if (typeof message !== "object" || message === null || !("kind" in message)) {
    return false;
  }
  if (message.kind !== "offscreen-infer") return false;

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
});

void warmVisualClassifier().catch(() => {
  // Model may not be installed yet; popup setup handles download.
});
