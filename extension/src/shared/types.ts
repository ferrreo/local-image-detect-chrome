/** Confidence in [0, 1] that an image is AI-generated. */
export type AiConfidence = number & { readonly __brand: "AiConfidence" };

export type DetectionLabel =
  | { kind: "ai"; confidence: AiConfidence }
  | { kind: "real"; confidence: AiConfidence }
  | { kind: "uncertain"; confidence: AiConfidence }
  | { kind: "error"; message: string };

export type DetectorTierId =
  | "provenance"
  | "spectral"
  | "visual"
  | "fusion";

export type TierSignal = {
  tier: DetectorTierId;
  /** P(AI) contribution from this tier, [0, 1]. */
  aiScore: AiConfidence;
  weight: number;
  detail: string;
  shortCircuit?: boolean;
};

export type DetectionResult = {
  imageId: string;
  label: DetectionLabel;
  /** Final calibrated P(AI) used for the 65% evaluation threshold. */
  confidence: AiConfidence;
  tiers: TierSignal[];
  backend: InferenceBackend;
  elapsedMs: number;
};

export type InferenceBackend =
  | { kind: "webgpu" }
  | { kind: "wasm" }
  | { kind: "stub" }
  | { kind: "none" };

export type ModelStatus =
  | { kind: "missing" }
  | { kind: "downloading"; progress: number }
  | { kind: "ready"; version: string; bytes: number }
  | { kind: "error"; message: string };

export type AnalyzeImageRequest = {
  kind: "analyze-image";
  requestId: string;
  imageId: string;
  /** Absolute http(s)/blob/data URL, or chrome-extension URL. */
  src: string;
  width: number;
  height: number;
};

export type AnalyzeImageResponse = {
  kind: "analyze-image-result";
  requestId: string;
  result: DetectionResult;
};

export type SetupModelsRequest = {
  kind: "setup-models";
  requestId: string;
};

export type SetupModelsResponse = {
  kind: "setup-models-result";
  requestId: string;
  status: ModelStatus;
};

export type GetStatusRequest = {
  kind: "get-status";
  requestId: string;
};

export type GetStatusResponse = {
  kind: "get-status-result";
  requestId: string;
  models: ModelStatus;
  backend: InferenceBackend;
  autoScan: boolean;
  threshold: number;
};

export type SetOptionsRequest = {
  kind: "set-options";
  requestId: string;
  autoScan?: boolean;
  threshold?: number;
  debug?: boolean;
};

export type SetOptionsResponse = {
  kind: "set-options-result";
  requestId: string;
  ok: true;
};

export type ExtensionRequest =
  | AnalyzeImageRequest
  | SetupModelsRequest
  | GetStatusRequest
  | SetOptionsRequest;

export type ExtensionResponse =
  | AnalyzeImageResponse
  | SetupModelsResponse
  | GetStatusResponse
  | SetOptionsResponse
  | { kind: "error"; requestId: string; message: string };

export type OffscreenInferRequest = {
  kind: "offscreen-infer";
  requestId: string;
  imageId: string;
  /** Transferable ArrayBuffer of image bytes. */
  bytes: ArrayBuffer;
  mimeType: string;
};

export type OffscreenInferResponse = {
  kind: "offscreen-infer-result";
  requestId: string;
  result: DetectionResult;
};

export type ExtensionOptions = {
  autoScan: boolean;
  /** Evaluation threshold; default 0.65 per bounty brief. */
  threshold: number;
  debug: boolean;
  /** When true, skip network model fetch and use deterministic stub. */
  stubInference: boolean;
};

export const DEFAULT_OPTIONS = {
  autoScan: true,
  threshold: 0.65,
  debug: false,
  stubInference: false,
} as const satisfies ExtensionOptions;

export const EVAL_CONFIDENCE_THRESHOLD = 0.65;

export function asAiConfidence(value: number): AiConfidence {
  if (!Number.isFinite(value)) {
    throw new Error(`AiConfidence must be finite, got ${value}`);
  }
  const clamped = Math.min(1, Math.max(0, value));
  return clamped as AiConfidence;
}
