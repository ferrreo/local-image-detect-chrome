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

/** Browser ORT execution provider preference for eval / options. */
export type VisualProvider =
  | { kind: "auto" }
  | { kind: "webgpu" }
  | { kind: "wasm" };

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
  visualProvider: VisualProvider["kind"];
  gpuAvailable: boolean;
};

export type SetOptionsRequest = {
  kind: "set-options";
  requestId: string;
  autoScan?: boolean;
  threshold?: number;
  debug?: boolean;
  visualProvider?: VisualProvider["kind"];
  stubInference?: boolean;
};

export type SetOptionsResponse = {
  kind: "set-options-result";
  requestId: string;
  ok: true;
};

/** Offline eval: analyze raw image bytes without a network image URL. */
export type AnalyzeBytesRequest = {
  kind: "analyze-bytes";
  requestId: string;
  imageId: string;
  bytes: ArrayBuffer;
  mimeType: string;
};

export type AnalyzeBytesResponse = {
  kind: "analyze-bytes-result";
  requestId: string;
  result: DetectionResult;
};

/** Reset ORT sessions (switch WebGPU ↔ WASM) and optionally re-warm. */
export type ResetVisualRequest = {
  kind: "reset-visual";
  requestId: string;
  warm?: boolean;
};

/** Which visual runtime the offscreen document warmed. */
export type VisualEngineId = "zig-ort-wasm" | "onnxruntime-web" | "stub" | "none";

export type ResetVisualResponse = {
  kind: "reset-visual-result";
  requestId: string;
  backend: InferenceBackend;
  gpuAvailable: boolean;
  visualEngine: VisualEngineId;
};

export type ExtensionRequest =
  | AnalyzeImageRequest
  | AnalyzeBytesRequest
  | SetupModelsRequest
  | GetStatusRequest
  | SetOptionsRequest
  | ResetVisualRequest;

export type ExtensionResponse =
  | AnalyzeImageResponse
  | AnalyzeBytesResponse
  | SetupModelsResponse
  | GetStatusResponse
  | SetOptionsResponse
  | ResetVisualResponse
  | { kind: "error"; requestId: string; message: string };

export type OffscreenInferRequest = {
  kind: "offscreen-infer";
  requestId: string;
  imageId: string;
  /**
   * Image bytes as base64. ArrayBuffer is not reliable across SW↔offscreen
   * structured clone (often arrives non-buffer → decode failures).
   */
  bytesBase64: string;
  mimeType: string;
};

export type OffscreenInferResponse = {
  kind: "offscreen-infer-result";
  requestId: string;
  result: DetectionResult;
};

export type OffscreenResetRequest = {
  kind: "offscreen-reset";
  requestId: string;
  warm?: boolean;
  visualProvider?: VisualProvider["kind"];
};

export type OffscreenResetResponse = {
  kind: "offscreen-reset-result";
  requestId: string;
  backend: InferenceBackend;
  gpuAvailable: boolean;
  visualEngine: VisualEngineId;
};

export type ExtensionOptions = {
  autoScan: boolean;
  /** Evaluation threshold; default 0.65 per bounty brief. */
  threshold: number;
  debug: boolean;
  /** When true, skip network model fetch and use deterministic stub. */
  stubInference: boolean;
  /** ORT EP preference for the offscreen visual classifier. */
  visualProvider: VisualProvider["kind"];
};

export const DEFAULT_OPTIONS = {
  autoScan: true,
  threshold: 0.65,
  debug: false,
  stubInference: false,
  visualProvider: "auto",
} as const satisfies ExtensionOptions;

export const EVAL_CONFIDENCE_THRESHOLD = 0.65;

export function asAiConfidence(value: number): AiConfidence {
  if (!Number.isFinite(value)) {
    throw new Error(`AiConfidence must be finite, got ${value}`);
  }
  const clamped = Math.min(1, Math.max(0, value));
  return clamped as AiConfidence;
}
