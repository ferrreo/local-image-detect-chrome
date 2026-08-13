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

/** Per-stage wall times for the local pipeline (milliseconds). */
export type PipelineTiming = {
  decodeMs: number;
  spectralMs: number;
  /** RGBA / NCHW prep inside the visual backend (when reported). */
  preprocessMs: number;
  distilledMs: number;
  forensicsMs: number;
  fuseMs: number;
  totalMs: number;
  ranForensics: boolean;
};

export type DetectionResult = {
  imageId: string;
  label: DetectionLabel;
  /** Final calibrated P(AI) used for the 65% evaluation threshold. */
  confidence: AiConfidence;
  tiers: TierSignal[];
  backend: InferenceBackend;
  elapsedMs: number;
  timing?: PipelineTiming;
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

/**
 * `accurate` — cascade dual (distilled → CF when gated). Eval / overlay refine.
 * `realtime` — distilled+spectral only (fast first paint; may miss CF recoveries).
 */
export type AnalyzeSpeedMode = "accurate" | "realtime";

export type AnalyzeImageRequest = {
  kind: "analyze-image";
  requestId: string;
  imageId: string;
  /** Absolute http(s)/blob/data URL, or chrome-extension URL. */
  src: string;
  width: number;
  height: number;
  speedMode?: AnalyzeSpeedMode;
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

/** How to hide images labeled AI until the user clicks the badge. */
export type AiConcealMode = "none" | "blur" | "blank";

export type GetStatusResponse = {
  kind: "get-status-result";
  requestId: string;
  models: ModelStatus;
  backend: InferenceBackend;
  autoScan: boolean;
  threshold: number;
  visualProvider: VisualProvider["kind"];
  gpuAvailable: boolean;
  aiConceal: AiConcealMode;
};

export type SetOptionsRequest = {
  kind: "set-options";
  requestId: string;
  autoScan?: boolean;
  threshold?: number;
  debug?: boolean;
  visualProvider?: VisualProvider["kind"];
  stubInference?: boolean;
  aiConceal?: AiConcealMode;
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

/** Which visual stack to warm / use for inference. */
export type VisualEnginePreference =
  | "auto"
  | "zig-ort-wasm"
  | "onnxruntime-web";

/** Reset ORT sessions (switch WebGPU ↔ WASM) and optionally re-warm. */
export type ResetVisualRequest = {
  kind: "reset-visual";
  requestId: string;
  warm?: boolean;
  /**
   * Force engine for this warm. Eval uses `onnxruntime-web` to avoid
   * Zig WASM shadowing the ort-web cascade path.
   */
  visualEngine?: VisualEnginePreference;
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
   * Prefer `src` when the offscreen document can fetch it (eval / http images).
   * Avoids SW fetch + base64 + second decode on the hot path.
   */
  src?: string;
  /**
   * Image bytes as base64 when `src` is unavailable.
   * ArrayBuffer is not reliable across SW↔offscreen structured clone.
   */
  bytesBase64?: string;
  mimeType: string;
  /**
   * `realtime` → cascade off (distilled+spectral only).
   * `accurate` / omitted → full cascade (eval default).
   */
  speedMode?: AnalyzeSpeedMode;
  /** Override visual stack for this infer (else last reset preference). */
  visualEngine?: VisualEnginePreference;
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
  visualEngine?: VisualEnginePreference;
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
  /** Hide AI-labeled images (click badge to reveal). */
  aiConceal: AiConcealMode;
};

export const DEFAULT_OPTIONS = {
  autoScan: true,
  threshold: 0.65,
  debug: false,
  stubInference: false,
  visualProvider: "auto",
  aiConceal: "blur",
} as const satisfies ExtensionOptions;

export function parseAiConcealMode(value: unknown): AiConcealMode {
  if (value === "blur" || value === "blank" || value === "none") return value;
  return DEFAULT_OPTIONS.aiConceal;
}

export const EVAL_CONFIDENCE_THRESHOLD = 0.65;

export function asAiConfidence(value: number): AiConfidence {
  if (!Number.isFinite(value)) {
    throw new Error(`AiConfidence must be finite, got ${value}`);
  }
  const clamped = Math.min(1, Math.max(0, value));
  return clamped as AiConfidence;
}
