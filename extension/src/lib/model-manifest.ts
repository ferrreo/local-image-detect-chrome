/**
 * One-time setup downloads public distilled weights; the accurate head is
 * packaged locally (no public URL). After caching, inference never fetches
 * models again.
 *
 * Accurate-head distillation (NeoPixel-trained) lives in
 * `scripts/distill-accurate-head.py` and ships as FORENSICS_MODEL.
 */
export type ModelPreprocess = "stretch" | "short440-center384";
export type ModelOutputKind = "logits2" | "logit";

export type ModelArtifact = {
  id: string;
  /** Relative path under the Cache Storage namespace. */
  cacheKey: string;
  /** Local path used by Node eval (onnxruntime-node). */
  localPath: string;
  /** Public Hugging Face URL (resolved at setup time only). Empty = package-only. */
  url: string;
  /** Expected SHA-256 hex digest of the artifact bytes. */
  sha256: string;
  bytes: number;
  role: "visual-classifier" | "watermark-detector";
  inputSize: number;
  /** OpenSynthID uses 6-channel NCHW; leave 3 for RGB classifiers. */
  inputChannels?: number;
  /** Label index that corresponds to AI / fake (logits2 only). */
  aiLabelIndex: number;
  mean: readonly [number, number, number];
  std: readonly [number, number, number];
  /** ORT graphOptimizationLevel for this export. */
  graphOptimizationLevel: "disabled" | "all";
  preprocess: ModelPreprocess;
  outputKind: ModelOutputKind;
};

export const MODEL_CACHE_NAME = "neopixel-models-v5";

/**
 * Cache / package version. Bump when the required artifact set changes.
 * fp32 distilled is packaged for legacy / experiments; runtime prefers fp16
 * on WASM when the adapter lacks shader-f16 (never fp32-on-WebGPU).
 */
export const MODEL_VERSION =
  "distilled-fp16+fp32+neopixel-accurate-v1-q8+opensynthid-q8";

/**
 * Distilled ViT AI image detector (MIT), fp16 ONNX.
 * Source: https://huggingface.co/onnx-community/ai-image-detect-distilled-ONNX
 * Labels: 0=fake, 1=real
 *
 * WebGPU only when adapter has shader-f16 and passes a latency probe; else WASM.
 */
export const DISTILLED_MODEL = {
  id: "ai-image-detect-distilled",
  cacheKey: "models/ai-image-detect-distilled/model_fp16.onnx",
  localPath: "models/ai-image-detect-distilled/model_fp16.onnx",
  url: "https://huggingface.co/onnx-community/ai-image-detect-distilled-ONNX/resolve/main/onnx/model_fp16.onnx",
  sha256: "9594bacb70d9c65fcaa656e0d17038c5cac7a6c48d04cd10f2ebf972a01ba3ee",
  bytes: 29_273_325,
  role: "visual-classifier",
  inputSize: 224,
  aiLabelIndex: 0,
  mean: [0.5, 0.5, 0.5],
  std: [0.5, 0.5, 0.5],
  // "all" triggers SimplifiedLayerNormFusion bugs on this fp16 ViT with some
  // ORT builds (session create → GetIndexFromName / InsertedPrecisionFreeCast).
  graphOptimizationLevel: "disabled",
  preprocess: "stretch",
  outputKind: "logits2",
} as const satisfies ModelArtifact;

/**
 * Same distilled detector, fp32 ONNX.
 * Used on real WebGPU adapters that lack shader-f16, and only when a warm
 * timing probe beats WASM fp16 (software WebGPU must not win).
 */
export const DISTILLED_MODEL_FP32 = {
  id: "ai-image-detect-distilled-fp32",
  cacheKey: "models/ai-image-detect-distilled/model.onnx",
  localPath: "models/ai-image-detect-distilled/model.onnx",
  url: "https://huggingface.co/onnx-community/ai-image-detect-distilled-ONNX/resolve/main/onnx/model.onnx",
  sha256: "87b4331f22418a4cb50901851a1c28f64a0ca4f58728442d073b4bed9922ba86",
  bytes: 58_410_332,
  role: "visual-classifier",
  inputSize: 224,
  aiLabelIndex: 0,
  mean: [0.5, 0.5, 0.5],
  std: [0.5, 0.5, 0.5],
  graphOptimizationLevel: "disabled",
  preprocess: "stretch",
  outputKind: "logits2",
} as const satisfies ModelArtifact;

/**
 * NeoPixel accurate secondary head (MIT): OwensLab/commfor-model-384 backbone
 * + NeoPixel-trained 384→64→1 head, Q8 ONNX.
 * Built by `npm run distill:accurate` / `distill:loop`. Packaged only (no public URL).
 * Labels: single logit → sigmoid = P(AI).
 */
export const NEOPIXEL_ACCURATE_MODEL = {
  id: "neopixel-accurate-v1",
  cacheKey: "models/neopixel-accurate-v1/model_quantized.onnx",
  localPath: "models/neopixel-accurate-v1/model_quantized.onnx",
  url: "",
  sha256: "25ef06372b8e5eb5cb183a85d34cc3e9a670c47d6eb7a72c109d7107aa467b0e",
  bytes: 24_044_443,
  role: "visual-classifier",
  inputSize: 384,
  aiLabelIndex: 0,
  mean: [0.485, 0.456, 0.406],
  std: [0.229, 0.224, 0.225],
  graphOptimizationLevel: "disabled",
  preprocess: "short440-center384",
  outputKind: "logit",
} as const satisfies ModelArtifact;

/** Accurate secondary head used by cascade / dual inference. */
export const FORENSICS_MODEL = NEOPIXEL_ACCURATE_MODEL;

/**
 * OpenSynthID (Apache-2.0): community SynthID pixel-watermark surrogate.
 * Google DeepMind’s official detector is proprietary; this covers Gemini /
 * Imagen / OpenAI ChatGPT-image SynthID marks when C2PA Soft Binding is gone.
 * Built by `python3 scripts/convert-opensynthid.py` (then dynamic Q8).
 */
export const OPENSYNTHID_MODEL = {
  id: "opensynthid-detect",
  cacheKey: "models/opensynthid-detect/model_quantized.onnx",
  localPath: "models/opensynthid-detect/model_quantized.onnx",
  url: "",
  sha256: "d3801422608b2a0f7b51a08e7417946a0dc88c5d7879e5a6fb8fc4008aaf630f",
  bytes: 21_665_653,
  role: "watermark-detector",
  inputSize: 512,
  inputChannels: 6,
  aiLabelIndex: 0,
  mean: [0, 0, 0],
  std: [1, 1, 1],
  graphOptimizationLevel: "disabled",
  preprocess: "stretch",
  outputKind: "logit",
} as const satisfies ModelArtifact;

/** Primary model kept for backwards-compatible imports. */
export const VISUAL_MODEL = DISTILLED_MODEL;

/**
 * Artifacts that must be present in Cache Storage / packaged models/.
 * Includes both distilled precision variants so WebGPU can pick at runtime.
 */
export const ALL_MODELS: readonly ModelArtifact[] = [
  DISTILLED_MODEL,
  DISTILLED_MODEL_FP32,
  FORENSICS_MODEL,
  OPENSYNTHID_MODEL,
];

/**
 * Heads used by Node dual / cascade eval (fp16 distilled + accurate secondary).
 * Do not include the WebGPU-only fp32 distilled variant here.
 */
export const INFERENCE_MODELS: readonly ModelArtifact[] = [
  DISTILLED_MODEL,
  FORENSICS_MODEL,
];

export function isDistilledModelId(id: string): boolean {
  return id === DISTILLED_MODEL.id || id === DISTILLED_MODEL_FP32.id;
}
