/**
 * One-time setup downloads only these publicly available weights.
 * After caching, inference never fetches models again.
 * Proofmark is packaged-only (no public HF URL in this repo).
 */
export type ModelPreprocess = "stretch" | "short440-center384";
export type ModelOutputKind = "logits2" | "logit";

export type ModelArtifact = {
  id: string;
  /** Relative path under the Cache Storage namespace. */
  cacheKey: string;
  /** Local path used by Node eval (onnxruntime-node). */
  localPath: string;
  /**
   * Public Hugging Face URL (resolved at setup time only).
   * Empty when the artifact must be seeded from a packaged/vendored file.
   */
  url: string;
  /** Expected SHA-256 hex digest of the artifact bytes. */
  sha256: string;
  bytes: number;
  role: "visual-classifier";
  inputSize: number;
  /** Label index that corresponds to AI / fake (logits2 only). */
  aiLabelIndex: number;
  mean: readonly [number, number, number];
  std: readonly [number, number, number];
  /** ORT graphOptimizationLevel for this export. */
  graphOptimizationLevel: "disabled" | "all";
  preprocess: ModelPreprocess;
  outputKind: ModelOutputKind;
};

export const MODEL_CACHE_NAME = "truepixel-models-v2";

/**
 * Cache / package version. Bump when the required artifact set changes.
 * fp32 distilled is packaged for legacy / experiments; runtime prefers fp16
 * on WASM when the adapter lacks shader-f16 (never fp32-on-WebGPU).
 */
export const MODEL_VERSION =
  "distilled-fp16+fp32+proofmark-webwild-v3-q8-v1";

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
 * Proofmark webwild-v3 Q8 (accurate secondary head).
 * Backbone: OwensLab/commfor-model-384; head fine-tune shipped by
 * Dyno-man/Dino-ImageGen-Ext. HF repo `Proofmark/proofmark-webwild-v3` is
 * private (401); public bytes match the GitHub-bundled ONNX below.
 * Preprocess: shortest side → 440, center-crop 384, ImageNet norm, logit+sigmoid.
 *
 * Zig+ORT WASM still hardcodes stretch/0.5 — pipeline prefers ort-web while
 * this head is active.
 */
export const FORENSICS_MODEL = {
  id: "proofmark-webwild-v3",
  cacheKey: "models/proofmark-webwild-v3/model_quantized.onnx",
  localPath: "models/proofmark-webwild-v3/model_quantized.onnx",
  url: "https://raw.githubusercontent.com/Dyno-man/Dino-ImageGen-Ext/main/public/models/Proofmark/proofmark-webwild-v3/onnx/model_quantized.onnx",
  sha256: "ed17ceb332bef84d0adcc2fa537eef85ed3ac6fb32c30393c326321fbbe54683",
  bytes: 24_031_833,
  role: "visual-classifier",
  inputSize: 384,
  aiLabelIndex: 0,
  mean: [0.485, 0.456, 0.406],
  std: [0.229, 0.224, 0.225],
  graphOptimizationLevel: "disabled",
  preprocess: "short440-center384",
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

/** True when the accurate head needs JS preprocess (not Zig stretch/0.5). */
export function accurateHeadNeedsOrtWeb(): boolean {
  const preprocess: string = FORENSICS_MODEL.preprocess;
  const outputKind: string = FORENSICS_MODEL.outputKind;
  return preprocess !== "stretch" || outputKind !== "logits2";
}
