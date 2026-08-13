/**
 * One-time setup downloads only these publicly available weights.
 * After caching, inference never fetches models again.
 */
export type ModelArtifact = {
  id: string;
  /** Relative path under the Cache Storage namespace. */
  cacheKey: string;
  /** Local path used by Node eval (onnxruntime-node). */
  localPath: string;
  /** Public Hugging Face URL (resolved at setup time only). */
  url: string;
  /** Expected SHA-256 hex digest of the artifact bytes. */
  sha256: string;
  bytes: number;
  role: "visual-classifier";
  inputSize: number;
  /** Label index that corresponds to AI / fake. */
  aiLabelIndex: number;
  mean: readonly [number, number, number];
  std: readonly [number, number, number];
  /** ORT graphOptimizationLevel for this export. */
  graphOptimizationLevel: "disabled" | "all";
};

export const MODEL_CACHE_NAME = "truepixel-models-v1";

/**
 * Cache / package version. Bump when the required artifact set changes.
 * fp32 distilled is required for WebGPU adapters without shader-f16.
 */
export const MODEL_VERSION =
  "distilled-fp16+fp32+community-forensics-q4-v1";

/**
 * Distilled ViT AI image detector (MIT), fp16 ONNX.
 * Source: https://huggingface.co/onnx-community/ai-image-detect-distilled-ONNX
 * Labels: 0=fake, 1=real
 *
 * Prefer on WASM, and on WebGPU when the adapter supports shader-f16.
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
  graphOptimizationLevel: "disabled",
} as const satisfies ModelArtifact;

/**
 * Same distilled detector, fp32 ONNX.
 * Required for WebGPU when the adapter lacks shader-f16 (fp16 Transpose fails).
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
} as const satisfies ModelArtifact;

/**
 * Community Forensics ViT-Small detector (MIT), q4 ONNX.
 * Source: https://huggingface.co/onnx-community/CommunityForensics-DeepfakeDet-ViT-ONNX
 * Labels: softmax index 1 treated as AI/fake for this export.
 *
 * Browser WebGPU EP hangs / OOMs on this export — always run on WASM in-browser.
 */
export const FORENSICS_MODEL = {
  id: "community-forensics-deepfake-det",
  cacheKey: "models/community-forensics/model_q4.onnx",
  localPath: "models/community-forensics/model_q4.onnx",
  url: "https://huggingface.co/onnx-community/CommunityForensics-DeepfakeDet-ViT-ONNX/resolve/main/onnx/model_q4.onnx",
  sha256: "263c46052167a15b981848465b8adb9f28dbd1f9ad8ecf8157cb05d876f7091b",
  bytes: 24_416_892,
  role: "visual-classifier",
  inputSize: 384,
  aiLabelIndex: 1,
  mean: [0.5, 0.5, 0.5],
  std: [0.5, 0.5, 0.5],
  graphOptimizationLevel: "all",
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
 * Heads used by Node dual / cascade eval (fp16 distilled + forensics).
 * Do not include the WebGPU-only fp32 distilled variant here.
 */
export const INFERENCE_MODELS: readonly ModelArtifact[] = [
  DISTILLED_MODEL,
  FORENSICS_MODEL,
];

export function isDistilledModelId(id: string): boolean {
  return id === DISTILLED_MODEL.id || id === DISTILLED_MODEL_FP32.id;
}
