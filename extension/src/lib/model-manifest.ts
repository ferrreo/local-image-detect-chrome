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

/** Models required by the extension runtime (Cache Storage). */
export const MODEL_VERSION = "distilled-fp16+community-forensics-q4-v1";

/**
 * Distilled ViT AI image detector (MIT).
 * Source: https://huggingface.co/onnx-community/ai-image-detect-distilled-ONNX
 * Labels: 0=fake, 1=real
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
 * Community Forensics ViT-Small detector (MIT), q4 ONNX.
 * Source: https://huggingface.co/onnx-community/CommunityForensics-DeepfakeDet-ViT-ONNX
 * Labels: softmax index 1 treated as AI/fake for this export.
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

/** Required for browser Cache Storage / extension readiness. */
export const ALL_MODELS: readonly ModelArtifact[] = [
  DISTILLED_MODEL,
  FORENSICS_MODEL,
];
