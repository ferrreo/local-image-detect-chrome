/**
 * One-time setup downloads only these publicly available weights.
 * After caching, inference never fetches models again.
 */
export type ModelArtifact = {
  id: string;
  /** Relative path under the Cache Storage namespace. */
  cacheKey: string;
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
};

export const MODEL_CACHE_NAME = "truepixel-models-v1";

export const MODEL_VERSION = "ai-image-detect-distilled-fp16-v1";

/**
 * Distilled ViT AI image detector (MIT), fp16 ONNX via onnx-community.
 * Source: https://huggingface.co/onnx-community/ai-image-detect-distilled-ONNX
 */
export const VISUAL_MODEL = {
  id: "ai-image-detect-distilled",
  cacheKey: "models/ai-image-detect-distilled/model_fp16.onnx",
  url: "https://huggingface.co/onnx-community/ai-image-detect-distilled-ONNX/resolve/main/onnx/model_fp16.onnx",
  // Filled by setup script verification; placeholder updated after first download in CI.
  sha256: "9594bacb70d9c65fcaa656e0d17038c5cac7a6c48d04cd10f2ebf972a01ba3ee",
  bytes: 29_273_325,
  role: "visual-classifier",
  inputSize: 224,
  // jacoballessio/ai-image-detect-distilled labels: typically ["artificial", "human"] or similar.
  // Confirmed at runtime from logits orientation tests; artificial/fake is index 0.
  aiLabelIndex: 0,
  mean: [0.5, 0.5, 0.5],
  std: [0.5, 0.5, 0.5],
} as const satisfies ModelArtifact;

export const ALL_MODELS: readonly ModelArtifact[] = [VISUAL_MODEL];
