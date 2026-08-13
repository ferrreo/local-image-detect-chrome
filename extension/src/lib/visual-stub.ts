import { asAiConfidence, type AiConfidence, type InferenceBackend } from "../shared/types";

export type VisualClassification = {
  /** Primary (distilled) AI probability. */
  score: AiConfidence;
  /** Optional secondary model AI probability (Community Forensics). */
  secondaryScore?: AiConfidence;
  backend: InferenceBackend;
  detail: string;
  distilledMs?: number;
  forensicsMs?: number;
  preprocessMs?: number;
};

function clamp(v: number, lo: number, hi: number): number {
  return Math.min(hi, Math.max(lo, v));
}

/**
 * Deterministic stub used by integration tests and offline CI without weights.
 * High spatial luminance variance ⇒ real-like; smooth fields ⇒ AI-like.
 */
export function stubVisualClassify(imageData: ImageData): VisualClassification {
  const { data, width, height } = imageData;
  let lumaSum = 0;
  let lumaSq = 0;
  let chromaSum = 0;
  let samples = 0;

  for (let y = 0; y < height; y += 2) {
    for (let x = 0; x < width; x += 2) {
      const p = (y * width + x) * 4;
      const r = data[p] ?? 0;
      const g = data[p + 1] ?? 0;
      const b = data[p + 2] ?? 0;
      const luma = 0.299 * r + 0.587 * g + 0.114 * b;
      lumaSum += luma;
      lumaSq += luma * luma;
      chromaSum += (Math.max(r, g, b) - Math.min(r, g, b)) / 255;
      samples += 1;
    }
  }

  const mean = samples === 0 ? 0 : lumaSum / samples;
  const variance = samples === 0 ? 0 : Math.max(0, lumaSq / samples - mean * mean);
  const chroma = samples === 0 ? 0 : chromaSum / samples;

  // Texture dominates: smooth synthetic fields score AI-like, speckled fields real-like.
  const texture = clamp(variance / 2200, 0, 1);
  const colorfulness = clamp(chroma / 0.55, 0, 1);
  const realEvidence = 0.85 * texture + 0.15 * colorfulness;
  const score = asAiConfidence(clamp(0.9 - realEvidence * 0.8, 0.05, 0.95));

  return {
    score,
    backend: { kind: "stub" },
    detail: `stub:var=${variance.toFixed(1)},chroma=${chroma.toFixed(3)}`,
  };
}
