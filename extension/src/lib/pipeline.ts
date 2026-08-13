import { analyzeProvenance } from "./provenance";
import { analyzeSpectral } from "./spectral";
import { classifyVisual } from "./visual-classifier";
import {
  decodeImageBytes,
  guessMimeType,
  rasterizeForSpectral,
} from "./image-decode";
import { fuseDetection } from "./fusion";
import {
  asAiConfidence,
  type DetectionResult,
  type InferenceBackend,
} from "../shared/types";

export type PipelineOptions = {
  imageId: string;
  bytes: ArrayBuffer;
  mimeType?: string;
  threshold?: number;
  stubVisual?: boolean;
};

/**
 * Full local detection pipeline:
 * provenance (bytes) → spectral forensics → visual ONNX (WebGPU/WASM) → fusion.
 */
export async function detectAiImage(
  options: PipelineOptions,
): Promise<DetectionResult> {
  const started = performance.now();
  const bytesView = new Uint8Array(options.bytes);
  const mimeType = options.mimeType ?? guessMimeType(bytesView);

  const provenance = analyzeProvenance(bytesView);
  const provenanceTier = {
    tier: "provenance" as const,
    aiScore: provenance.score,
    weight: 0.08,
    detail: provenance.detail,
    shortCircuit: provenance.shortCircuit,
  };

  let spectralScore = asAiConfidence(0.5);
  let spectralDetail = "skipped";
  let visualScore = asAiConfidence(0.5);
  let visualDetail = "skipped";
  let backend: InferenceBackend = { kind: "none" };

  if (!provenance.shortCircuit) {
    const decoded = await decodeImageBytes(options.bytes, mimeType);
    try {
      const spectralImage = await rasterizeForSpectral(decoded.bitmap);
      const spectral = analyzeSpectral(spectralImage);
      spectralScore = spectral.score;
      spectralDetail = spectral.detail;

      const visual = await classifyVisual(decoded.bitmap, {
        stub: options.stubVisual === true,
      });
      visualScore = visual.score;
      visualDetail = visual.detail;
      backend = visual.backend;
    } finally {
      decoded.bitmap.close();
    }
  } else {
    backend = { kind: "none" };
  }

  const fused = fuseDetection({
    provenance: provenanceTier,
    spectral: {
      tier: "spectral",
      aiScore: spectralScore,
      weight: 0.2,
      detail: spectralDetail,
    },
    visual: {
      tier: "visual",
      aiScore: visualScore,
      weight: 0.72,
      detail: visualDetail,
    },
    ...(options.threshold !== undefined ? { threshold: options.threshold } : {}),
  });

  return {
    imageId: options.imageId,
    label: fused.label,
    confidence: fused.confidence,
    tiers: fused.tiers,
    backend,
    elapsedMs: performance.now() - started,
  };
}
