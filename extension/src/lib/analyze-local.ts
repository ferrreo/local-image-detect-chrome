import { analyzeProvenance } from "./provenance";
import { analyzeSpectral } from "./spectral";
import { stubVisualClassify } from "./visual-stub";
import {
  decodeImageBytes,
  guessMimeType,
  rasterizeForSpectral,
  rasterizeForModel,
} from "./image-decode";
import { fuseDetection } from "./fusion";
import { VISUAL_MODEL } from "./model-manifest";
import {
  asAiConfidence,
  type DetectionResult,
} from "../shared/types";

/**
 * Service-worker-safe analysis path (no onnxruntime).
 * Used for stub inference and as a fallback when the offscreen document is unavailable.
 */
export async function analyzeLocalStub(args: {
  imageId: string;
  bytes: ArrayBuffer;
  mimeType?: string;
  threshold?: number;
  realThreshold?: number;
}): Promise<DetectionResult> {
  const started = performance.now();
  const bytesView = new Uint8Array(args.bytes);
  const mimeType = args.mimeType ?? guessMimeType(bytesView);
  const provenance = analyzeProvenance(bytesView);

  if (provenance.shortCircuit) {
    const fused = fuseDetection({
      provenance: {
        tier: "provenance",
        aiScore: provenance.score,
        weight: 0.08,
        detail: provenance.detail,
        shortCircuit: true,
      },
      spectral: {
        tier: "spectral",
        aiScore: asAiConfidence(0.5),
        weight: 0.2,
        detail: "skipped",
      },
      visual: {
        tier: "visual",
        aiScore: asAiConfidence(0.5),
        weight: 0.72,
        detail: "skipped",
      },
      ...(args.threshold !== undefined ? { threshold: args.threshold } : {}),
      ...(args.realThreshold !== undefined
        ? { realThreshold: args.realThreshold }
        : {}),
    });
    return {
      imageId: args.imageId,
      label: fused.label,
      confidence: fused.confidence,
      tiers: fused.tiers,
      backend: { kind: "none" },
      elapsedMs: performance.now() - started,
    };
  }

  const decoded = await decodeImageBytes(args.bytes, mimeType);
  try {
    const spectralImage = await rasterizeForSpectral(decoded.bitmap);
    const spectral = analyzeSpectral(spectralImage);
    const modelImage = await rasterizeForModel(
      decoded.bitmap,
      VISUAL_MODEL.inputSize,
    );
    const visual = stubVisualClassify(modelImage);
    const fused = fuseDetection({
      provenance: {
        tier: "provenance",
        aiScore: provenance.score,
        weight: 0.08,
        detail: provenance.detail,
      },
      spectral: {
        tier: "spectral",
        aiScore: spectral.score,
        weight: 0.2,
        detail: spectral.detail,
      },
      visual: {
        tier: "visual",
        aiScore: visual.score,
        weight: 0.72,
        detail: visual.detail,
      },
      spectralFeatures: spectral.features,
      ...(args.threshold !== undefined ? { threshold: args.threshold } : {}),
      ...(args.realThreshold !== undefined
        ? { realThreshold: args.realThreshold }
        : {}),
    });
    return {
      imageId: args.imageId,
      label: fused.label,
      confidence: fused.confidence,
      tiers: fused.tiers,
      backend: { kind: "stub" },
      elapsedMs: performance.now() - started,
    };
  } finally {
    decoded.bitmap.close();
  }
}
