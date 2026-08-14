import { analyzeProvenance } from "./provenance";
import { analyzeVisibleWatermark } from "./visible-watermark";
import { analyzeSynthIdPixels } from "./synthid-detect";
import {
  analyzeSpectral,
  looksLikeNonPhotoGraphic,
  looksPhotographic,
} from "./spectral";
import { classifyVisual } from "./visual-classifier";
import {
  decodeImageBytes,
  guessMimeType,
  rasterizeForSpectral,
} from "./image-decode";
import { fuseDetection } from "./fusion";
import { needsForensicsCascade } from "./forensics-cascade";
import {
  asAiConfidence,
  type AiConfidence,
  type DetectionResult,
  type InferenceBackend,
  type PipelineTiming,
} from "../shared/types";

export type PipelineOptions = {
  imageId: string;
  bytes: ArrayBuffer;
  mimeType?: string;
  threshold?: number;
  realThreshold?: number;
  stubVisual?: boolean;
  /**
   * Distilled → optional accurate secondary head (default true).
   * Overridden by `speedMode: "realtime"` (cascade off).
   */
  cascade?: boolean;
  /** Interactive overlay uses `realtime`; eval leaves unset / `accurate`. */
  speedMode?: "accurate" | "realtime";
  /** Skip OpenSynthID pixel pass (tests / stub). */
  skipSynthId?: boolean;
};

/**
 * Image-only detection pipeline:
 * byte watermarks → visible marks → SynthID pixels → spectral → visual.
 */
export async function detectAiImage(
  options: PipelineOptions,
): Promise<DetectionResult> {
  const started = performance.now();
  const bytesView = new Uint8Array(options.bytes);
  const mimeType = options.mimeType ?? guessMimeType(bytesView);
  const cascade =
    options.speedMode === "realtime"
      ? false
      : options.cascade !== false;
  // OpenSynthID is heavy; run on accurate refine (and always-dual eval).
  const runSynthId =
    !options.skipSynthId &&
    options.stubVisual !== true &&
    options.speedMode !== "realtime";

  let decodeMs = 0;
  let spectralMs = 0;
  let preprocessMs = 0;
  let distilledMs = 0;
  let forensicsMs = 0;
  let ranForensics = false;
  let sourceMinSide = 0;

  let provenance = analyzeProvenance(bytesView);

  let spectralScore = asAiConfidence(0.5);
  let spectralDetail = "skipped";
  let spectralFeatures: ReturnType<typeof analyzeSpectral>["features"] | undefined;
  let visualScore = asAiConfidence(0.5);
  let visualSecondary: AiConfidence | undefined;
  let visualDetail = "skipped";
  let backend: InferenceBackend = { kind: "none" };

  if (!provenance.shortCircuit) {
    const tDecode = performance.now();
    const decoded = await decodeImageBytes(options.bytes, mimeType);
    decodeMs = performance.now() - tDecode;
    sourceMinSide = Math.min(decoded.bitmap.width, decoded.bitmap.height);
    try {
      const watermarkRaster = await rasterizeForSpectral(
        decoded.bitmap,
        Math.min(768, Math.max(sourceMinSide, 256)),
      );
      const visible = analyzeVisibleWatermark(watermarkRaster);
      if (visible.shortCircuit) {
        provenance = visible;
      } else if (runSynthId && sourceMinSide >= 96) {
        const synth = await analyzeSynthIdPixels(watermarkRaster);
        if (synth.shortCircuit) {
          provenance = synth;
        }
      }

      if (!provenance.shortCircuit) {
        const spectralPromise = (async () => {
          const t0 = performance.now();
          const spectralImage = await rasterizeForSpectral(decoded.bitmap);
          const result = analyzeSpectral(spectralImage);
          spectralMs = performance.now() - t0;
          return result;
        })();
        const distilledPromise =
          options.stubVisual === true
            ? classifyVisual(decoded.bitmap, { stub: true, cascade: false })
            : classifyVisual(decoded.bitmap, {
                cascade: false,
                runDistilled: true,
                runForensics: false,
              });

        const [spectral, distilledVisual] = await Promise.all([
          spectralPromise,
          distilledPromise,
        ]);
        distilledMs = distilledVisual.distilledMs ?? 0;
        preprocessMs = distilledVisual.preprocessMs ?? 0;

        spectralScore = spectral.score;
        spectralDetail = spectral.detail;
        spectralFeatures = spectral.features;
        visualScore = distilledVisual.score;
        visualDetail = `${distilledVisual.detail},engine=ort-web`;
        backend = distilledVisual.backend;

        if (
          cascade &&
          !options.stubVisual &&
          spectralFeatures &&
          (needsForensicsCascade({
            distilled: visualScore,
            spectral: spectralScore,
            laplacianVariance: spectralFeatures.laplacianVariance,
            chromaFlatness: spectralFeatures.chromaFlatness,
          }) ||
            sourceMinSide < 256 ||
            !looksPhotographic(spectralFeatures) ||
            looksLikeNonPhotoGraphic(spectralFeatures))
        ) {
          const tCf = performance.now();
          const forensicsOnly = await classifyVisual(decoded.bitmap, {
            cascade: false,
            runDistilled: false,
            runForensics: true,
          });
          forensicsMs = performance.now() - tCf;
          ranForensics = true;
          visualSecondary =
            forensicsOnly.secondaryScore ?? forensicsOnly.score;
          visualDetail = `${visualDetail};${forensicsOnly.detail}`;
          backend = forensicsOnly.backend;
        }
      }
    } finally {
      decoded.bitmap.close();
    }
  }

  const provenanceTier = {
    tier: "provenance" as const,
    aiScore: provenance.score,
    weight: 0.08,
    detail: provenance.detail,
    shortCircuit: provenance.shortCircuit,
  };

  const tFuse = performance.now();
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
    ...(visualSecondary !== undefined
      ? {
          visualSecondary: {
            tier: "visual" as const,
            aiScore: visualSecondary,
            weight: 0.5,
            detail: "accurate-head",
          },
        }
      : {}),
    ...(spectralFeatures !== undefined ? { spectralFeatures } : {}),
    ...(options.threshold !== undefined ? { threshold: options.threshold } : {}),
    ...(options.realThreshold !== undefined
      ? { realThreshold: options.realThreshold }
      : {}),
    ...(sourceMinSide > 0 ? { sourceMinSide } : {}),
  });
  const fuseMs = performance.now() - tFuse;
  const totalMs = performance.now() - started;

  const timing: PipelineTiming = {
    decodeMs,
    spectralMs,
    preprocessMs,
    distilledMs,
    forensicsMs,
    fuseMs,
    totalMs,
    ranForensics,
  };

  return {
    imageId: options.imageId,
    label: fused.label,
    confidence: fused.confidence,
    tiers: fused.tiers,
    backend,
    elapsedMs: totalMs,
    timing,
  };
}
