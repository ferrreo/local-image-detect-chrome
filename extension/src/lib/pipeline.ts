import { analyzeProvenance } from "./provenance";
import { analyzeSpectral } from "./spectral";
import { classifyVisual } from "./visual-classifier";
import {
  classifyVisualZigWasm,
  isZigWasmOrtReady,
} from "./visual-zig-wasm";
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

export type VisualEngineKind = "onnxruntime-web" | "zig-wasm";

export type PipelineOptions = {
  imageId: string;
  bytes: ArrayBuffer;
  mimeType?: string;
  threshold?: number;
  stubVisual?: boolean;
  /**
   * Distilled → optional Community Forensics (default true).
   * Applies to onnxruntime-web and Zig+ORT WASM visual backends.
   * Overridden by `speedMode: "realtime"` (cascade off).
   */
  cascade?: boolean;
  /** Prefer Zig+ORT WASM when linked; otherwise onnxruntime-web. */
  visualEngine?: VisualEngineKind | "auto";
  /** Interactive overlay uses `realtime`; eval leaves unset / `accurate`. */
  speedMode?: "accurate" | "realtime";
};

/**
 * Full local detection pipeline:
 * provenance → spectral → visual (cascade dual) → fusion.
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

  let decodeMs = 0;
  let spectralMs = 0;
  let preprocessMs = 0;
  let distilledMs = 0;
  let forensicsMs = 0;
  let ranForensics = false;

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
  let spectralFeatures: ReturnType<typeof analyzeSpectral>["features"] | undefined;
  let visualScore = asAiConfidence(0.5);
  let visualSecondary: AiConfidence | undefined;
  let visualDetail = "skipped";
  let backend: InferenceBackend = { kind: "none" };

  if (!provenance.shortCircuit) {
    const tDecode = performance.now();
    const decoded = await decodeImageBytes(options.bytes, mimeType);
    decodeMs = performance.now() - tDecode;
    try {
      const prefer =
        options.visualEngine ??
        (typeof globalThis !== "undefined" &&
        (globalThis as { TRUEPIXEL_VISUAL_ENGINE?: string })
          .TRUEPIXEL_VISUAL_ENGINE === "zig-wasm"
          ? "zig-wasm"
          : "auto");
      const useZig =
        prefer === "zig-wasm" ||
        (prefer === "auto" && (await isZigWasmOrtReady()));
      const classify = useZig ? classifyVisualZigWasm : classifyVisual;

      // Overlap spectral CPU with distilled infer — gate CF after both finish.
      const spectralPromise = (async () => {
        const t0 = performance.now();
        const spectralImage = await rasterizeForSpectral(decoded.bitmap);
        const result = analyzeSpectral(spectralImage);
        spectralMs = performance.now() - t0;
        return result;
      })();
      const distilledPromise =
        options.stubVisual === true
          ? classify(decoded.bitmap, { stub: true, cascade: false })
          : classify(decoded.bitmap, {
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
      visualDetail = `${distilledVisual.detail}${useZig ? ",engine=zig-ort-wasm" : ",engine=ort-web"}`;
      backend = distilledVisual.backend;

      if (
        cascade &&
        !options.stubVisual &&
        spectralFeatures &&
        needsForensicsCascade({
          distilled: visualScore,
          spectral: spectralScore,
          laplacianVariance: spectralFeatures.laplacianVariance,
          chromaFlatness: spectralFeatures.chromaFlatness,
        })
      ) {
        const tCf = performance.now();
        const forensicsOnly = await classify(decoded.bitmap, {
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
    } finally {
      decoded.bitmap.close();
    }
  } else {
    backend = { kind: "none" };
  }

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
            detail: "community-forensics",
          },
        }
      : {}),
    ...(spectralFeatures !== undefined ? { spectralFeatures } : {}),
    ...(options.threshold !== undefined ? { threshold: options.threshold } : {}),
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
