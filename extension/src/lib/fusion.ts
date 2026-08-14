import {
  asAiConfidence,
  AI_LABEL_THRESHOLD,
  EVAL_CONFIDENCE_THRESHOLD,
  REAL_LABEL_THRESHOLD,
  type AiConfidence,
  type DetectionLabel,
  type TierSignal,
} from "../shared/types";
import { aiThresholdBumpForSourceSide } from "./best-image-url";
import {
  looksLikeNeonAiSubject,
  looksLikeNonPhotoGraphic,
  looksLikeSyntheticCgi,
  looksPhotographic,
} from "./spectral";

export type SpectralFusionFeatures = {
  highFreqEnergyRatio: number;
  laplacianVariance: number;
  chromaFlatness: number;
  blockiness: number;
  axisAlignedEdgeRatio: number;
  quantizedColorCount: number;
  topColorShare: number;
  frameAxisAlignedEdgeRatio?: number;
  frameTopColorShare?: number;
  frameQuantizedColorCount?: number;
  windowChromeScore?: number;
};

export type FusionInput = {
  provenance: TierSignal;
  spectral: TierSignal;
  visual: TierSignal;
  /** Optional secondary visual model (accurate head). */
  visualSecondary?: TierSignal;
  spectralFeatures?: SpectralFusionFeatures;
  /** AI label threshold (product default 69.51%; eval often passes 0.65). */
  threshold?: number;
  /** Real label threshold (product default 40.99%). */
  realThreshold?: number;
  /**
   * Min(decoded width, height). When the asset is smaller than the model
   * input we raise the AI floor — upscaling looks synthetic.
   */
  sourceMinSide?: number;
};

export type FusionOutput = {
  confidence: AiConfidence;
  label: DetectionLabel;
  tiers: TierSignal[];
};

/** Cap for mild dual-head holds — stays under the product AI label floor. */
const MILD_HOLD_CAP = 0.69;

/**
 * Calibrated fusion for product labels (AI ≥ 69.51%, real ≤ 40.99%) while
 * remaining usable at the bounty 65% eval point.
 * Distilled is primary; Community Forensics recovers StyleGAN-class misses
 * only when decisive — mild dual agreement was FPing real photos at ~72–81%.
 */
export function fuseDetection(input: FusionInput): FusionOutput {
  const baseThreshold = input.threshold ?? AI_LABEL_THRESHOLD;
  const bump = aiThresholdBumpForSourceSide(input.sourceMinSide ?? 4096);
  const threshold = Math.min(0.97, baseThreshold + bump);
  const realThreshold = input.realThreshold ?? REAL_LABEL_THRESHOLD;
  const tiers = [
    input.provenance,
    input.spectral,
    input.visual,
    ...(input.visualSecondary ? [input.visualSecondary] : []),
  ];

  if (input.provenance.shortCircuit) {
    const confidence = input.provenance.aiScore;
    return {
      confidence,
      label: labelFor(confidence, threshold, realThreshold),
      tiers: [
        ...tiers,
        {
          tier: "fusion",
          aiScore: confidence,
          weight: 1,
          detail: "provenance-short-circuit",
        },
      ],
    };
  }

  const distilled = input.visual.aiScore;
  const forensics = input.visualSecondary?.aiScore;
  const spectral = input.spectral.aiScore;
  const feats = input.spectralFeatures;
  const smallSource = (input.sourceMinSide ?? 4096) < 224;
  // Small-src bump raises the floor for distilled-alone. Accurate-head still
  // promotes against the product floor — otherwise Google Images thumbs
  // (forensics ~70–84%) were held to Real/uncertain forever.
  const accurateFloor = baseThreshold;
  const visualPeak = Math.max(distilled, forensics ?? 0);

  let confidence: AiConfidence;
  let detail: string;
  let labelThreshold = threshold;
  const baseline = calibrate(
    asAiConfidence(0.78 * distilled + 0.22 * spectral),
  );

  if (
    // Upscaled thumbs: don't let distilled-alone paint AI — unless the
    // pixels themselves look like synthetic CGI / soft non-photo AI slop.
    smallSource &&
    distilled < 0.92 &&
    (forensics === undefined || forensics < accurateFloor)
  ) {
    const softPeak = Math.max(spectral, distilled, forensics ?? 0);
    if (
      feats &&
      (looksLikeNeonAiSubject(feats) || looksLikeSyntheticCgi(feats))
    ) {
      confidence = asAiConfidence(
        Math.max(
          accurateFloor,
          Math.min(0.95, 0.7 + 0.2 * softPeak),
        ),
      );
      detail = `small-cgi-ai(side=${input.sourceMinSide})`;
      labelThreshold = accurateFloor;
    } else if (
      feats &&
      !looksPhotographic(feats) &&
      !looksLikeNonPhotoGraphic(feats) &&
      softPeak >= 0.28 &&
      feats.chromaFlatness >= 0.55 &&
      feats.highFreqEnergyRatio <= 0.5
    ) {
      // Mushy Google thumbs of AI stock / robots: not camera, not UI chrome.
      confidence = asAiConfidence(
        Math.max(
          accurateFloor,
          Math.min(0.93, Math.max(0.72, softPeak + 0.2)),
        ),
      );
      detail = `small-nonphoto-ai(side=${input.sourceMinSide})`;
      labelThreshold = accurateFloor;
    } else {
      const held = Math.min(baseline, MILD_HOLD_CAP);
      confidence = asAiConfidence(
        held <= realThreshold
          ? Math.min(MILD_HOLD_CAP, realThreshold + 0.02)
          : held,
      );
      detail = `small-source-hold(side=${input.sourceMinSide},thr=${threshold.toFixed(2)})`;
    }
  } else if (
    // Structured digital graphics (UI / charts / laptop screens / code cards)
    // BEFORE neon promote — otherwise maxed distilled (AI 98% on KPI cards)
    // skips past narrow holds that miss busy legends or laptop bezels.
    feats &&
    looksLikeNonPhotoGraphic(feats) &&
    !looksLikeNeonAiSubject(feats)
  ) {
    const held = Math.min(baseline, MILD_HOLD_CAP);
    confidence = asAiConfidence(
      held <= realThreshold
        ? Math.min(MILD_HOLD_CAP, realThreshold + 0.05)
        : held,
    );
    detail = `digital-graphic-hold(axis=${feats.axisAlignedEdgeRatio.toFixed(2)},colors=${feats.quantizedColorCount},flat=${feats.chromaFlatness.toFixed(2)},share=${feats.topColorShare.toFixed(2)})`;
  } else if (
    // Neon CGI subjects — after UI/chart holds so toast/chart FPs win.
    feats &&
    looksLikeNeonAiSubject(feats) &&
    distilled < 0.92 &&
    (forensics === undefined || forensics < accurateFloor)
  ) {
    confidence = asAiConfidence(
      Math.max(
        accurateFloor,
        Math.min(0.95, Math.max(0.72, distilled, spectral, forensics ?? 0)),
      ),
    );
    detail = "neon-ai-subject";
    labelThreshold = accurateFloor;
  } else if (
    // Both heads elevated but not decisive — real indoor / product / group
    // photos land here (observed AI 72% and AI 81% FPs).
    // Keep distilled < 0.6 free for StyleGAN / TPDNE CF recovery.
    // Skip synthetic CGI / smooth generative faces (chromaFlatness high).
    forensics !== undefined &&
    distilled >= 0.6 &&
    distilled < 0.88 &&
    forensics >= 0.55 &&
    forensics < 0.88 &&
    feats &&
    looksPhotographic(feats) &&
    !looksLikeSyntheticCgi(feats) &&
    feats.chromaFlatness < 0.54
  ) {
    confidence = asAiConfidence(Math.min(baseline, MILD_HOLD_CAP));
    detail = "dual-mild-hold";
  } else if (
    // Busy natural scenes with only a strong distilled head.
    feats &&
    looksPhotographic(feats) &&
    feats.laplacianVariance >= 900 &&
    distilled >= 0.6 &&
    distilled < 0.9 &&
    (forensics === undefined || forensics < 0.88) &&
    !looksLikeSyntheticCgi(feats) &&
    feats.chromaFlatness < 0.54
  ) {
    confidence = asAiConfidence(Math.min(baseline, MILD_HOLD_CAP));
    detail = "busy-scene-hold";
  } else if (
    distilled >= 0.88 &&
    spectral <= 0.4 &&
    !(feats && looksLikeNonPhotoGraphic(feats))
  ) {
    confidence = asAiConfidence(Math.max(distilled, threshold));
    detail = "distilled-near-threshold";
  } else if (
    // Real camera macros (circuit boards, fabric, foliage): accurate head
    // often FPs while spectral still looks photographic and distilled is soft.
    // Keep this narrow — mid accurate scores on photoreal generators must not
    // be held, and mushy thumbs never get the camera-macro benefit of the doubt.
    feats &&
    !smallSource &&
    forensics !== undefined &&
    forensics >= 0.92 &&
    distilled < 0.4 &&
    spectral < 0.4 &&
    looksPhotographic(feats)
  ) {
    confidence = asAiConfidence(Math.min(baseline, MILD_HOLD_CAP));
    detail = "photo-evidence-hold";
  } else if (
    // Proofmark-class accurate head: trust it when it clears the AI floor
    // even if distilled is near-zero (Lexica / modern generator misses).
    // Never override structured digital graphics (chart/laptop FPs).
    forensics !== undefined &&
    forensics >= accurateFloor &&
    distilled < 0.88 &&
    !(feats && looksLikeNonPhotoGraphic(feats))
  ) {
    confidence = asAiConfidence(Math.max(forensics, accurateFloor));
    detail = "accurate-head";
    labelThreshold = accurateFloor;
  } else if (
    // Near-floor accurate on non-photo frames — common on AI stock thumbs.
    forensics !== undefined &&
    forensics >= 0.58 &&
    forensics < accurateFloor &&
    distilled < 0.88 &&
    feats &&
    !looksPhotographic(feats) &&
    !looksLikeNonPhotoGraphic(feats)
  ) {
    confidence = asAiConfidence(accurateFloor);
    detail = "accurate-near-floor";
    labelThreshold = accurateFloor;
  } else if (
    forensics !== undefined &&
    feats &&
    forensics >= 0.88 &&
    distilled >= 0.3 &&
    feats.laplacianVariance >= 580 &&
    feats.chromaFlatness >= 0.34 &&
    feats.chromaFlatness <= 0.7 &&
    !looksLikeNonPhotoGraphic(feats)
  ) {
    confidence = asAiConfidence(Math.max(forensics, threshold));
    detail = "forensics-flatness-band";
  } else if (
    forensics !== undefined &&
    feats &&
    distilled >= 0.62 &&
    forensics >= 0.88 &&
    feats.chromaFlatness >= 0.74 &&
    feats.laplacianVariance >= 700 &&
    !looksLikeNonPhotoGraphic(feats)
  ) {
    confidence = asAiConfidence(Math.max(0.7, threshold));
    detail = "distilled-forensics-flatness";
  } else if (
    forensics !== undefined &&
    feats &&
    forensics >= 0.88 &&
    distilled >= 0.4 &&
    feats.laplacianVariance >= 800 &&
    feats.chromaFlatness >= 0.6 &&
    feats.chromaFlatness <= 0.72 &&
    spectral <= 0.38 &&
    !looksLikeNonPhotoGraphic(feats)
  ) {
    confidence = asAiConfidence(Math.max(forensics, threshold));
    detail = "forensics-high-flat-texture";
  } else if (
    // StyleGAN / TPDNE: distilled stuck mid-low + decisive CF.
    forensics !== undefined &&
    feats &&
    distilled >= 0.5 &&
    distilled < 0.6 &&
    forensics >= 0.88 &&
    forensics >= distilled + 0.25 &&
    feats.laplacianVariance >= 600 &&
    feats.chromaFlatness <= 0.58 &&
    !looksLikeNonPhotoGraphic(feats)
  ) {
    confidence = asAiConfidence(Math.max(forensics, threshold));
    detail = "forensics-ambiguous-distilled";
  } else if (
    // Neon / 3D / illustration AI slop from pixel geometry alone.
    // Does not require a high spectral AI score — busy CGI often has high
    // laplacian which the spectral heuristic treats as "real texture".
    feats &&
    looksLikeSyntheticCgi(feats) &&
    !looksLikeNonPhotoGraphic(feats) &&
    distilled < 0.92 &&
    (forensics === undefined || forensics < accurateFloor)
  ) {
    confidence = asAiConfidence(
      Math.max(
        accurateFloor,
        Math.min(0.95, 0.7 + 0.2 * Math.max(spectral, distilled)),
      ),
    );
    detail = "cgi-spectral-boost";
    labelThreshold = accurateFloor;
  } else if (
    // Non-photographic frame + mid visual signal → AI (photoreal generators
    // that dodge CGI geometry but still lack camera capture cues).
    feats &&
    !looksPhotographic(feats) &&
    !looksLikeNonPhotoGraphic(feats) &&
    distilled < 0.92 &&
    (forensics === undefined || forensics < accurateFloor) &&
    (Math.max(distilled, spectral) >= 0.32 ||
      (forensics !== undefined && forensics >= 0.5) ||
      looksLikeSyntheticCgi(feats))
  ) {
    confidence = asAiConfidence(
      Math.max(
        accurateFloor,
        Math.min(0.93, Math.max(distilled, spectral, forensics ?? 0, 0.72)),
      ),
    );
    detail = "non-photo-visual-promote";
    labelThreshold = accurateFloor;
  } else {
    confidence = asAiConfidence(baseline);
    detail =
      bump > 0
        ? `threshold=${threshold.toFixed(2)}(+${bump.toFixed(2)} small-src)`
        : `threshold=${threshold}`;
  }

  // Camera-photo gate + uncertain-band CGI rescue.
  // "? 41%" / "? 45%" neon heads were stuck mid-band after soft holds.
  // Do not override an explicit digital-graphic UI hold (Activity Monitor FPs).
  if (
    feats &&
    !looksPhotographic(feats) &&
    (looksLikeNeonAiSubject(feats) || looksLikeSyntheticCgi(feats)) &&
    confidence < accurateFloor &&
    !detail.includes("digital-graphic") &&
    !detail.includes("chart-infographic")
  ) {
    confidence = asAiConfidence(
      Math.max(accurateFloor, Math.min(0.95, Math.max(confidence, 0.72))),
    );
    detail = `${detail}|cgi-floor`;
    labelThreshold = accurateFloor;
  } else if (
    feats &&
    !looksPhotographic(feats) &&
    !looksLikeNonPhotoGraphic(feats) &&
    confidence < accurateFloor &&
    (feats.chromaFlatness >= 0.5 || looksLikeSyntheticCgi(feats)) &&
    !detail.includes("dual-mild") &&
    !detail.includes("busy-scene") &&
    !detail.includes("photo-evidence") &&
    !detail.includes("digital-graphic") &&
    !detail.includes("chart-infographic")
  ) {
    confidence = asAiConfidence(
      Math.max(accurateFloor, Math.min(0.93, Math.max(confidence, 0.72))),
    );
    detail = `${detail}|nonphoto-floor`;
    labelThreshold = accurateFloor;
  }

  // Absolute: never stamp Real unless the pixels look like a camera photo.
  // UI / flat brand / charts / laptop screens → uncertain (not AI).
  // CGI / smooth generative → AI floor.
  if (feats && confidence <= realThreshold && !looksPhotographic(feats)) {
    if (
      looksLikeNonPhotoGraphic(feats) &&
      !looksLikeNeonAiSubject(feats)
    ) {
      confidence = asAiConfidence(
        Math.min(MILD_HOLD_CAP, Math.max(realThreshold + 0.05, confidence + 0.12)),
      );
      detail = `${detail}|ui-no-real`;
    } else if (
      looksLikeNeonAiSubject(feats) ||
      looksLikeSyntheticCgi(feats) ||
      feats.chromaFlatness >= 0.5
    ) {
      confidence = asAiConfidence(
        Math.max(accurateFloor, Math.min(0.95, Math.max(confidence, 0.72))),
      );
      detail = `${detail}|stock-ai-floor`;
      labelThreshold = accurateFloor;
    } else {
      confidence = asAiConfidence(
        Math.min(MILD_HOLD_CAP, Math.max(realThreshold + 0.05, confidence + 0.08)),
      );
      detail = `${detail}|nonphoto-no-real`;
    }
  } else if (smallSource && confidence <= realThreshold) {
    confidence = asAiConfidence(
      Math.min(MILD_HOLD_CAP, Math.max(realThreshold + 0.05, confidence + 0.08)),
    );
    detail = `${detail}|small-no-real`;
  }

  return {
    confidence,
    label: labelFor(confidence, labelThreshold, realThreshold),
    tiers: [
      ...tiers,
      {
        tier: "fusion",
        aiScore: confidence,
        weight: 1,
        detail,
      },
    ],
  };
}

function calibrate(score: AiConfidence): number {
  const centered = score - 0.5;
  return 0.5 + Math.tanh(centered * 2.1) / 2;
}

export function labelFor(
  confidence: AiConfidence,
  aiThreshold: number = AI_LABEL_THRESHOLD,
  realThreshold: number = REAL_LABEL_THRESHOLD,
): DetectionLabel {
  if (confidence >= aiThreshold) {
    return { kind: "ai", confidence };
  }
  if (confidence <= realThreshold) {
    return { kind: "real", confidence: asAiConfidence(1 - confidence) };
  }
  return { kind: "uncertain", confidence };
}

export function isAiAtThreshold(
  confidence: AiConfidence,
  threshold: number = AI_LABEL_THRESHOLD,
): boolean {
  return confidence >= threshold;
}

/** Balanced accuracy helper for local evaluation harnesses. */
export function balancedAccuracy(args: {
  truePositive: number;
  trueNegative: number;
  falsePositive: number;
  falseNegative: number;
}): number {
  const tpr =
    args.truePositive + args.falseNegative === 0
      ? 0
      : args.truePositive / (args.truePositive + args.falseNegative);
  const tnr =
    args.trueNegative + args.falsePositive === 0
      ? 0
      : args.trueNegative / (args.trueNegative + args.falsePositive);
  return (tpr + tnr) / 2;
}
