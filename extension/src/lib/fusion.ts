import {
  asAiConfidence,
  AI_LABEL_THRESHOLD,
  EVAL_CONFIDENCE_THRESHOLD,
  REAL_LABEL_THRESHOLD,
  type AiConfidence,
  type DetectionLabel,
  type TierSignal,
} from "../shared/types";

export type SpectralFusionFeatures = {
  highFreqEnergyRatio: number;
  laplacianVariance: number;
  chromaFlatness: number;
  blockiness: number;
};

export type FusionInput = {
  provenance: TierSignal;
  spectral: TierSignal;
  visual: TierSignal;
  /** Optional secondary visual model (Community Forensics). */
  visualSecondary?: TierSignal;
  spectralFeatures?: SpectralFusionFeatures;
  /** AI label threshold (product default 69.51%; eval often passes 0.65). */
  threshold?: number;
  /** Real label threshold (product default 40.99%). */
  realThreshold?: number;
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
  const threshold = input.threshold ?? AI_LABEL_THRESHOLD;
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

  let confidence: AiConfidence;
  let detail: string;
  const baseline = calibrate(
    asAiConfidence(0.78 * distilled + 0.22 * spectral),
  );

  if (
    // Both heads elevated but not decisive — real indoor / product / group
    // photos land here (observed AI 72% and AI 81% FPs).
    // Keep distilled < 0.6 free for StyleGAN / TPDNE CF recovery.
    forensics !== undefined &&
    distilled >= 0.6 &&
    distilled < 0.88 &&
    forensics >= 0.55 &&
    forensics < 0.88
  ) {
    confidence = asAiConfidence(Math.min(baseline, MILD_HOLD_CAP));
    detail = "dual-mild-hold";
  } else if (
    // Busy natural scenes with only a strong distilled head.
    feats &&
    feats.laplacianVariance >= 900 &&
    distilled >= 0.6 &&
    distilled < 0.9 &&
    (forensics === undefined || forensics < 0.88)
  ) {
    confidence = asAiConfidence(Math.min(baseline, MILD_HOLD_CAP));
    detail = "busy-scene-hold";
  } else if (distilled >= 0.88 && spectral <= 0.4) {
    confidence = asAiConfidence(Math.max(distilled, threshold));
    detail = "distilled-near-threshold";
  } else if (
    // Proofmark-class accurate head: trust it when it clears the AI floor
    // even if distilled is near-zero (Lexica / modern generator misses).
    forensics !== undefined &&
    forensics >= threshold &&
    distilled < 0.88
  ) {
    confidence = asAiConfidence(Math.max(forensics, threshold));
    detail = "accurate-head";
  } else if (
    forensics !== undefined &&
    feats &&
    forensics >= 0.88 &&
    distilled >= 0.3 &&
    feats.laplacianVariance >= 580 &&
    feats.chromaFlatness >= 0.34 &&
    feats.chromaFlatness <= 0.7
  ) {
    confidence = asAiConfidence(Math.max(forensics, threshold));
    detail = "forensics-flatness-band";
  } else if (
    forensics !== undefined &&
    feats &&
    distilled >= 0.62 &&
    forensics >= 0.88 &&
    feats.chromaFlatness >= 0.74 &&
    feats.laplacianVariance >= 700
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
    spectral <= 0.38
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
    feats.chromaFlatness <= 0.58
  ) {
    confidence = asAiConfidence(Math.max(forensics, threshold));
    detail = "forensics-ambiguous-distilled";
  } else {
    confidence = asAiConfidence(baseline);
    detail = `threshold=${threshold}`;
  }

  return {
    confidence,
    label: labelFor(confidence, threshold, realThreshold),
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
