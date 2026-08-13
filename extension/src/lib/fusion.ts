import {
  asAiConfidence,
  EVAL_CONFIDENCE_THRESHOLD,
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
  threshold?: number;
};

export type FusionOutput = {
  confidence: AiConfidence;
  label: DetectionLabel;
  tiers: TierSignal[];
};

/**
 * Calibrated fusion for the bounty evaluation threshold (default 65%).
 * Provenance short-circuits when generators declare themselves.
 * Distilled visual is primary; Community Forensics + spectral texture/flatness
 * gates recover modern generators that fool a single head without flipping
 * smooth real photos that CF over-scores.
 */
export function fuseDetection(input: FusionInput): FusionOutput {
  const threshold = input.threshold ?? EVAL_CONFIDENCE_THRESHOLD;
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
      label: labelFor(confidence, threshold),
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

  if (distilled >= 0.63 && spectral <= 0.43) {
    confidence = asAiConfidence(Math.max(distilled, 0.66));
    detail = "distilled-near-threshold";
  } else if (
    forensics !== undefined &&
    feats &&
    forensics >= 0.78 &&
    distilled >= 0.3 &&
    feats.laplacianVariance >= 580 &&
    feats.chromaFlatness >= 0.34 &&
    feats.chromaFlatness <= 0.7
  ) {
    confidence = asAiConfidence(Math.max(forensics, 0.66));
    detail = "forensics-flatness-band";
  } else if (
    forensics !== undefined &&
    feats &&
    distilled >= 0.62 &&
    forensics >= 0.72 &&
    feats.chromaFlatness >= 0.74 &&
    feats.laplacianVariance >= 700
  ) {
    confidence = asAiConfidence(0.68);
    detail = "distilled-forensics-flatness";
  } else if (
    forensics !== undefined &&
    feats &&
    forensics >= 0.78 &&
    distilled >= 0.4 &&
    feats.laplacianVariance >= 800 &&
    feats.chromaFlatness >= 0.6 &&
    feats.chromaFlatness <= 0.72 &&
    spectral <= 0.38
  ) {
    confidence = asAiConfidence(Math.max(forensics, 0.66));
    detail = "forensics-high-flat-texture";
  } else if (
    // StyleGAN / TPDNE: distilled stuck mid-low + decisive CF.
    // Mild CF (~0.72) over-fires on real restaurant / busy photos.
    forensics !== undefined &&
    feats &&
    distilled >= 0.5 &&
    distilled < 0.6 &&
    forensics >= 0.78 &&
    forensics >= distilled + 0.18 &&
    feats.laplacianVariance >= 600 &&
    feats.chromaFlatness <= 0.58
  ) {
    confidence = asAiConfidence(Math.max(forensics, 0.66));
    detail = "forensics-ambiguous-distilled";
  } else {
    const fused = 0.78 * distilled + 0.22 * spectral;
    confidence = asAiConfidence(calibrate(asAiConfidence(fused)));
    detail = `threshold=${threshold}`;
  }

  return {
    confidence,
    label: labelFor(confidence, threshold),
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
  threshold = EVAL_CONFIDENCE_THRESHOLD,
): DetectionLabel {
  if (confidence >= threshold) {
    return { kind: "ai", confidence };
  }
  if (confidence <= 1 - threshold) {
    return { kind: "real", confidence: asAiConfidence(1 - confidence) };
  }
  return { kind: "uncertain", confidence };
}

export function isAiAtThreshold(
  confidence: AiConfidence,
  threshold = EVAL_CONFIDENCE_THRESHOLD,
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
