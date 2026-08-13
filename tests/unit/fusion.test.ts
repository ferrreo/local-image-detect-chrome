import { describe, expect, it } from "vitest";
import {
  balancedAccuracy,
  fuseDetection,
  isAiAtThreshold,
  labelFor,
} from "../../extension/src/lib/fusion";
import {
  AI_LABEL_THRESHOLD,
  asAiConfidence,
  EVAL_CONFIDENCE_THRESHOLD,
  REAL_LABEL_THRESHOLD,
} from "../../extension/src/shared/types";

function tier(
  id: "provenance" | "spectral" | "visual",
  score: number,
  extra?: { shortCircuit?: boolean },
) {
  return {
    tier: id,
    aiScore: asAiConfidence(score),
    weight: 1,
    detail: id,
    ...(extra?.shortCircuit !== undefined
      ? { shortCircuit: extra.shortCircuit }
      : {}),
  };
}

describe("fuseDetection", () => {
  it("short-circuits on strong provenance", () => {
    const out = fuseDetection({
      provenance: tier("provenance", 0.98, { shortCircuit: true }),
      spectral: tier("spectral", 0.2),
      visual: tier("visual", 0.2),
    });
    expect(out.confidence).toBeGreaterThanOrEqual(0.9);
    expect(out.label.kind).toBe("ai");
  });

  it("weights visual score most heavily", () => {
    const out = fuseDetection({
      provenance: tier("provenance", 0.5),
      spectral: tier("spectral", 0.2),
      visual: tier("visual", 0.9),
      threshold: EVAL_CONFIDENCE_THRESHOLD,
    });
    expect(out.confidence).toBeGreaterThan(0.65);
    expect(isAiAtThreshold(out.confidence, EVAL_CONFIDENCE_THRESHOLD)).toBe(
      true,
    );
  });

  it("labels low scores as real at the product real threshold", () => {
    const out = fuseDetection({
      provenance: tier("provenance", 0.5),
      spectral: tier("spectral", 0.3),
      visual: tier("visual", 0.2),
    });
    expect(out.label.kind).toBe("real");
  });

  it("promotes accurate head when secondary clears AI floor (Lexica)", () => {
    const out = fuseDetection({
      provenance: tier("provenance", 0.5),
      spectral: tier("spectral", 0.3),
      visual: tier("visual", 0.12),
      visualSecondary: tier("visual", 0.78),
    });
    expect(out.confidence).toBeGreaterThanOrEqual(AI_LABEL_THRESHOLD);
    expect(out.label.kind).toBe("ai");
    expect(out.tiers.at(-1)?.detail).toBe("accurate-head");
  });

  it("does not let sub-threshold secondary promote reals", () => {
    const out = fuseDetection({
      provenance: tier("provenance", 0.5),
      spectral: tier("spectral", 0.39),
      visual: tier("visual", 0.45),
      visualSecondary: tier("visual", 0.64),
      spectralFeatures: {
        highFreqEnergyRatio: 0.43,
        laplacianVariance: 635,
        chromaFlatness: 0.72,
        blockiness: 0.25,
      },
    });
    expect(out.confidence).toBeLessThan(AI_LABEL_THRESHOLD);
  });

  it("uses strong accurate head when distilled is stuck mid-band", () => {
    const out = fuseDetection({
      provenance: tier("provenance", 0.5),
      spectral: tier("spectral", 0.5),
      visual: tier("visual", 0.55),
      visualSecondary: tier("visual", 0.9),
      spectralFeatures: {
        highFreqEnergyRatio: 0.35,
        laplacianVariance: 900,
        chromaFlatness: 0.45,
        blockiness: 0.2,
      },
    });
    expect(out.confidence).toBeGreaterThanOrEqual(AI_LABEL_THRESHOLD);
    expect(out.label.kind).toBe("ai");
    expect(out.tiers.at(-1)?.detail).toBe("accurate-head");
  });

  it("holds mild dual agreement under AI label (group selfie / Soylent FPs)", () => {
    const out = fuseDetection({
      provenance: tier("provenance", 0.5),
      spectral: tier("spectral", 0.42),
      visual: tier("visual", 0.72),
      visualSecondary: tier("visual", 0.81),
      spectralFeatures: {
        highFreqEnergyRatio: 0.4,
        laplacianVariance: 1600,
        chromaFlatness: 0.5,
        blockiness: 0.2,
      },
    });
    expect(out.confidence).toBeLessThan(AI_LABEL_THRESHOLD);
    expect(out.label.kind).not.toBe("ai");
    expect(out.tiers.at(-1)?.detail).toBe("dual-mild-hold");
  });

  it("holds busy real product photos even with strong distilled alone", () => {
    const out = fuseDetection({
      provenance: tier("provenance", 0.5),
      spectral: tier("spectral", 0.4),
      visual: tier("visual", 0.81),
      spectralFeatures: {
        highFreqEnergyRatio: 0.42,
        laplacianVariance: 1400,
        chromaFlatness: 0.48,
        blockiness: 0.2,
      },
    });
    expect(out.confidence).toBeLessThan(AI_LABEL_THRESHOLD);
    expect(out.label.kind).not.toBe("ai");
    expect(out.tiers.at(-1)?.detail).toBe("busy-scene-hold");
  });

  it("does not promote kitchen-style mild dual scores (prefer no FP)", () => {
    const out = fuseDetection({
      provenance: tier("provenance", 0.5),
      spectral: tier("spectral", 0.423),
      visual: tier("visual", 0.644),
      visualSecondary: tier("visual", 0.727),
      spectralFeatures: {
        highFreqEnergyRatio: 0.35,
        laplacianVariance: 900,
        chromaFlatness: 0.5,
        blockiness: 0.2,
      },
    });
    expect(out.confidence).toBeLessThan(AI_LABEL_THRESHOLD);
    expect(out.label.kind).not.toBe("ai");
    expect(out.tiers.at(-1)?.detail).toBe("dual-mild-hold");
  });
});

describe("labelFor / balancedAccuracy", () => {
  it("uses asymmetric product bands by default (≤40.99% real, ≥69.51% AI)", () => {
    expect(labelFor(asAiConfidence(0.6951)).kind).toBe("ai");
    expect(labelFor(asAiConfidence(0.695)).kind).toBe("uncertain");
    expect(labelFor(asAiConfidence(0.5)).kind).toBe("uncertain");
    expect(labelFor(asAiConfidence(0.4099)).kind).toBe("real");
    expect(labelFor(asAiConfidence(0.41)).kind).toBe("uncertain");
    expect(REAL_LABEL_THRESHOLD).toBeCloseTo(0.4099);
    expect(AI_LABEL_THRESHOLD).toBeCloseTo(0.6951);
  });

  it("still supports bounty 65% eval threshold when passed", () => {
    expect(labelFor(asAiConfidence(0.65), 0.65, 0.35).kind).toBe("ai");
    expect(labelFor(asAiConfidence(0.64), 0.65, 0.35).kind).toBe("uncertain");
    expect(labelFor(asAiConfidence(0.35), 0.65, 0.35).kind).toBe("real");
  });

  it("computes balanced accuracy", () => {
    expect(
      balancedAccuracy({
        truePositive: 80,
        trueNegative: 70,
        falsePositive: 30,
        falseNegative: 20,
      }),
    ).toBeCloseTo((0.8 + 0.7) / 2);
  });
});
