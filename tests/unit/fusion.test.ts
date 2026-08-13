import { describe, expect, it } from "vitest";
import {
  balancedAccuracy,
  fuseDetection,
  isAiAtThreshold,
  labelFor,
} from "../../extension/src/lib/fusion";
import { asAiConfidence } from "../../extension/src/shared/types";

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
    });
    expect(out.confidence).toBeGreaterThan(0.65);
    expect(isAiAtThreshold(out.confidence)).toBe(true);
  });

  it("labels low scores as real at the 65% threshold", () => {
    const out = fuseDetection({
      provenance: tier("provenance", 0.5),
      spectral: tier("spectral", 0.3),
      visual: tier("visual", 0.2),
    });
    expect(out.label.kind).toBe("real");
  });

  it("uses forensics + flatness band when distilled is weak", () => {
    const out = fuseDetection({
      provenance: tier("provenance", 0.5),
      spectral: tier("spectral", 0.3),
      visual: tier("visual", 0.4),
      visualSecondary: tier("visual", 0.8),
      spectralFeatures: {
        highFreqEnergyRatio: 0.3,
        laplacianVariance: 800,
        chromaFlatness: 0.5,
        blockiness: 0.25,
      },
    });
    expect(out.confidence).toBeGreaterThanOrEqual(0.65);
    expect(out.tiers.at(-1)?.detail).toMatch(/forensics/);
  });

  it("does not let high-CF smooth reals short-circuit via flatness band", () => {
    const out = fuseDetection({
      provenance: tier("provenance", 0.5),
      spectral: tier("spectral", 0.39),
      visual: tier("visual", 0.45),
      visualSecondary: tier("visual", 0.79),
      spectralFeatures: {
        highFreqEnergyRatio: 0.43,
        laplacianVariance: 635,
        chromaFlatness: 0.72,
        blockiness: 0.25,
      },
    });
    expect(out.confidence).toBeLessThan(0.65);
  });

  it("uses strong CF when distilled is stuck mid-band (StyleGAN / TPDNE)", () => {
    const out = fuseDetection({
      provenance: tier("provenance", 0.5),
      spectral: tier("spectral", 0.5),
      visual: tier("visual", 0.58),
      visualSecondary: tier("visual", 0.77),
    });
    expect(out.confidence).toBeGreaterThanOrEqual(0.65);
    expect(out.label.kind).toBe("ai");
    expect(out.tiers.at(-1)?.detail).toBe("forensics-ambiguous-distilled");
  });

  it("catches weaker CF mid-band faces (~0.72) that still beat distilled", () => {
    const out = fuseDetection({
      provenance: tier("provenance", 0.5),
      spectral: tier("spectral", 0.5),
      visual: tier("visual", 0.56),
      visualSecondary: tier("visual", 0.72),
    });
    expect(out.label.kind).toBe("ai");
    expect(out.tiers.at(-1)?.detail).toBe("forensics-ambiguous-distilled");
  });
});

describe("labelFor / balancedAccuracy", () => {
  it("uses 65% operating point by default", () => {
    expect(labelFor(asAiConfidence(0.65)).kind).toBe("ai");
    expect(labelFor(asAiConfidence(0.64)).kind).toBe("uncertain");
    expect(labelFor(asAiConfidence(0.35)).kind).toBe("real");
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
