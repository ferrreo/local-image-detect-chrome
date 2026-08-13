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
