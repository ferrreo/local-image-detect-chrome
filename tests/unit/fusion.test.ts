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
        highFreqEnergyRatio: 0.44,
        laplacianVariance: 1200,
        chromaFlatness: 0.4,
        axisAlignedEdgeRatio: 0.4,
        quantizedColorCount: 200,
        topColorShare: 0.4,
        blockiness: 0.32,
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
        axisAlignedEdgeRatio: 0.4,
        quantizedColorCount: 200,
        topColorShare: 0.4,
        blockiness: 0.2,
      },
    });
    expect(out.confidence).toBeGreaterThanOrEqual(AI_LABEL_THRESHOLD);
    expect(out.label.kind).toBe("ai");
    expect(out.tiers.at(-1)?.detail).toBe("accurate-head");
  });

  it("holds photographic macros when accurate head FPs (perfboard)", () => {
    const out = fuseDetection({
      provenance: tier("provenance", 0.5),
      spectral: tier("spectral", 0.32),
      visual: tier("visual", 0.28),
      visualSecondary: tier("visual", 0.98),
      spectralFeatures: {
        highFreqEnergyRatio: 0.44,
        laplacianVariance: 1800,
        chromaFlatness: 0.42,
        axisAlignedEdgeRatio: 0.55,
        quantizedColorCount: 180,
        topColorShare: 0.35,
        blockiness: 0.35,
      },
    });
    expect(out.confidence).toBeLessThan(AI_LABEL_THRESHOLD);
    expect(out.label.kind).not.toBe("ai");
    expect(out.tiers.at(-1)?.detail).toBe("photo-evidence-hold");
  });

  it("does not photo-hold mid accurate scores on photographic frames", () => {
    const out = fuseDetection({
      provenance: tier("provenance", 0.5),
      spectral: tier("spectral", 0.32),
      visual: tier("visual", 0.28),
      visualSecondary: tier("visual", 0.78),
      spectralFeatures: {
        highFreqEnergyRatio: 0.44,
        laplacianVariance: 1800,
        chromaFlatness: 0.42,
        axisAlignedEdgeRatio: 0.55,
        quantizedColorCount: 180,
        topColorShare: 0.35,
        blockiness: 0.35,
      },
    });
    expect(out.confidence).toBeGreaterThanOrEqual(AI_LABEL_THRESHOLD);
    expect(out.label.kind).toBe("ai");
    expect(out.tiers.at(-1)?.detail).toBe("accurate-head");
  });

  it("still promotes Lexica-style AI when spectral is not photographic", () => {
    const out = fuseDetection({
      provenance: tier("provenance", 0.5),
      spectral: tier("spectral", 0.3),
      visual: tier("visual", 0.12),
      visualSecondary: tier("visual", 0.78),
      spectralFeatures: {
        highFreqEnergyRatio: 0.22,
        laplacianVariance: 420,
        chromaFlatness: 0.72,
        axisAlignedEdgeRatio: 0.3,
        quantizedColorCount: 60,
        topColorShare: 0.55,
        blockiness: 0.15,
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
        highFreqEnergyRatio: 0.42,
        laplacianVariance: 1600,
        chromaFlatness: 0.42,
        axisAlignedEdgeRatio: 0.4,
        quantizedColorCount: 200,
        topColorShare: 0.4,
        blockiness: 0.3,
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
        chromaFlatness: 0.42,
        axisAlignedEdgeRatio: 0.4,
        quantizedColorCount: 200,
        topColorShare: 0.4,
        blockiness: 0.3,
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
        highFreqEnergyRatio: 0.4,
        laplacianVariance: 900,
        chromaFlatness: 0.42,
        axisAlignedEdgeRatio: 0.4,
        quantizedColorCount: 200,
        topColorShare: 0.4,
        blockiness: 0.28,
      },
    });
    expect(out.confidence).toBeLessThan(AI_LABEL_THRESHOLD);
    expect(out.label.kind).not.toBe("ai");
    expect(out.tiers.at(-1)?.detail).toBe("dual-mild-hold");
  });

  it("promotes near-floor accurate head on non-photo frames", () => {
    const out = fuseDetection({
      provenance: tier("provenance", 0.5),
      spectral: tier("spectral", 0.4),
      visual: tier("visual", 0.3),
      visualSecondary: tier("visual", 0.62),
      spectralFeatures: {
        highFreqEnergyRatio: 0.3,
        laplacianVariance: 500,
        chromaFlatness: 0.65,
        axisAlignedEdgeRatio: 0.3,
        quantizedColorCount: 90,
        topColorShare: 0.45,
        blockiness: 0.12,
      },
    });
    expect(out.confidence).toBeGreaterThanOrEqual(AI_LABEL_THRESHOLD);
    expect(out.label.kind).toBe("ai");
    expect(out.tiers.at(-1)?.detail).toMatch(/accurate-near-floor|cgi|nonphoto|neon-ai/);
  });

  it("labels blue CGI creature faces as AI not Real", () => {
    const out = fuseDetection({
      provenance: tier("provenance", 0.5),
      spectral: tier("spectral", 0.28),
      visual: tier("visual", 0.22),
      spectralFeatures: {
        // Detailed 3D alien: busy edges but smooth generative chroma.
        highFreqEnergyRatio: 0.44,
        laplacianVariance: 2200,
        chromaFlatness: 0.58,
        axisAlignedEdgeRatio: 0.28,
        quantizedColorCount: 140,
        topColorShare: 0.38,
        blockiness: 0.22,
      },
      sourceMinSide: 180,
    });
    expect(out.label.kind).toBe("ai");
    expect(out.confidence).toBeGreaterThanOrEqual(AI_LABEL_THRESHOLD);
  });

  it("labels neon AI circuit stock graphics as AI not Real", () => {
    const out = fuseDetection({
      provenance: tier("provenance", 0.5),
      spectral: tier("spectral", 0.35),
      visual: tier("visual", 0.25),
      spectralFeatures: {
        highFreqEnergyRatio: 0.3,
        laplacianVariance: 400,
        chromaFlatness: 0.72,
        axisAlignedEdgeRatio: 0.45,
        quantizedColorCount: 40,
        topColorShare: 0.7,
        blockiness: 0.1,
      },
      sourceMinSide: 200,
    });
    expect(out.label.kind).toBe("ai");
    expect(out.confidence).toBeGreaterThanOrEqual(AI_LABEL_THRESHOLD);
  });

  it("labels robot-hand / digital-head AI stock as AI not Real", () => {
    const out = fuseDetection({
      provenance: tier("provenance", 0.5),
      spectral: tier("spectral", 0.3),
      visual: tier("visual", 0.2),
      spectralFeatures: {
        highFreqEnergyRatio: 0.33,
        laplacianVariance: 800,
        chromaFlatness: 0.62,
        axisAlignedEdgeRatio: 0.35,
        quantizedColorCount: 100,
        topColorShare: 0.42,
        blockiness: 0.15,
      },
    });
    expect(out.label.kind).toBe("ai");
    expect(out.confidence).toBeGreaterThanOrEqual(AI_LABEL_THRESHOLD);
  });

  it("labels Craiyon neon digital-head over grid as AI not Real", () => {
    const out = fuseDetection({
      provenance: tier("provenance", 0.5),
      spectral: tier("spectral", 0.25),
      visual: tier("visual", 0.18),
      spectralFeatures: {
        // Perspective grid can lift axis edges — must not count as UI/photo.
        highFreqEnergyRatio: 0.4,
        laplacianVariance: 1600,
        chromaFlatness: 0.6,
        axisAlignedEdgeRatio: 0.55,
        quantizedColorCount: 110,
        topColorShare: 0.4,
        blockiness: 0.2,
      },
      sourceMinSide: 220,
    });
    expect(out.label.kind).toBe("ai");
    expect(out.confidence).toBeGreaterThanOrEqual(AI_LABEL_THRESHOLD);
  });

  it("promotes uncertain-band neon CGI heads to AI (not ? 41%)", () => {
    const out = fuseDetection({
      provenance: tier("provenance", 0.5),
      spectral: tier("spectral", 0.35),
      visual: tier("visual", 0.38),
      spectralFeatures: {
        highFreqEnergyRatio: 0.38,
        laplacianVariance: 1400,
        chromaFlatness: 0.58,
        axisAlignedEdgeRatio: 0.5,
        quantizedColorCount: 95,
        topColorShare: 0.45,
        blockiness: 0.16,
      },
      sourceMinSide: 320,
    });
    expect(out.label.kind).toBe("ai");
    expect(out.confidence).toBeGreaterThanOrEqual(AI_LABEL_THRESHOLD);
  });

  it("labels stubborn neon head even when perspective grid lifts axis", () => {
    const out = fuseDetection({
      provenance: tier("provenance", 0.5),
      spectral: tier("spectral", 0.3),
      visual: tier("visual", 0.24),
      spectralFeatures: {
        highFreqEnergyRatio: 0.36,
        laplacianVariance: 1700,
        chromaFlatness: 0.61,
        axisAlignedEdgeRatio: 0.6,
        quantizedColorCount: 105,
        topColorShare: 0.44,
        blockiness: 0.18,
      },
      sourceMinSide: 200,
    });
    expect(out.label.kind).toBe("ai");
    expect(out.confidence).toBeGreaterThanOrEqual(AI_LABEL_THRESHOLD);
    expect(out.tiers.at(-1)?.detail).toMatch(/neon-ai-subject|cgi|small-cgi/);
  });

  it("labels blue alien CGI face thumbs as AI", () => {
    const out = fuseDetection({
      provenance: tier("provenance", 0.5),
      spectral: tier("spectral", 0.28),
      visual: tier("visual", 0.19),
      spectralFeatures: {
        highFreqEnergyRatio: 0.42,
        laplacianVariance: 2100,
        chromaFlatness: 0.59,
        axisAlignedEdgeRatio: 0.32,
        quantizedColorCount: 130,
        topColorShare: 0.36,
        blockiness: 0.2,
      },
      sourceMinSide: 160,
    });
    expect(out.label.kind).toBe("ai");
    expect(out.confidence).toBeGreaterThanOrEqual(AI_LABEL_THRESHOLD);
  });

  it("labels soft blocky thumb digital-heads as AI (first-fold mush)", () => {
    const out = fuseDetection({
      provenance: tier("provenance", 0.5),
      spectral: tier("spectral", 0.22),
      visual: tier("visual", 0.18),
      spectralFeatures: {
        highFreqEnergyRatio: 0.58,
        laplacianVariance: 480,
        chromaFlatness: 0.58,
        axisAlignedEdgeRatio: 0.54,
        quantizedColorCount: 48,
        topColorShare: 0.45,
        blockiness: 0.5,
      },
      sourceMinSide: 140,
    });
    expect(out.label.kind).toBe("ai");
    expect(out.confidence).toBeGreaterThanOrEqual(AI_LABEL_THRESHOLD);
    expect(out.tiers.at(-1)?.detail).toMatch(/neon-ai|cgi|small-cgi|small-nonphoto/);
  });

  it("labels network digital-head stock even with mid chroma", () => {
    const out = fuseDetection({
      provenance: tier("provenance", 0.5),
      spectral: tier("spectral", 0.25),
      visual: tier("visual", 0.2),
      spectralFeatures: {
        highFreqEnergyRatio: 0.45,
        laplacianVariance: 1600,
        chromaFlatness: 0.56,
        axisAlignedEdgeRatio: 0.42,
        quantizedColorCount: 110,
        topColorShare: 0.4,
        blockiness: 0.28,
      },
      sourceMinSide: 220,
    });
    expect(out.label.kind).toBe("ai");
    expect(out.confidence).toBeGreaterThanOrEqual(AI_LABEL_THRESHOLD);
  });

  it("labels sparkly neon creature faces (high HF bokeh) as AI", () => {
    const out = fuseDetection({
      provenance: tier("provenance", 0.5),
      spectral: tier("spectral", 0.2),
      visual: tier("visual", 0.16),
      spectralFeatures: {
        // Glowing-eye CGI with yellow sparkle dots — busy HF that used to miss.
        highFreqEnergyRatio: 0.7,
        laplacianVariance: 12_000,
        chromaFlatness: 0.56,
        axisAlignedEdgeRatio: 0.28,
        quantizedColorCount: 140,
        topColorShare: 0.32,
        blockiness: 0.22,
      },
      sourceMinSide: 180,
    });
    expect(out.label.kind).toBe("ai");
    expect(out.confidence).toBeGreaterThanOrEqual(AI_LABEL_THRESHOLD);
    expect(out.tiers.at(-1)?.detail).toMatch(/neon-ai|cgi|stock-ai|small-cgi/);
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
