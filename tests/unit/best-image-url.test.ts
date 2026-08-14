import { describe, expect, it } from "vitest";
import {
  aiThresholdBumpForSourceSide,
  parseSrcset,
  pickBestCandidate,
} from "../../extension/src/lib/best-image-url";
import { fuseDetection } from "../../extension/src/lib/fusion";
import { asAiConfidence } from "../../extension/src/shared/types";

describe("parseSrcset / pickBestCandidate", () => {
  it("prefers the largest width descriptor", () => {
    const parsed = parseSrcset(
      "https://cdn.example/a.jpg 400w, https://cdn.example/b.jpg 1200w, https://cdn.example/c.jpg 800w",
    );
    expect(pickBestCandidate(parsed)).toBe("https://cdn.example/b.jpg");
  });

  it("scores density descriptors below typical width descriptors", () => {
    const parsed = parseSrcset(
      "https://cdn.example/1x.jpg 1x, https://cdn.example/2x.jpg 2x, https://cdn.example/wide.jpg 1600w",
    );
    expect(pickBestCandidate(parsed)).toBe("https://cdn.example/wide.jpg");
  });
});

describe("aiThresholdBumpForSourceSide", () => {
  it("does not bump full-resolution assets", () => {
    expect(aiThresholdBumpForSourceSide(512)).toBe(0);
    expect(aiThresholdBumpForSourceSide(384)).toBe(0);
  });

  it("bumps more aggressively as the source shrinks", () => {
    expect(aiThresholdBumpForSourceSide(300)).toBeGreaterThan(0);
    expect(aiThresholdBumpForSourceSide(120)).toBeGreaterThan(
      aiThresholdBumpForSourceSide(300),
    );
    expect(aiThresholdBumpForSourceSide(64)).toBeGreaterThan(
      aiThresholdBumpForSourceSide(120),
    );
  });
});

describe("fuseDetection small-source hold", () => {
  function tier(
    id: "provenance" | "spectral" | "visual",
    score: number,
  ) {
    return {
      tier: id,
      aiScore: asAiConfidence(score),
      weight: 1,
      detail: id,
    };
  }

  it("holds AI on upscaled thumbs unless accurate head clears the raised floor", () => {
    const held = fuseDetection({
      provenance: tier("provenance", 0.5),
      spectral: tier("spectral", 0.35),
      visual: tier("visual", 0.82),
      threshold: 0.6951,
      sourceMinSide: 128,
    });
    expect(held.label.kind).not.toBe("ai");
    expect(held.tiers.at(-1)?.detail).toMatch(/small-source-hold/);

    const decisive = fuseDetection({
      provenance: tier("provenance", 0.5),
      spectral: tier("spectral", 0.35),
      visual: tier("visual", 0.4),
      visualSecondary: tier("visual", 0.92),
      threshold: 0.6951,
      sourceMinSide: 128,
    });
    expect(decisive.label.kind).toBe("ai");
  });

  it("holds AI on digital UI screenshots even when both heads are maxed", () => {
    const out = fuseDetection({
      provenance: tier("provenance", 0.5),
      spectral: tier("spectral", 0.55),
      visual: tier("visual", 0.99),
      visualSecondary: tier("visual", 0.99),
      threshold: 0.6951,
      sourceMinSide: 800,
      spectralFeatures: {
        highFreqEnergyRatio: 0.2,
        laplacianVariance: 800,
        chromaFlatness: 0.83,
        blockiness: 0.6,
        axisAlignedEdgeRatio: 0.95,
        quantizedColorCount: 22,
        topColorShare: 0.94,
      },
    });
    expect(out.label.kind).not.toBe("ai");
    expect(out.tiers.at(-1)?.detail).toMatch(/digital-graphic-hold/);
  });

  it("holds AI on flat brand posters without UI chrome", () => {
    const out = fuseDetection({
      provenance: tier("provenance", 0.5),
      spectral: tier("spectral", 0.45),
      visual: tier("visual", 0.88),
      visualSecondary: tier("visual", 0.91),
      threshold: 0.6951,
      sourceMinSide: 480,
      spectralFeatures: {
        highFreqEnergyRatio: 0.22,
        laplacianVariance: 900,
        chromaFlatness: 0.94,
        blockiness: 0.25,
        axisAlignedEdgeRatio: 0.35,
        quantizedColorCount: 6,
        topColorShare: 0.97,
      },
    });
    expect(out.label.kind).not.toBe("ai");
    expect(out.tiers.at(-1)?.detail).toMatch(/digital-graphic-hold/);
  });

  it("holds Activity Monitor / process-table UI screenshots (not AI 72%)", () => {
    const out = fuseDetection({
      provenance: tier("provenance", 0.5),
      spectral: tier("spectral", 0.25),
      visual: tier("visual", 0.2),
      spectralFeatures: {
        highFreqEnergyRatio: 0.28,
        laplacianVariance: 1100,
        chromaFlatness: 0.7,
        blockiness: 0.15,
        axisAlignedEdgeRatio: 0.74,
        quantizedColorCount: 90,
        topColorShare: 0.7,
      },
      sourceMinSide: 900,
    });
    expect(out.label.kind).not.toBe("ai");
    expect(out.tiers.at(-1)?.detail).toMatch(/digital-graphic|ui-no-real/);
  });

  it("holds dark usage-meter dashboard screenshots", () => {
    const out = fuseDetection({
      provenance: tier("provenance", 0.5),
      spectral: tier("spectral", 0.2),
      visual: tier("visual", 0.15),
      spectralFeatures: {
        highFreqEnergyRatio: 0.22,
        laplacianVariance: 350,
        chromaFlatness: 0.78,
        blockiness: 0.08,
        axisAlignedEdgeRatio: 0.65,
        quantizedColorCount: 28,
        topColorShare: 0.82,
      },
      sourceMinSide: 700,
    });
    expect(out.label.kind).not.toBe("ai");
    expect(out.tiers.at(-1)?.detail).toMatch(/digital-graphic|ui-no-real/);
  });

  it("holds light text UI roast cards (not AI 72%)", () => {
    const out = fuseDetection({
      provenance: tier("provenance", 0.5),
      spectral: tier("spectral", 0.3),
      visual: tier("visual", 0.25),
      visualSecondary: tier("visual", 0.79),
      spectralFeatures: {
        highFreqEnergyRatio: 0.3,
        laplacianVariance: 1400,
        chromaFlatness: 0.62,
        blockiness: 0.12,
        axisAlignedEdgeRatio: 0.64,
        quantizedColorCount: 55,
        topColorShare: 0.72,
      },
      sourceMinSide: 900,
    });
    expect(out.label.kind).not.toBe("ai");
    expect(out.tiers.at(-1)?.detail).toMatch(/digital-graphic|ui-no-real/);
  });

  it("holds dark docs / README screenshots (not AI 79%)", () => {
    const out = fuseDetection({
      provenance: tier("provenance", 0.5),
      spectral: tier("spectral", 0.28),
      visual: tier("visual", 0.22),
      visualSecondary: tier("visual", 0.79),
      spectralFeatures: {
        highFreqEnergyRatio: 0.26,
        laplacianVariance: 900,
        chromaFlatness: 0.7,
        blockiness: 0.1,
        axisAlignedEdgeRatio: 0.66,
        quantizedColorCount: 40,
        topColorShare: 0.78,
      },
      sourceMinSide: 1000,
    });
    expect(out.label.kind).not.toBe("ai");
    expect(out.tiers.at(-1)?.detail).toMatch(/digital-graphic|ui-no-real/);
  });

  it("holds geometric ePaper / dithered design-system posters", () => {
    const out = fuseDetection({
      provenance: tier("provenance", 0.5),
      spectral: tier("spectral", 0.4),
      visual: tier("visual", 0.35),
      visualSecondary: tier("visual", 0.78),
      spectralFeatures: {
        highFreqEnergyRatio: 0.36,
        laplacianVariance: 2200,
        chromaFlatness: 0.55,
        blockiness: 0.2,
        axisAlignedEdgeRatio: 0.7,
        quantizedColorCount: 36,
        topColorShare: 0.58,
      },
      sourceMinSide: 800,
    });
    expect(out.label.kind).not.toBe("ai");
    expect(out.tiers.at(-1)?.detail).toMatch(/digital-graphic|ui-no-real/);
  });

  it("holds macOS notification / toast screenshots (not AI 72%)", () => {
    const out = fuseDetection({
      provenance: tier("provenance", 0.5),
      spectral: tier("spectral", 0.22),
      visual: tier("visual", 0.18),
      visualSecondary: tier("visual", 0.74),
      spectralFeatures: {
        // Dark banner on black: dominant fill, soft card axis, glyph HF.
        highFreqEnergyRatio: 0.24,
        laplacianVariance: 900,
        chromaFlatness: 0.72,
        blockiness: 0.1,
        axisAlignedEdgeRatio: 0.48,
        quantizedColorCount: 64,
        topColorShare: 0.76,
      },
      sourceMinSide: 600,
    });
    expect(out.label.kind).not.toBe("ai");
    expect(out.tiers.at(-1)?.detail).toMatch(/digital-graphic|ui-no-real/);
  });

  it("holds bar-chart / KPI infographics (not AI 98%)", () => {
    const out = fuseDetection({
      provenance: tier("provenance", 0.5),
      spectral: tier("spectral", 0.35),
      visual: tier("visual", 0.98),
      visualSecondary: tier("visual", 0.96),
      spectralFeatures: {
        highFreqEnergyRatio: 0.28,
        laplacianVariance: 1800,
        chromaFlatness: 0.68,
        blockiness: 0.12,
        axisAlignedEdgeRatio: 0.68,
        quantizedColorCount: 42,
        topColorShare: 0.62,
      },
      sourceMinSide: 900,
    });
    expect(out.label.kind).not.toBe("ai");
    expect(out.tiers.at(-1)?.detail).toMatch(/chart-infographic|digital-graphic|ui-no-real/);
  });

  it("holds tweet-framed LLM bench charts even with maxed distilled", () => {
    const out = fuseDetection({
      provenance: tier("provenance", 0.5),
      spectral: tier("spectral", 0.25),
      visual: tier("visual", 0.98),
      visualSecondary: tier("visual", 0.97),
      spectralFeatures: {
        // Dark social chrome + white KPI card: diluted topShare, strong bars.
        highFreqEnergyRatio: 0.32,
        laplacianVariance: 2200,
        chromaFlatness: 0.62,
        blockiness: 0.14,
        axisAlignedEdgeRatio: 0.64,
        quantizedColorCount: 58,
        topColorShare: 0.36,
        frameAxisAlignedEdgeRatio: 0.78,
        frameTopColorShare: 0.7,
      },
      sourceMinSide: 1000,
    });
    expect(out.label.kind).not.toBe("ai");
    expect(out.tiers.at(-1)?.detail).toMatch(/chart-infographic|digital-graphic|ui-no-real/);
  });

  it("holds multi-series scatter plots (inflated palette, high page fill)", () => {
    // Measured from tests/fixtures/images/ui-fp/scatter_chart.png
    const out = fuseDetection({
      provenance: tier("provenance", 0.5),
      spectral: tier("spectral", 0.35),
      visual: tier("visual", 0.98),
      visualSecondary: tier("visual", 0.96),
      spectralFeatures: {
        highFreqEnergyRatio: 0.44,
        laplacianVariance: 567,
        chromaFlatness: 0.989,
        blockiness: 0.24,
        axisAlignedEdgeRatio: 0.694,
        quantizedColorCount: 119,
        topColorShare: 0.99,
      },
      sourceMinSide: 900,
    });
    expect(out.label.kind).not.toBe("ai");
    expect(out.tiers.at(-1)?.detail).toMatch(
      /digital-graphic|chart-infographic|ui-no-real/,
    );
  });

  it("holds dark code cards with glyph HF (not AI 72/98%)", () => {
    // Measured from tests/fixtures/images/ui-fp/code_card.png
    const out = fuseDetection({
      provenance: tier("provenance", 0.5),
      spectral: tier("spectral", 0.32),
      visual: tier("visual", 0.98),
      visualSecondary: tier("visual", 0.72),
      spectralFeatures: {
        highFreqEnergyRatio: 0.51,
        laplacianVariance: 189,
        chromaFlatness: 0.821,
        blockiness: 0.18,
        axisAlignedEdgeRatio: 0.736,
        quantizedColorCount: 26,
        topColorShare: 0.993,
      },
      sourceMinSide: 520,
    });
    expect(out.label.kind).not.toBe("ai");
    expect(out.tiers.at(-1)?.detail).toMatch(
      /digital-graphic|chart-infographic|ui-no-real/,
    );
  });

  it("holds tweet-framed busy scatter plots (colors ~190)", () => {
    // Measured from tests/fixtures/images/ui-fp/scatter_tweet.jpg
    const out = fuseDetection({
      provenance: tier("provenance", 0.5),
      spectral: tier("spectral", 0.25),
      visual: tier("visual", 0.98),
      visualSecondary: tier("visual", 0.96),
      spectralFeatures: {
        highFreqEnergyRatio: 0.164,
        laplacianVariance: 1052,
        chromaFlatness: 0.989,
        blockiness: 0.086,
        axisAlignedEdgeRatio: 0.788,
        quantizedColorCount: 193,
        topColorShare: 0.981,
        frameAxisAlignedEdgeRatio: 0.839,
        frameTopColorShare: 0.983,
      },
      sourceMinSide: 900,
    });
    expect(out.label.kind).not.toBe("ai");
    expect(out.tiers.at(-1)?.detail).toMatch(/digital-graphic|ui-no-real/);
  });

  it("holds laptop screen captures of KPI charts (not AI 98%)", () => {
    // Measured from tests/fixtures/images/ui-fp/laptop_screen.jpg
    const out = fuseDetection({
      provenance: tier("provenance", 0.5),
      spectral: tier("spectral", 0.3),
      visual: tier("visual", 0.98),
      visualSecondary: tier("visual", 0.97),
      spectralFeatures: {
        highFreqEnergyRatio: 0.214,
        laplacianVariance: 1785,
        chromaFlatness: 0.953,
        blockiness: 0.633,
        axisAlignedEdgeRatio: 0.909,
        quantizedColorCount: 109,
        topColorShare: 0.906,
        frameAxisAlignedEdgeRatio: 0.949,
        frameTopColorShare: 0.925,
      },
      sourceMinSide: 900,
    });
    expect(out.label.kind).not.toBe("ai");
    expect(out.tiers.at(-1)?.detail).toMatch(/digital-graphic|ui-no-real/);
  });

  it("holds LLM Performance Evaluation KPI cards at maxed distilled", () => {
    // Measured from tests/fixtures/images/ui-fp/llm_perf_eval.jpg
    const out = fuseDetection({
      provenance: tier("provenance", 0.5),
      spectral: tier("spectral", 0.35),
      visual: tier("visual", 0.98),
      visualSecondary: tier("visual", 0.96),
      spectralFeatures: {
        highFreqEnergyRatio: 0.219,
        laplacianVariance: 1960,
        chromaFlatness: 0.95,
        blockiness: 0.822,
        axisAlignedEdgeRatio: 0.851,
        quantizedColorCount: 110,
        topColorShare: 0.949,
      },
      sourceMinSide: 900,
    });
    expect(out.label.kind).not.toBe("ai");
    expect(out.tiers.at(-1)?.detail).toMatch(/digital-graphic|ui-no-real/);
  });
});
