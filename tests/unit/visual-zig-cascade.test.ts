import { describe, expect, it } from "vitest";
import { needsForensicsCascade } from "../../extension/src/lib/forensics-cascade";

describe("needsForensicsCascade", () => {
  it("skips forensics when distilled near-threshold gate applies", () => {
    expect(
      needsForensicsCascade({
        distilled: 0.65,
        spectral: 0.4,
        laplacianVariance: 900,
        chromaFlatness: 0.5,
      }),
    ).toBe(false);
  });

  it("requests forensics in the flatness band used by fusion", () => {
    expect(
      needsForensicsCascade({
        distilled: 0.35,
        spectral: 0.5,
        laplacianVariance: 600,
        chromaFlatness: 0.45,
      }),
    ).toBe(true);
  });

  it("skips obvious low-signal reals outside forensics gates", () => {
    expect(
      needsForensicsCascade({
        distilled: 0.25,
        spectral: 0.55,
        laplacianVariance: 200,
        chromaFlatness: 0.2,
      }),
    ).toBe(false);
  });
});
