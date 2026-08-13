import { describe, expect, it } from "vitest";
import { needsForensicsCascade } from "../../extension/src/lib/forensics-cascade";

describe("needsForensicsCascade", () => {
  it("skips forensics only when distilled is already decisive", () => {
    expect(
      needsForensicsCascade({
        distilled: 0.88,
        spectral: 0.4,
        laplacianVariance: 900,
        chromaFlatness: 0.5,
      }),
    ).toBe(false);
  });

  it("asks forensics to confirm elevated distilled scores", () => {
    expect(
      needsForensicsCascade({
        distilled: 0.81,
        spectral: 0.42,
        laplacianVariance: 900,
        chromaFlatness: 0.5,
      }),
    ).toBe(true);
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

  it("requests forensics when distilled is mid-band even without texture gates", () => {
    expect(
      needsForensicsCascade({
        distilled: 0.58,
        spectral: 0.5,
        laplacianVariance: 100,
        chromaFlatness: 0.1,
      }),
    ).toBe(true);
  });

  it("requests forensics from 0.48 mid-band floor", () => {
    expect(
      needsForensicsCascade({
        distilled: 0.49,
        spectral: 0.5,
        laplacianVariance: 100,
        chromaFlatness: 0.1,
      }),
    ).toBe(true);
  });
});
