import { describe, expect, it } from "vitest";
import { analyzeProvenance } from "../../extension/src/lib/provenance";

function bytesFrom(text: string): Uint8Array {
  return new TextEncoder().encode(text);
}

describe("analyzeProvenance", () => {
  it("returns neutral score for ordinary bytes", () => {
    const hit = analyzeProvenance(bytesFrom("JFIF ordinary photograph metadata"));
    expect(hit.shortCircuit).toBe(false);
    expect(hit.score).toBe(0.5);
  });

  it("short-circuits on Midjourney software tags", () => {
    const hit = analyzeProvenance(
      bytesFrom("Exif\0\0Software\0Midjourney v6 output"),
    );
    expect(hit.shortCircuit).toBe(true);
    expect(hit.score).toBeGreaterThanOrEqual(0.9);
    expect(hit.detail).toMatch(/midjourney/i);
  });

  it("short-circuits on Stable Diffusion parameter blocks", () => {
    const hit = analyzeProvenance(
      bytesFrom("Steps: 30, Sampler: Euler a, CFG scale: 7, Negative prompt: blurry"),
    );
    expect(hit.shortCircuit).toBe(true);
    expect(hit.score).toBeGreaterThanOrEqual(0.9);
  });

  it("short-circuits on C2PA digitalSourceType trainedAlgorithmicMedia", () => {
    const hit = analyzeProvenance(
      bytesFrom(
        'c2pa.actions {"digitalSourceType":"trainedAlgorithmicMedia"}',
      ),
    );
    expect(hit.shortCircuit).toBe(true);
    expect(hit.score).toBeGreaterThanOrEqual(0.9);
  });

  it("handles empty input", () => {
    const hit = analyzeProvenance(new Uint8Array());
    expect(hit.detail).toBe("empty-bytes");
  });
});
