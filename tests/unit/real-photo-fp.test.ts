import { describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";
import path from "node:path";
import {
  analyzeSpectral,
  looksLikeCameraCapture,
  looksLikeNonPhotoGraphic,
  looksPhotographic,
} from "../../extension/src/lib/spectral";
import { fuseDetection } from "../../extension/src/lib/fusion";
import {
  AI_LABEL_THRESHOLD,
  asAiConfidence,
} from "../../extension/src/shared/types";
import {
  decodeImageBytes,
  guessMimeType,
  rasterizeForSpectral,
} from "../../extension/src/lib/image-decode";

const fixtures = path.resolve("tests/fixtures/images");

async function loadSpectral(name: string) {
  const buf = readFileSync(path.join(fixtures, name));
  const bytes = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
  const decoded = await decodeImageBytes(
    bytes,
    guessMimeType(new Uint8Array(bytes)) ?? "image/jpeg",
  );
  try {
    const spectralImage = await rasterizeForSpectral(decoded.bitmap);
    return analyzeSpectral(spectralImage);
  } finally {
    decoded.bitmap.close();
  }
}

describe("real photo false-positive holds", () => {
  for (const name of [
    "real_noise_1.png",
    "real_noise_2.png",
    "real_noise_3.png",
  ]) {
    it(`does not accurate-head promote ${name}`, async () => {
      const spectral = await loadSpectral(name);
      const feats = spectral.features;
      expect(looksLikeCameraCapture(feats)).toBe(true);
      const out = fuseDetection({
        provenance: {
          tier: "provenance",
          aiScore: asAiConfidence(0.5),
          weight: 0.08,
          detail: "none",
        },
        spectral: {
          tier: "spectral",
          aiScore: spectral.score,
          weight: 0.2,
          detail: spectral.detail,
        },
        visual: {
          tier: "visual",
          aiScore: asAiConfidence(0.12),
          weight: 0.72,
          detail: "distilled",
        },
        visualSecondary: {
          tier: "visual",
          aiScore: asAiConfidence(1),
          weight: 0.5,
          detail: "accurate",
        },
        spectralFeatures: feats,
        sourceMinSide: 256,
        threshold: AI_LABEL_THRESHOLD,
      });
      expect(out.label.kind).not.toBe("ai");
      expect(out.tiers.at(-1)?.detail).toBe("photo-evidence-hold");
    });
  }

  it("holds outdoor iPad lifestyle JPEG as camera photo, not AI", async () => {
    const spectral = await loadSpectral("real-fp/outdoor_ipad.jpg");
    const feats = spectral.features;
    expect(looksPhotographic(feats)).toBe(false);
    expect(looksLikeCameraCapture(feats)).toBe(true);
    // Screen geometry alone would look like a UI card — camera prior wins.
    expect(looksLikeNonPhotoGraphic(feats)).toBe(true);

    const out = fuseDetection({
      provenance: {
        tier: "provenance",
        aiScore: asAiConfidence(0.5),
        weight: 0.08,
        detail: "none",
      },
      spectral: {
        tier: "spectral",
        aiScore: spectral.score,
        weight: 0.2,
        detail: spectral.detail,
      },
      visual: {
        tier: "visual",
        aiScore: asAiConfidence(0.45),
        weight: 0.72,
        detail: "distilled",
      },
      visualSecondary: {
        tier: "visual",
        aiScore: asAiConfidence(0.99),
        weight: 0.5,
        detail: "accurate",
      },
      spectralFeatures: feats,
      sourceMinSide: 1024,
      threshold: AI_LABEL_THRESHOLD,
    });
    expect(out.confidence).toBeLessThan(AI_LABEL_THRESHOLD);
    expect(out.label.kind).not.toBe("ai");
    expect(out.tiers.at(-1)?.detail).toBe("photo-evidence-hold");
  });
});
