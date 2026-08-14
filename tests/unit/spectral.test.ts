import { describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";
import path from "node:path";
import {
  analyzeSpectral,
  fftMagnitude,
  looksLikeChartOrInfographic,
  looksLikeDigitalUi,
  looksLikeFlatGraphic,
  looksLikeNeonAiSubject,
  looksLikeNonPhotoGraphic,
  looksPhotographic,
} from "../../extension/src/lib/spectral";
import {
  decodeImageBytes,
  guessMimeType,
  rasterizeForSpectral,
} from "../../extension/src/lib/image-decode";

function makeImageData(
  width: number,
  height: number,
  fill: (x: number, y: number) => [number, number, number],
): ImageData {
  const data = new Uint8ClampedArray(width * height * 4);
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const [r, g, b] = fill(x, y);
      const i = (y * width + x) * 4;
      data[i] = r;
      data[i + 1] = g;
      data[i + 2] = b;
      data[i + 3] = 255;
    }
  }
  return { data, width, height, colorSpace: "srgb" } as ImageData;
}

describe("fftMagnitude", () => {
  it("peaks at the expected frequency bin for a pure sine", () => {
    const n = 64;
    const signal = new Float32Array(n);
    const freq = 4;
    for (let i = 0; i < n; i += 1) {
      signal[i] = Math.sin((2 * Math.PI * freq * i) / n);
    }
    const mag = fftMagnitude(signal);
    let peak = 0;
    let peakIdx = 0;
    for (let i = 0; i < mag.length; i += 1) {
      const v = mag[i] ?? 0;
      if (v > peak) {
        peak = v;
        peakIdx = i;
      }
    }
    expect(peakIdx).toBe(freq);
  });

  it("rejects non power-of-two lengths", () => {
    expect(() => fftMagnitude(new Float32Array(10))).toThrow(/power-of-two/);
  });
});

describe("analyzeSpectral", () => {
  it("scores smooth gradients higher (more AI-like) than noisy images", () => {
    const smooth = makeImageData(64, 64, (x, y) => {
      const v = Math.floor((x / 63) * 255);
      return [v, v, 200];
    });
    const noisy = makeImageData(64, 64, (x, y) => {
      const v = (x * 37 + y * 91) % 256;
      return [v, (v * 3) % 256, (v * 7) % 256];
    });

    const smoothScore = analyzeSpectral(smooth).score;
    const noisyScore = analyzeSpectral(noisy).score;
    expect(smoothScore).toBeGreaterThan(noisyScore);
  });

  it("returns neutral for tiny images", () => {
    const tiny = makeImageData(16, 16, () => [10, 20, 30]);
    expect(analyzeSpectral(tiny).detail).toBe("too-small");
  });

  it("flags dark-mode UI chrome as digital UI, not smooth AI blobs", () => {
    const ui = makeImageData(128, 96, (x, y) => {
      if (x < 36) {
        if (y > 16 && y < 28 && x > 8 && x < 28) return [40, 120, 220];
        if (y > 20 && y < 24 && x > 32 && x < 34) return [220, 220, 230];
        return [22, 22, 26];
      }
      if (y > 30 && y < 70 && x > 48 && x < 110) {
        if (y > 40 && y < 44 && x > 56 && x < 100) return [235, 235, 240];
        return [40, 40, 48];
      }
      return [18, 18, 22];
    });
    const smooth = makeImageData(128, 96, (x, y) => {
      const v = Math.floor(((x + y) / 220) * 255);
      return [v, Math.floor(v * 0.9), 180];
    });
    const uiFeats = analyzeSpectral(ui).features;
    const smoothFeats = analyzeSpectral(smooth).features;
    expect(looksLikeDigitalUi(uiFeats)).toBe(true);
    expect(looksLikeDigitalUi(smoothFeats)).toBe(false);
  });

  it("flags curved brand marks on solid fills as flat graphics", () => {
    // No axis chrome — only a curved white mark on black (GitHub-style poster).
    const brand = makeImageData(320, 180, (x, y) => {
      const cx = 160;
      const cy = 100;
      const dx = x - cx;
      const dy = y - cy;
      const r1 = Math.hypot(dx + 28, dy - 14);
      const r2 = Math.hypot(dx - 22, dy + 18);
      if ((r1 > 38 && r1 < 54) || (r2 > 34 && r2 < 50)) return [250, 250, 252];
      if (Math.hypot(dx, dy) < 16) return [250, 250, 252];
      return [0, 0, 0];
    });
    const smooth = makeImageData(320, 180, (x, y) => {
      const v = Math.floor(((x + y) / 500) * 255);
      return [v, Math.floor(v * 0.85), 170];
    });
    const brandFeats = analyzeSpectral(brand).features;
    const smoothFeats = analyzeSpectral(smooth).features;
    expect(looksLikeFlatGraphic(brandFeats)).toBe(true);
    expect(looksLikeNonPhotoGraphic(brandFeats)).toBe(true);
    expect(looksLikeFlatGraphic(smoothFeats)).toBe(false);
    expect(looksLikeNonPhotoGraphic(smoothFeats)).toBe(false);
  });

  it("flags UI with a flashy center hero via frame/chrome cues", () => {
    const uiHero = makeImageData(240, 160, (x, y) => {
      // Dark title bar with macOS traffic lights.
      if (y < 18) {
        if (y >= 6 && y <= 12) {
          if (x >= 10 && x <= 16) return [220, 70, 70];
          if (x >= 22 && x <= 28) return [220, 180, 60];
          if (x >= 34 && x <= 40) return [70, 180, 80];
        }
        return [40, 40, 44];
      }
      // Left sidebar.
      if (x < 48) return [28, 28, 32];
      // Colorful center hero that washes out full-frame palette stats.
      const v = Math.floor(80 + 140 * Math.sin(x / 9) * Math.cos(y / 11));
      return [v, (v * 2) % 255, 255 - (v % 200)];
    });
    const feats = analyzeSpectral(uiHero).features;
    expect(feats.windowChromeScore).toBeGreaterThanOrEqual(0.7);
    expect(looksLikeDigitalUi(feats)).toBe(true);
  });

  it("treats high-texture multi-color scenes as photographic", () => {
    expect(
      looksPhotographic({
        highFreqEnergyRatio: 0.44,
        laplacianVariance: 1800,
        chromaFlatness: 0.42,
        quantizedColorCount: 180,
        blockiness: 0.35,
      }),
    ).toBe(true);
    expect(
      looksPhotographic({
        highFreqEnergyRatio: 0.2,
        laplacianVariance: 200,
        chromaFlatness: 0.8,
        quantizedColorCount: 40,
        blockiness: 0.1,
      }),
    ).toBe(false);
  });

  it("flags UI FP fixtures (scatter / terminal / code) without neon promote", async () => {
    const dir = path.resolve("tests/fixtures/images/ui-fp");
    for (const name of [
      "scatter_chart.png",
      "scatter_tweet.jpg",
      "bar_chart.png",
      "terminal.png",
      "code_card.png",
      "llm_perf_eval.jpg",
      "llm_perf_user_officechai.png",
      "llm_perf_user_6panel.png",
      "laptop_screen.jpg",
      "desktop_screenshot.png",
    ]) {
      const buf = readFileSync(path.join(dir, name));
      const bytes = buf.buffer.slice(
        buf.byteOffset,
        buf.byteOffset + buf.byteLength,
      );
      const decoded = await decodeImageBytes(
        bytes,
        guessMimeType(new Uint8Array(bytes)) ?? "image/png",
      );
      const spectralImage = await rasterizeForSpectral(decoded.bitmap);
      const feats = analyzeSpectral(spectralImage).features;
      const held = looksLikeNonPhotoGraphic(feats);
      expect(held, name).toBe(true);
      expect(looksLikeNeonAiSubject(feats), name).toBe(false);
    }
  });
});
