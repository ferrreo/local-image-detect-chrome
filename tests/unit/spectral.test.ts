import { describe, expect, it } from "vitest";
import {
  analyzeSpectral,
  fftMagnitude,
} from "../../extension/src/lib/spectral";

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
});
