import { asAiConfidence, type AiConfidence } from "../shared/types";

export type SpectralAnalysis = {
  score: AiConfidence;
  detail: string;
  features: {
    highFreqEnergyRatio: number;
    laplacianVariance: number;
    chromaFlatness: number;
    blockiness: number;
  };
};

function clamp01(value: number): number {
  return Math.min(1, Math.max(0, value));
}

function toGray(data: Uint8ClampedArray, width: number, height: number): Float32Array {
  const gray = new Float32Array(width * height);
  for (let i = 0, p = 0; i < gray.length; i += 1, p += 4) {
    const r = data[p] ?? 0;
    const g = data[p + 1] ?? 0;
    const b = data[p + 2] ?? 0;
    gray[i] = 0.299 * r + 0.587 * g + 0.114 * b;
  }
  return gray;
}

function mean(values: Float32Array): number {
  let sum = 0;
  for (let i = 0; i < values.length; i += 1) sum += values[i] ?? 0;
  return values.length === 0 ? 0 : sum / values.length;
}

function variance(values: Float32Array, avg?: number): number {
  if (values.length === 0) return 0;
  const m = avg ?? mean(values);
  let acc = 0;
  for (let i = 0; i < values.length; i += 1) {
    const d = (values[i] ?? 0) - m;
    acc += d * d;
  }
  return acc / values.length;
}

/** Compact radix-2 real FFT magnitude spectrum for power-of-two lengths. */
export function fftMagnitude(signal: Float32Array): Float32Array {
  const n = signal.length;
  if (n === 0 || (n & (n - 1)) !== 0) {
    throw new Error(`fftMagnitude requires power-of-two length, got ${n}`);
  }

  const re = new Float32Array(n);
  const im = new Float32Array(n);
  re.set(signal);

  // Bit-reversal permutation
  for (let i = 1, j = 0; i < n; i += 1) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      const tr = re[i] ?? 0;
      re[i] = re[j] ?? 0;
      re[j] = tr;
      const ti = im[i] ?? 0;
      im[i] = im[j] ?? 0;
      im[j] = ti;
    }
  }

  for (let len = 2; len <= n; len <<= 1) {
    const ang = (-2 * Math.PI) / len;
    const wlenRe = Math.cos(ang);
    const wlenIm = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let wRe = 1;
      let wIm = 0;
      for (let j = 0; j < len / 2; j += 1) {
        const uRe = re[i + j] ?? 0;
        const uIm = im[i + j] ?? 0;
        const vRe0 = re[i + j + len / 2] ?? 0;
        const vIm0 = im[i + j + len / 2] ?? 0;
        const vRe = vRe0 * wRe - vIm0 * wIm;
        const vIm = vRe0 * wIm + vIm0 * wRe;
        re[i + j] = uRe + vRe;
        im[i + j] = uIm + vIm;
        re[i + j + len / 2] = uRe - vRe;
        im[i + j + len / 2] = uIm - vIm;
        const nextWRe = wRe * wlenRe - wIm * wlenIm;
        wIm = wRe * wlenIm + wIm * wlenRe;
        wRe = nextWRe;
      }
    }
  }

  const mag = new Float32Array(n / 2);
  for (let i = 0; i < mag.length; i += 1) {
    const r = re[i] ?? 0;
    const m = im[i] ?? 0;
    mag[i] = Math.hypot(r, m);
  }
  return mag;
}

function nextPowerOfTwo(n: number): number {
  let p = 1;
  while (p < n) p <<= 1;
  return p;
}

function sampleRow(gray: Float32Array, width: number, row: number): Float32Array {
  const out = new Float32Array(width);
  const base = row * width;
  for (let x = 0; x < width; x += 1) out[x] = gray[base + x] ?? 0;
  return out;
}

function laplacianVariance(
  gray: Float32Array,
  width: number,
  height: number,
): number {
  if (width < 3 || height < 3) return 0;
  const vals = new Float32Array((width - 2) * (height - 2));
  let k = 0;
  for (let y = 1; y < height - 1; y += 1) {
    for (let x = 1; x < width - 1; x += 1) {
      const c = gray[y * width + x] ?? 0;
      const n = gray[(y - 1) * width + x] ?? 0;
      const s = gray[(y + 1) * width + x] ?? 0;
      const w = gray[y * width + (x - 1)] ?? 0;
      const e = gray[y * width + (x + 1)] ?? 0;
      vals[k] = Math.abs(4 * c - n - s - w - e);
      k += 1;
    }
  }
  return variance(vals);
}

function chromaFlatness(
  data: Uint8ClampedArray,
  width: number,
  height: number,
): number {
  const n = width * height;
  if (n === 0) return 0;
  let sum = 0;
  for (let i = 0, p = 0; i < n; i += 1, p += 4) {
    const r = data[p] ?? 0;
    const g = data[p + 1] ?? 0;
    const b = data[p + 2] ?? 0;
    const mx = Math.max(r, g, b);
    const mn = Math.min(r, g, b);
    sum += mx === 0 ? 0 : (mx - mn) / mx;
  }
  // Low chroma variation / over-smooth color often appears in generative images.
  return 1 - sum / n;
}

function jpegBlockiness(
  gray: Float32Array,
  width: number,
  height: number,
): number {
  if (width < 16 || height < 16) return 0;
  let edge = 0;
  let interior = 0;
  let edgeCount = 0;
  let interiorCount = 0;

  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width - 1; x += 1) {
      const diff = Math.abs((gray[y * width + x] ?? 0) - (gray[y * width + x + 1] ?? 0));
      if ((x + 1) % 8 === 0) {
        edge += diff;
        edgeCount += 1;
      } else {
        interior += diff;
        interiorCount += 1;
      }
    }
  }

  const edgeMean = edgeCount === 0 ? 0 : edge / edgeCount;
  const interiorMean = interiorCount === 0 ? 1 : interior / interiorCount;
  return clamp01(edgeMean / (interiorMean + 1e-3) / 4);
}

function highFreqEnergyRatio(gray: Float32Array, width: number, height: number): number {
  const row = sampleRow(gray, width, Math.floor(height / 2));
  const n = nextPowerOfTwo(row.length);
  const padded = new Float32Array(n);
  padded.set(row);
  // Remove DC bias
  const m = mean(padded);
  for (let i = 0; i < padded.length; i += 1) padded[i] = (padded[i] ?? 0) - m;

  const mag = fftMagnitude(padded);
  let low = 0;
  let high = 0;
  const split = Math.floor(mag.length * 0.35);
  for (let i = 1; i < mag.length; i += 1) {
    const v = mag[i] ?? 0;
    if (i < split) low += v;
    else high += v;
  }
  const total = low + high + 1e-6;
  return high / total;
}

/**
 * Hand-crafted spectral / noise features that complement learned classifiers.
 * Diffusion and GAN images often show oversmoothed high frequencies, unnatural
 * chroma flatness, and atypical block energy.
 */
export function analyzeSpectral(
  imageData: ImageData,
): SpectralAnalysis {
  const { data, width, height } = imageData;
  if (width < 32 || height < 32) {
    return {
      score: asAiConfidence(0.5),
      detail: "too-small",
      features: {
        highFreqEnergyRatio: 0,
        laplacianVariance: 0,
        chromaFlatness: 0,
        blockiness: 0,
      },
    };
  }

  const gray = toGray(data, width, height);
  const hf = highFreqEnergyRatio(gray, width, height);
  const lap = laplacianVariance(gray, width, height);
  const flat = chromaFlatness(data, width, height);
  const block = jpegBlockiness(gray, width, height);

  // Map features into P(AI). Low high-frequency energy + high chroma flatness
  // push toward AI; strong natural texture and moderate blockiness push toward real.
  const textureScore = clamp01(1 - lap / 120);
  const hfScore = clamp01(1 - hf / 0.55);
  const flatScore = clamp01((flat - 0.35) / 0.45);
  const blockScore = clamp01((block - 0.25) / 0.5);

  const raw =
    0.34 * hfScore + 0.28 * textureScore + 0.22 * flatScore + 0.16 * blockScore;
  const score = asAiConfidence(0.15 + 0.7 * raw);

  return {
    score,
    detail: `hf=${hf.toFixed(3)},lap=${lap.toFixed(1)},flat=${flat.toFixed(3)},block=${block.toFixed(3)}`,
    features: {
      highFreqEnergyRatio: hf,
      laplacianVariance: lap,
      chromaFlatness: flat,
      blockiness: block,
    },
  };
}
