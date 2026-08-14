import { asAiConfidence, type AiConfidence } from "../shared/types";

export type SpectralFeatures = {
  highFreqEnergyRatio: number;
  laplacianVariance: number;
  chromaFlatness: number;
  blockiness: number;
  /** Fraction of strong edges that are horizontal or vertical (UI chrome). */
  axisAlignedEdgeRatio: number;
  /** Unique colors after 4-bit/channel quantization. */
  quantizedColorCount: number;
  /** Share of pixels in the 8 most common quantized colors. */
  topColorShare: number;
  /** Title-bar + sidebar band axis-edge ratio (hero-safe UI cue). */
  frameAxisAlignedEdgeRatio: number;
  frameTopColorShare: number;
  frameQuantizedColorCount: number;
  /** 0–1 confidence that macOS traffic lights sit in the title bar. */
  windowChromeScore: number;
};

export type SpectralAnalysis = {
  score: AiConfidence;
  detail: string;
  features: SpectralFeatures;
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

/**
 * UI screenshots: sharp axis-aligned chrome + tiny quantized palette.
 * Smooth AI art also has few colors but almost no axis edges.
 * When a flashy hero/render fills the center, full-frame stats wash out —
 * so we also score title-bar / sidebar bands and macOS traffic lights.
 */
function digitalUiGeometry(
  data: Uint8ClampedArray,
  width: number,
  height: number,
): {
  axisAlignedEdgeRatio: number;
  quantizedColorCount: number;
  topColorShare: number;
  frameAxisAlignedEdgeRatio: number;
  frameTopColorShare: number;
  frameQuantizedColorCount: number;
  windowChromeScore: number;
} {
  let hEdge = 0;
  let vEdge = 0;
  let otherEdge = 0;
  let frameH = 0;
  let frameV = 0;
  let frameOther = 0;
  const counts = new Map<number, number>();
  const frameCounts = new Map<number, number>();
  const quant = (v: number) => v >> 4;
  const thr = 40;
  const leftBand = Math.max(8, Math.floor(width * 0.2));
  const topBand = Math.max(8, Math.floor(height * 0.12));

  for (let y = 1; y < height - 1; y += 1) {
    for (let x = 1; x < width - 1; x += 1) {
      const i = (y * width + x) * 4;
      const r = data[i] ?? 0;
      const g = data[i + 1] ?? 0;
      const b = data[i + 2] ?? 0;
      const key = (quant(r) << 8) | (quant(g) << 4) | quant(b);
      counts.set(key, (counts.get(key) ?? 0) + 1);

      const inFrame = x < leftBand || y < topBand;
      if (inFrame) {
        frameCounts.set(key, (frameCounts.get(key) ?? 0) + 1);
      }

      const right = i + 4;
      const below = i + width * 4;
      const gx =
        Math.abs(r - (data[right] ?? 0)) +
        Math.abs(g - (data[right + 1] ?? 0)) +
        Math.abs(b - (data[right + 2] ?? 0));
      const gy =
        Math.abs(r - (data[below] ?? 0)) +
        Math.abs(g - (data[below + 1] ?? 0)) +
        Math.abs(b - (data[below + 2] ?? 0));

      let kind: "h" | "v" | "o" | null = null;
      if (gx >= thr && gy < thr * 0.5) kind = "h";
      else if (gy >= thr && gx < thr * 0.5) kind = "v";
      else if (gx >= thr && gy >= thr) kind = "o";

      if (kind === "h") {
        hEdge += 1;
        if (inFrame) frameH += 1;
      } else if (kind === "v") {
        vEdge += 1;
        if (inFrame) frameV += 1;
      } else if (kind === "o") {
        otherEdge += 1;
        if (inFrame) frameOther += 1;
      }
    }
  }

  const edgeTotal = hEdge + vEdge + otherEdge;
  const axisAlignedEdgeRatio =
    edgeTotal === 0 ? 0 : (hEdge + vEdge) / edgeTotal;
  const frameEdgeTotal = frameH + frameV + frameOther;
  const frameAxisAlignedEdgeRatio =
    frameEdgeTotal === 0 ? 0 : (frameH + frameV) / frameEdgeTotal;

  const summarize = (map: Map<number, number>) => {
    let pixels = 0;
    const sorted: number[] = [];
    for (const c of map.values()) {
      sorted.push(c);
      pixels += c;
    }
    sorted.sort((a, b) => b - a);
    const top = sorted.slice(0, 8).reduce((a, b) => a + b, 0);
    return {
      quantizedColorCount: map.size,
      topColorShare: pixels === 0 ? 0 : top / pixels,
    };
  };

  const full = summarize(counts);
  const frame = summarize(frameCounts);

  return {
    axisAlignedEdgeRatio,
    quantizedColorCount: full.quantizedColorCount,
    topColorShare: full.topColorShare,
    frameAxisAlignedEdgeRatio,
    frameTopColorShare: frame.topColorShare,
    frameQuantizedColorCount: frame.quantizedColorCount,
    windowChromeScore: macTrafficLightScore(data, width, height),
  };
}

/** macOS close/minimize/zoom dots in the title-bar corner. */
function macTrafficLightScore(
  data: Uint8ClampedArray,
  width: number,
  height: number,
): number {
  const maxY = Math.min(height, Math.max(12, Math.floor(height * 0.1)));
  const maxX = Math.min(width, Math.max(48, Math.floor(width * 0.2)));
  const red: number[] = [];
  const yellow: number[] = [];
  const green: number[] = [];

  for (let y = 0; y < maxY; y += 1) {
    for (let x = 0; x < maxX; x += 1) {
      const i = (y * width + x) * 4;
      const r = data[i] ?? 0;
      const g = data[i + 1] ?? 0;
      const b = data[i + 2] ?? 0;
      // Traffic lights are saturated and relatively bright vs dark title bars.
      if (r > 140 && r > g + 40 && r > b + 40) red.push(x);
      else if (r > 140 && g > 100 && g > b + 30 && r > b + 30) yellow.push(x);
      else if (g > 120 && g > r + 25 && g > b + 15) green.push(x);
    }
  }

  if (red.length < 4 || yellow.length < 4 || green.length < 4) return 0;

  const median = (xs: number[]) => {
    const s = [...xs].sort((a, b) => a - b);
    return s[Math.floor(s.length / 2)] ?? 0;
  };
  const rx = median(red);
  const yx = median(yellow);
  const gx = median(green);
  // Expected left-to-right: red, yellow, green with similar spacing.
  if (!(rx < yx && yx < gx)) return 0;
  const d1 = yx - rx;
  const d2 = gx - yx;
  if (d1 < 4 || d2 < 4) return 0;
  const spacingOk = Math.abs(d1 - d2) <= Math.max(4, 0.45 * Math.max(d1, d2));
  if (!spacingOk) return 0;
  // Strong when clusters are compact (true circular buttons).
  const spread = (xs: number[], m: number) => {
    let acc = 0;
    for (const v of xs) acc += Math.abs(v - m);
    return acc / xs.length;
  };
  const compact =
    spread(red, rx) < 10 && spread(yellow, yx) < 10 && spread(green, gx) < 10;
  return compact ? 1 : 0.7;
}

/**
 * True for software UI screenshots: full-frame chrome, dense tables /
 * dashboards (Activity Monitor, usage meters), text docs, title/sidebar bands,
 * or macOS dots.
 */
export function looksLikeDigitalUi(features: {
  axisAlignedEdgeRatio: number;
  quantizedColorCount: number;
  topColorShare: number;
  chromaFlatness?: number;
  highFreqEnergyRatio?: number;
  laplacianVariance?: number;
  frameAxisAlignedEdgeRatio?: number;
  frameTopColorShare?: number;
  frameQuantizedColorCount?: number;
  windowChromeScore?: number;
}): boolean {
  if ((features.windowChromeScore ?? 0) >= 0.7) return true;

  const axis = features.axisAlignedEdgeRatio;
  const colors = features.quantizedColorCount;
  const share = features.topColorShare;
  const chroma = features.chromaFlatness ?? 0;
  const hf = features.highFreqEnergyRatio ?? 1;
  const lap = features.laplacianVariance ?? 0;

  // Classic chrome: hard edges, tiny palette, solid fills.
  if (axis >= 0.78 && colors > 0 && colors <= 96 && share >= 0.75) {
    return true;
  }

  // Dense tables / process lists / settings panes — lots of H/V rules,
  // limited palette, not camera grain (Activity Monitor FPs).
  // Axis floor is high so neon CGI grid landscapes do not match.
  if (
    axis >= 0.72 &&
    colors > 0 &&
    colors <= 140 &&
    share >= 0.65 &&
    chroma >= 0.55 &&
    hf <= 0.4
  ) {
    return true;
  }

  // Soft dark UI cards (usage meters, progress bars, dashboards).
  if (
    axis >= 0.62 &&
    colors > 0 &&
    colors <= 48 &&
    share >= 0.78 &&
    chroma >= 0.65 &&
    hf <= 0.35
  ) {
    return true;
  }

  // OS notification / toast banners (macOS Login Items, etc.): dominant dark
  // fill, glyph-scale HF/laplacian from text — not neon circuit stock (lower
  // topShare / smoother lap) or colorful CGI heads (low topShare).
  if (
    share >= 0.74 &&
    colors > 0 &&
    colors <= 120 &&
    chroma >= 0.55 &&
    hf <= 0.35 &&
    lap >= 500 &&
    lap <= 8_000 &&
    axis >= 0.32 &&
    axis <= 0.72
  ) {
    return true;
  }

  // Text-heavy screenshots (light/dark docs, roast cards). Solid page fill
  // (high topShare) separates these from colorful neon CGI heads.
  if (
    axis >= 0.6 &&
    colors > 0 &&
    colors <= 100 &&
    share >= 0.68 &&
    chroma >= 0.45 &&
    hf <= 0.38 &&
    lap >= 300 &&
    lap <= 6_000
  ) {
    return true;
  }

  const frameAxis = features.frameAxisAlignedEdgeRatio ?? 0;
  const frameShare = features.frameTopColorShare ?? 0;
  const frameColors = features.frameQuantizedColorCount ?? 999;
  // Sidebar/title-bar only — survives a colorful center hero/render.
  return (
    frameAxis >= 0.82 &&
    frameColors > 0 &&
    frameColors <= 64 &&
    frameShare >= 0.8
  );
}

/**
 * Brand marks / marketing motion posters: huge solid fills, tiny palette,
 * hard logo edges — not soft generative gradients (near-zero laplacian).
 */
export function looksLikeFlatGraphic(features: {
  quantizedColorCount: number;
  topColorShare: number;
  chromaFlatness: number;
  laplacianVariance: number;
  highFreqEnergyRatio: number;
  axisAlignedEdgeRatio?: number;
}): boolean {
  if (
    features.quantizedColorCount > 0 &&
    features.quantizedColorCount <= 48 &&
    features.topColorShare >= 0.88 &&
    features.chromaFlatness >= 0.8 &&
    features.laplacianVariance >= 60 &&
    features.laplacianVariance <= 12_000 &&
    features.highFreqEnergyRatio <= 0.42
  ) {
    return true;
  }

  // Geometric / dithered design-system posters (ePaper swatch grids): strong
  // axis lattice, small palette, not organic CGI. Axis stays high so neon
  // creature heads (lower axis) do not match.
  const axis = features.axisAlignedEdgeRatio ?? 0;
  return (
    axis >= 0.58 &&
    features.quantizedColorCount > 0 &&
    features.quantizedColorCount <= 72 &&
    features.topColorShare >= 0.5 &&
    features.chromaFlatness >= 0.45 &&
    features.laplacianVariance >= 80 &&
    features.laplacianVariance <= 14_000 &&
    features.highFreqEnergyRatio <= 0.5
  );
}

/**
 * Bar / line charts and KPI infographics: strong H/V bars + page fill +
 * compact palette. Visual heads often score these as AI ~90%+.
 * Includes social screenshots where a dark chrome frame lowers topColorShare.
 */
export function looksLikeChartOrInfographic(features: {
  axisAlignedEdgeRatio: number;
  quantizedColorCount: number;
  topColorShare: number;
  chromaFlatness: number;
  highFreqEnergyRatio: number;
  laplacianVariance: number;
  blockiness?: number;
  frameAxisAlignedEdgeRatio?: number;
  frameTopColorShare?: number;
}): boolean {
  const block = features.blockiness ?? 0;
  const axis = features.axisAlignedEdgeRatio;
  const colors = features.quantizedColorCount;
  const share = features.topColorShare;
  const chroma = features.chromaFlatness;
  const hf = features.highFreqEnergyRatio;
  const lap = features.laplacianVariance;
  if (colors <= 0 || block > 0.5) return false;

  // Clean light-bg chart / KPI card (tight axis so mid-axis neon heads miss).
  if (
    axis >= 0.55 &&
    colors <= 90 &&
    share >= 0.5 &&
    chroma >= 0.42 &&
    hf <= 0.5 &&
    lap >= 60 &&
    lap <= 18_000
  ) {
    return true;
  }

  // Tweet / dark-UI framed chart: surround dilutes topShare; bars keep axis high.
  // Color ceiling stays below busy neon CGI palettes (~95–140).
  if (
    axis >= 0.58 &&
    colors <= 80 &&
    share >= 0.22 &&
    share < 0.5 &&
    chroma >= 0.4 &&
    hf <= 0.45 &&
    lap >= 150 &&
    lap <= 16_000
  ) {
    return true;
  }

  // Frame-only cue when the card is centered in dark chrome.
  const frameAxis = features.frameAxisAlignedEdgeRatio ?? 0;
  const frameShare = features.frameTopColorShare ?? 0;
  return (
    frameAxis >= 0.7 &&
    frameShare >= 0.55 &&
    axis >= 0.5 &&
    colors <= 100 &&
    chroma >= 0.42 &&
    hf <= 0.5 &&
    lap >= 80 &&
    lap <= 16_000
  );
}

/**
 * Strong photographic capture cues — product macros / real scenes that accurate
 * heads sometimes mislabel as AI. Requires JPEG-ish blockiness so detailed CGI
 * / AI "slop" with busy texture does not get held as a camera photo.
 * Chroma ceiling is tight: neon CGI heads (Craiyon) often sit ~0.5–0.7 and
 * must NOT count as camera photos.
 */
export function looksPhotographic(features: {
  highFreqEnergyRatio: number;
  laplacianVariance: number;
  chromaFlatness: number;
  quantizedColorCount: number;
  blockiness?: number;
}): boolean {
  const block = features.blockiness ?? 0;
  // Generative smoothness / neon glow — never "camera".
  if (features.chromaFlatness >= 0.48) return false;
  return (
    features.highFreqEnergyRatio >= 0.38 &&
    features.laplacianVariance >= 800 &&
    features.chromaFlatness <= 0.47 &&
    features.quantizedColorCount >= 100 &&
    block >= 0.25 &&
    block <= 0.55
  );
}

/**
 * Neon / CGI AI-stock subjects (digital heads, aliens, robot hands): colorful
 * generative chroma without chrome-level axis lattices. Checked before UI holds
 * so perspective-grid backgrounds cannot bury them as "docs screenshots".
 * Bands are wide — Google thumbs are blocky; sparkle/bokeh CGI lifts HF a lot.
 */
export function looksLikeNeonAiSubject(features: {
  highFreqEnergyRatio: number;
  laplacianVariance: number;
  chromaFlatness: number;
  quantizedColorCount: number;
  topColorShare: number;
  axisAlignedEdgeRatio: number;
  blockiness?: number;
  windowChromeScore?: number;
}): boolean {
  if ((features.windowChromeScore ?? 0) >= 0.7) return false;
  // Hard tables / docs panes only — perspective grids under CGI heads sit ~0.55–0.65.
  if (features.axisAlignedEdgeRatio >= 0.72) return false;
  // Geometric swatch / ePaper posters: strong axis + small palette.
  if (
    features.axisAlignedEdgeRatio >= 0.62 &&
    features.quantizedColorCount <= 48
  ) {
    return false;
  }
  // Solid page / swatch fills (UI cards, brand posters).
  if (features.quantizedColorCount <= 40 && features.topColorShare >= 0.78) {
    return false;
  }
  // Text screenshots (roast cards / docs): axis lattice + page fill, mid palette.
  // Colorful neon heads over grids usually clear 80+ quantized colors.
  if (
    features.axisAlignedEdgeRatio >= 0.58 &&
    features.topColorShare >= 0.68 &&
    features.quantizedColorCount <= 80
  ) {
    return false;
  }
  // Dark UI toasts: high topShare + soft HF + text-scale laplacian.
  if (
    features.topColorShare >= 0.74 &&
    features.chromaFlatness >= 0.55 &&
    features.highFreqEnergyRatio <= 0.35 &&
    features.laplacianVariance >= 500 &&
    features.quantizedColorCount <= 120
  ) {
    return false;
  }
  // Bar charts / KPI infographics — axis lattice + page fill, not neon subjects.
  if (looksLikeChartOrInfographic(features)) return false;
  const block = features.blockiness ?? 0;
  const colorful =
    features.quantizedColorCount >= 36 ||
    (features.chromaFlatness >= 0.54 && features.quantizedColorCount >= 24);
  // Sparkly neon creature faces (Vidu-style glowing eyes / bokeh dots) often
  // land HF 0.55–0.72 and laplacian well above clean renders.
  return (
    features.chromaFlatness >= 0.45 &&
    colorful &&
    features.topColorShare <= 0.8 &&
    features.highFreqEnergyRatio <= 0.78 &&
    features.laplacianVariance >= 80 &&
    features.laplacianVariance <= 40_000 &&
    block <= 0.62
  );
}

/**
 * 3D / illustration / neon "AI slop" that is not a camera photo and not UI chrome.
 * Tuned for Google Images AI-grid thumbs: compression can lift HF a bit and
 * busy CGI can look textured, so bands are wider than a pristine render.
 * Smooth generative chroma can override weak "photo" cues (blue alien CGI).
 */
export function looksLikeSyntheticCgi(features: {
  highFreqEnergyRatio: number;
  laplacianVariance: number;
  chromaFlatness: number;
  quantizedColorCount: number;
  topColorShare: number;
  axisAlignedEdgeRatio: number;
  blockiness?: number;
  windowChromeScore?: number;
}): boolean {
  if ((features.windowChromeScore ?? 0) >= 0.7) return false;
  // Hard UI chrome only — weaker table heuristics must not veto neon CGI.
  if (features.axisAlignedEdgeRatio >= 0.78) return false;
  // Geometric swatch / ePaper design grids are posters, not CGI creatures.
  // Require a tight palette so colorful neon heads over grids still match CGI.
  if (
    features.axisAlignedEdgeRatio >= 0.62 &&
    features.quantizedColorCount <= 56 &&
    features.topColorShare >= 0.55
  ) {
    return false;
  }
  // Dark notification / solid UI fills are not generative CGI.
  if (
    features.topColorShare >= 0.74 &&
    features.chromaFlatness >= 0.55 &&
    (features.highFreqEnergyRatio ?? 1) <= 0.35 &&
    features.laplacianVariance >= 500 &&
    features.quantizedColorCount <= 120
  ) {
    return false;
  }
  if (looksLikeChartOrInfographic(features)) return false;
  if (looksLikeNeonAiSubject(features)) return true;
  const block = features.blockiness ?? 0;
  const paletteOk =
    features.quantizedColorCount >= 16 &&
    features.topColorShare <= 0.94 &&
    features.laplacianVariance >= 60 &&
    features.laplacianVariance <= 40_000;
  // Smooth generative / neon stock — wins even if HF looks busy / sparkly.
  if (
    paletteOk &&
    features.chromaFlatness >= 0.48 &&
    features.highFreqEnergyRatio <= 0.72 &&
    block <= 0.55
  ) {
    return true;
  }
  if (looksPhotographic(features)) return false;
  return (
    paletteOk &&
    features.chromaFlatness >= 0.45 &&
    features.highFreqEnergyRatio <= 0.65 &&
    block <= 0.55
  );
}

/** UI chrome or flat brand/vector graphics — visual heads often FP these as AI. */
export function looksLikeNonPhotoGraphic(features: {
  axisAlignedEdgeRatio: number;
  quantizedColorCount: number;
  topColorShare: number;
  chromaFlatness: number;
  laplacianVariance: number;
  highFreqEnergyRatio: number;
  frameAxisAlignedEdgeRatio?: number;
  frameTopColorShare?: number;
  frameQuantizedColorCount?: number;
  windowChromeScore?: number;
  blockiness?: number;
}): boolean {
  return (
    looksLikeDigitalUi(features) ||
    looksLikeFlatGraphic(features) ||
    looksLikeChartOrInfographic(features)
  );
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
        axisAlignedEdgeRatio: 0,
        quantizedColorCount: 0,
        topColorShare: 0,
        frameAxisAlignedEdgeRatio: 0,
        frameTopColorShare: 0,
        frameQuantizedColorCount: 0,
        windowChromeScore: 0,
      },
    };
  }

  const gray = toGray(data, width, height);
  const hf = highFreqEnergyRatio(gray, width, height);
  const lap = laplacianVariance(gray, width, height);
  const flat = chromaFlatness(data, width, height);
  const block = jpegBlockiness(gray, width, height);
  const ui = digitalUiGeometry(data, width, height);

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
    detail: `hf=${hf.toFixed(3)},lap=${lap.toFixed(1)},flat=${flat.toFixed(3)},block=${block.toFixed(3)},axis=${ui.axisAlignedEdgeRatio.toFixed(2)},frameAxis=${ui.frameAxisAlignedEdgeRatio.toFixed(2)},colors=${ui.quantizedColorCount},chrome=${ui.windowChromeScore.toFixed(1)}`,
    features: {
      highFreqEnergyRatio: hf,
      laplacianVariance: lap,
      chromaFlatness: flat,
      blockiness: block,
      axisAlignedEdgeRatio: ui.axisAlignedEdgeRatio,
      quantizedColorCount: ui.quantizedColorCount,
      topColorShare: ui.topColorShare,
      frameAxisAlignedEdgeRatio: ui.frameAxisAlignedEdgeRatio,
      frameTopColorShare: ui.frameTopColorShare,
      frameQuantizedColorCount: ui.frameQuantizedColorCount,
      windowChromeScore: ui.windowChromeScore,
    },
  };
}
