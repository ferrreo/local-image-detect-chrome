import { asAiConfidence, type AiConfidence } from "../shared/types";
import { VISIBLE_WATERMARK_PHRASES } from "./ai-watermarks";

export type VisibleWatermarkHit = {
  score: AiConfidence;
  detail: string;
  shortCircuit: boolean;
};

/** Compact 5×7 glyphs for A–Z, 0–9, and a few punctuation marks. */
const FONT_W = 5;
const FONT_H = 7;
const FONT: Readonly<Record<string, string>> = {
  A: "01110100011000111111100011000110001",
  B: "11110100011111010001100011111000000",
  C: "01111100001000010000100000111100000",
  D: "11110100011000110001100011111000000",
  E: "11111100001111010000100001111100000",
  F: "11111100001111010000100001000000000",
  G: "01111100001000010111100011011100000",
  H: "10001100011111110001100011000100000",
  I: "11111001000010000100001001111100000",
  J: "00111000100001000010000101111000000",
  K: "10001100101110010100100101000100000",
  L: "10000100001000010000100001111100000",
  M: "10001110111010110001100011000100000",
  N: "10001110011010110011100011000100000",
  O: "01110100011000110001100010111000000",
  P: "11110100011111010000100001000000000",
  Q: "01110100011000110001101010111000001",
  R: "11110100011111010001100011000100000",
  S: "01111100000111000001000011111000000",
  T: "11111001000010000100001000010000000",
  U: "10001100011000110001100010111000000",
  V: "10001100011000101010010100010000000",
  W: "10001100011000110101110111000100000",
  X: "10001101010010000100101011000100000",
  Y: "10001100010101000100001000010000000",
  Z: "11111000100010001000100001111100000",
  "0": "01110100011000110001100010111000000",
  "1": "00100011000010000100001001111100000",
  "2": "01110100010000100100010001111100000",
  "3": "11111000100011000001000011111000000",
  "4": "10001100011111100001000010000100000",
  "5": "11111100001111000001000011111000000",
  "6": "01111100001111010001100010111000000",
  "7": "11111000010001000100010001000000000",
  "8": "01110100010111010001100010111000000",
  "9": "01110100011000101111000011111000000",
  "·": "00000000000010000100000000000000000",
  ".": "00000000000000000000001000010000000",
  "-": "00000000000111100000000000000000000",
};

type Region = { x0: number; y0: number; x1: number; y1: number };

function watermarkRegions(width: number, height: number): Region[] {
  const bw = Math.max(32, Math.floor(width * 0.4));
  const bh = Math.max(20, Math.floor(height * 0.22));
  const bottomH = Math.max(22, Math.floor(height * 0.18));
  return [
    { x0: 0, y0: 0, x1: bw, y1: bh },
    { x0: width - bw, y0: 0, x1: width, y1: bh },
    { x0: 0, y0: height - bh, x1: bw, y1: height },
    { x0: width - bw, y0: height - bh, x1: width, y1: height },
    {
      x0: Math.floor(width * 0.15),
      y0: height - bottomH,
      x1: Math.floor(width * 0.85),
      y1: height,
    },
    // Grok / xAI often parks a compact mark mid-bottom-right.
    {
      x0: Math.floor(width * 0.55),
      y0: Math.floor(height * 0.7),
      x1: width,
      y1: height,
    },
  ];
}

function regionGray(
  data: Uint8ClampedArray,
  width: number,
  region: Region,
): { gray: Float32Array; w: number; h: number } {
  const w = region.x1 - region.x0;
  const h = region.y1 - region.y0;
  const gray = new Float32Array(w * h);
  for (let y = 0; y < h; y += 1) {
    for (let x = 0; x < w; x += 1) {
      const i = ((region.y0 + y) * width + (region.x0 + x)) * 4;
      const r = data[i] ?? 0;
      const g = data[i + 1] ?? 0;
      const b = data[i + 2] ?? 0;
      gray[y * w + x] = 0.299 * r + 0.587 * g + 0.114 * b;
    }
  }
  return { gray, w, h };
}

function binarize(
  gray: Float32Array,
  invert: boolean,
): Uint8Array {
  let sum = 0;
  for (let i = 0; i < gray.length; i += 1) sum += gray[i] ?? 0;
  const mean = sum / Math.max(1, gray.length);
  const out = new Uint8Array(gray.length);
  for (let i = 0; i < gray.length; i += 1) {
    const lit = (gray[i] ?? 0) >= mean;
    out[i] = (invert ? !lit : lit) ? 1 : 0;
  }
  return out;
}

function rowDensity(bin: Uint8Array, w: number, h: number): number[] {
  const dens = new Array<number>(h).fill(0);
  for (let y = 0; y < h; y += 1) {
    let c = 0;
    for (let x = 0; x < w; x += 1) c += bin[y * w + x] ?? 0;
    dens[y] = c / w;
  }
  return dens;
}

function findTextBands(
  dens: number[],
  minFrac = 0.02,
  maxFrac = 0.55,
): Array<{ y0: number; y1: number }> {
  const bands: Array<{ y0: number; y1: number }> = [];
  let start = -1;
  for (let y = 0; y <= dens.length; y += 1) {
    const on =
      y < dens.length &&
      (dens[y] ?? 0) >= minFrac &&
      (dens[y] ?? 0) <= maxFrac;
    if (on && start < 0) start = y;
    if (!on && start >= 0) {
      if (y - start >= 5) bands.push({ y0: start, y1: y });
      start = -1;
    }
  }
  return bands.slice(0, 4);
}

function extractGlyphs(
  bin: Uint8Array,
  w: number,
  y0: number,
  y1: number,
): Uint8Array[] {
  const bandH = y1 - y0;
  const col = new Array<number>(w).fill(0);
  for (let x = 0; x < w; x += 1) {
    let c = 0;
    for (let y = y0; y < y1; y += 1) c += bin[y * w + x] ?? 0;
    col[x] = c;
  }
  const glyphs: Uint8Array[] = [];
  let x = 0;
  while (x < w) {
    while (x < w && (col[x] ?? 0) === 0) x += 1;
    if (x >= w) break;
    const x0 = x;
    while (x < w && (col[x] ?? 0) > 0) x += 1;
    const x1 = x;
    const gw = x1 - x0;
    if (gw < 2 || gw > bandH * 2.2) continue;
    const g = new Uint8Array(FONT_W * FONT_H);
    for (let gy = 0; gy < FONT_H; gy += 1) {
      for (let gx = 0; gx < FONT_W; gx += 1) {
        const sx = x0 + Math.floor((gx + 0.5) * (gw / FONT_W));
        const sy = y0 + Math.floor((gy + 0.5) * (bandH / FONT_H));
        g[gy * FONT_W + gx] = bin[sy * w + sx] ?? 0;
      }
    }
    glyphs.push(g);
    if (glyphs.length > 28) break;
  }
  return glyphs;
}

function scoreGlyph(glyph: Uint8Array, pattern: string): number {
  let match = 0;
  const n = Math.min(glyph.length, pattern.length);
  for (let i = 0; i < n; i += 1) {
    const bit = pattern[i] === "1" ? 1 : 0;
    if ((glyph[i] ?? 0) === bit) match += 1;
  }
  return match / Math.max(1, n);
}

function classifyGlyph(glyph: Uint8Array): string {
  let best = "?";
  let bestScore = 0.58;
  for (const [ch, pattern] of Object.entries(FONT)) {
    const s = scoreGlyph(glyph, pattern);
    if (s > bestScore) {
      bestScore = s;
      best = ch;
    }
  }
  return best;
}

function normalizeOcr(text: string): string {
  return text
    .toLowerCase()
    .replace(/[^a-z0-9·-]/g, "")
    .replace(/[·-]/g, "");
}

function matchPhrase(ocrNorm: string): string | undefined {
  if (ocrNorm.length < 3) return undefined;
  for (const phrase of VISIBLE_WATERMARK_PHRASES) {
    const p = phrase.replace(/[·\s_-]/g, "").toLowerCase();
    if (p.length < 3) continue;
    if (ocrNorm.includes(p)) return phrase;
    // Allow one-char OCR slip for short brands (grok / xai / bing).
    if (p.length <= 5 && fuzzyIncludes(ocrNorm, p, 1)) return phrase;
    if (p.length > 5 && fuzzyIncludes(ocrNorm, p, 2)) return phrase;
  }
  return undefined;
}

function fuzzyIncludes(hay: string, needle: string, maxDist: number): boolean {
  if (hay.includes(needle)) return true;
  const n = needle.length;
  if (hay.length < n - maxDist) return false;
  for (let i = 0; i <= hay.length - n + maxDist; i += 1) {
    const slice = hay.slice(i, i + n + maxDist);
    if (editDistance(slice.slice(0, n), needle) <= maxDist) return true;
    if (n + 1 <= slice.length && editDistance(slice.slice(0, n + 1), needle) <= maxDist)
      return true;
  }
  return false;
}

function editDistance(a: string, b: string): number {
  const m = a.length;
  const n = b.length;
  const dp = new Array<number>(n + 1);
  for (let j = 0; j <= n; j += 1) dp[j] = j;
  for (let i = 1; i <= m; i += 1) {
    let prev = dp[0]!;
    dp[0] = i;
    for (let j = 1; j <= n; j += 1) {
      const tmp = dp[j]!;
      const cost = a[i - 1] === b[j - 1] ? 0 : 1;
      dp[j] = Math.min(dp[j]! + 1, dp[j - 1]! + 1, prev + cost);
      prev = tmp;
    }
  }
  return dp[n]!;
}

function scanRegion(
  data: Uint8ClampedArray,
  width: number,
  region: Region,
): string | undefined {
  const { gray, w, h } = regionGray(data, width, region);
  if (w < 20 || h < 10) return undefined;

  for (const invert of [false, true]) {
    const bin = binarize(gray, invert);
    const dens = rowDensity(bin, w, h);
    for (const band of findTextBands(dens)) {
      const glyphs = extractGlyphs(bin, w, band.y0, band.y1);
      if (glyphs.length < 3) continue;
      const raw = glyphs.map(classifyGlyph).join("");
      const hit = matchPhrase(normalizeOcr(raw));
      if (hit) return hit;
    }
  }
  return undefined;
}

/**
 * Detect burned-in AI generator watermarks in corner / footer bands.
 * High precision, low recall — meant as a provenance short-circuit, not a
 * substitute for visual heads. SynthID and other invisible marks are out of
 * scope (proprietary detectors).
 */
export function analyzeVisibleWatermark(
  imageData: ImageData,
): VisibleWatermarkHit {
  const { data, width, height } = imageData;
  if (width < 64 || height < 64) {
    return {
      score: asAiConfidence(0.5),
      detail: "visible-watermark:too-small",
      shortCircuit: false,
    };
  }

  for (const region of watermarkRegions(width, height)) {
    const hit = scanRegion(data, width, region);
    if (hit) {
      return {
        score: asAiConfidence(0.99),
        detail: `visible-watermark:${hit}`,
        shortCircuit: true,
      };
    }
  }

  return {
    score: asAiConfidence(0.5),
    detail: "visible-watermark:none",
    shortCircuit: false,
  };
}
