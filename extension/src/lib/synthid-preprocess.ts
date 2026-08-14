/**
 * OpenSynthID 6-channel preprocess (Apache-2.0 model: fyxme/opensynthid-detect-0.1).
 * Reimplements infer.py: RGB + db4 VisuShrink residual + FFT log-mag + carrier mask.
 * Not affiliated with Google DeepMind; surrogate for SynthID pixel watermarks
 * used by Gemini/Imagen and OpenAI ChatGPT image / API.
 */

export const SYNTHID_INPUT_SIZE = 512;

const DB4_DEC_LO = [
  -0.010597401785069032, 0.0328830116668852, 0.030841381835560764,
  -0.18703481171909309, -0.027983769416859854, 0.6308807679298589,
  0.7148465705529157, 0.2303778133088965,
];
const DB4_DEC_HI = [
  -0.2303778133088965, 0.7148465705529157, -0.6308807679298589,
  -0.027983769416859854, 0.18703481171909309, 0.030841381835560764,
  -0.0328830116668852, -0.010597401785069032,
];
const DB4_REC_LO = [
  0.2303778133088965, 0.7148465705529157, 0.6308807679298589,
  -0.027983769416859854, -0.18703481171909309, 0.030841381835560764,
  0.0328830116668852, -0.010597401785069032,
];
const DB4_REC_HI = [
  -0.010597401785069032, -0.0328830116668852, 0.030841381835560764,
  0.18703481171909309, -0.027983769416859854, -0.6308807679298589,
  0.7148465705529157, -0.2303778133088965,
];

const CARRIERS: readonly [number, number][] = [
  [14, 14],
  [-14, -14],
  [126, 14],
  [-126, -14],
  [98, -14],
  [-98, 14],
  [128, 128],
  [-128, -128],
];

function symmExt(data: Float64Array, p: number): Float64Array {
  const n = data.length;
  const out = new Float64Array(n + 2 * p);
  for (let i = 0; i < p; i += 1) out[i] = data[p - 1 - i]!;
  for (let i = 0; i < n; i += 1) out[p + i] = data[i]!;
  for (let j = 0; j < p; j += 1) out[p + n + j] = data[n - 1 - j]!;
  return out;
}

function dwt1d(
  data: Float64Array,
  lo: readonly number[],
  hi: readonly number[],
): { a: Float64Array; d: Float64Array } {
  const f = lo.length;
  const n = data.length;
  const ext = symmExt(data, f - 1);
  const outLen = Math.floor((n + f - 1) / 2);
  const a = new Float64Array(outLen);
  const d = new Float64Array(outLen);
  for (let i = 0; i < outLen; i += 1) {
    let sa = 0;
    let sd = 0;
    const base = 2 * i + f;
    for (let k = 0; k < f; k += 1) {
      const idx = base - k;
      if (idx >= 0 && idx < ext.length) {
        sa += lo[k]! * ext[idx]!;
        sd += hi[k]! * ext[idx]!;
      }
    }
    a[i] = sa;
    d[i] = sd;
  }
  return { a, d };
}

function idwt1d(
  a: Float64Array,
  d: Float64Array,
  recLo: readonly number[],
  recHi: readonly number[],
  outLen: number,
): Float64Array {
  const f = recLo.length;
  const l = a.length;
  const up = 2 * l;
  const ua = new Float64Array(up);
  const ud = new Float64Array(up);
  for (let i = 0; i < l; i += 1) {
    ua[2 * i] = a[i]!;
    ud[2 * i] = d[i]!;
  }
  const convLen = up + f - 1;
  const rec = new Float64Array(convLen);
  for (let n = 0; n < convLen; n += 1) {
    let s = 0;
    const kmin = Math.max(0, n - (up - 1));
    const kmax = Math.min(n, f - 1);
    for (let k = kmin; k <= kmax; k += 1) {
      s += recLo[k]! * ua[n - k]! + recHi[k]! * ud[n - k]!;
    }
    rec[n] = s;
  }
  const start = f - 2;
  const out = new Float64Array(outLen);
  for (let i = 0; i < outLen; i += 1) out[i] = rec[start + i]!;
  return out;
}

function softThreshold(arr: Float64Array, t: number): Float64Array {
  const out = new Float64Array(arr.length);
  for (let i = 0; i < arr.length; i += 1) {
    const v = arr[i]!;
    const m = Math.abs(v) - t;
    out[i] = m > 0 ? (v < 0 ? -m : m) : 0;
  }
  return out;
}

function medianAbs(arr: Float64Array): number {
  const s = Float64Array.from(arr, (v) => Math.abs(v)).sort();
  const n = s.length;
  if (n === 0) return 0;
  return n % 2 === 1
    ? s[(n - 1) / 2]!
    : (s[n / 2 - 1]! + s[n / 2]!) / 2;
}

function rowsDwt(
  mat: Float64Array,
  w: number,
  h: number,
  lo: readonly number[],
  hi: readonly number[],
) {
  const w2 = Math.floor((w + lo.length - 1) / 2);
  const A = new Float64Array(h * w2);
  const D = new Float64Array(h * w2);
  const row = new Float64Array(w);
  for (let y = 0; y < h; y += 1) {
    for (let x = 0; x < w; x += 1) row[x] = mat[y * w + x]!;
    const { a, d } = dwt1d(row, lo, hi);
    A.set(a, y * w2);
    D.set(d, y * w2);
  }
  return { A, D, w2 };
}

function colsDwt(
  mat: Float64Array,
  w: number,
  h: number,
  lo: readonly number[],
  hi: readonly number[],
) {
  const h2 = Math.floor((h + lo.length - 1) / 2);
  const A = new Float64Array(h2 * w);
  const D = new Float64Array(h2 * w);
  const col = new Float64Array(h);
  for (let x = 0; x < w; x += 1) {
    for (let y = 0; y < h; y += 1) col[y] = mat[y * w + x]!;
    const { a, d } = dwt1d(col, lo, hi);
    for (let y = 0; y < h2; y += 1) {
      A[y * w + x] = a[y]!;
      D[y * w + x] = d[y]!;
    }
  }
  return { A, D, h2 };
}

function rowsIdwt(
  A: Float64Array,
  D: Float64Array,
  w2: number,
  h: number,
  lo: readonly number[],
  hi: readonly number[],
  wOut: number,
): Float64Array {
  const out = new Float64Array(h * wOut);
  const a = new Float64Array(w2);
  const d = new Float64Array(w2);
  for (let y = 0; y < h; y += 1) {
    for (let i = 0; i < w2; i += 1) {
      a[i] = A[y * w2 + i]!;
      d[i] = D[y * w2 + i]!;
    }
    out.set(idwt1d(a, d, lo, hi, wOut), y * wOut);
  }
  return out;
}

function colsIdwt(
  A: Float64Array,
  D: Float64Array,
  w: number,
  h2: number,
  lo: readonly number[],
  hi: readonly number[],
  hOut: number,
): Float64Array {
  const out = new Float64Array(hOut * w);
  const a = new Float64Array(h2);
  const d = new Float64Array(h2);
  for (let x = 0; x < w; x += 1) {
    for (let i = 0; i < h2; i += 1) {
      a[i] = A[i * w + x]!;
      d[i] = D[i * w + x]!;
    }
    const r = idwt1d(a, d, lo, hi, hOut);
    for (let y = 0; y < hOut; y += 1) out[y * w + x] = r[y]!;
  }
  return out;
}

function waveletDenoise(mat: Float64Array, size: number, level = 3): Float64Array {
  let cur = mat;
  let w = size;
  let h = size;
  const levels: Array<{
    LH: Float64Array;
    HL: Float64Array;
    HH: Float64Array;
    w: number;
    h: number;
    w2: number;
    h2: number;
  }> = [];
  for (let lv = 0; lv < level; lv += 1) {
    const { A: L, D: H, w2 } = rowsDwt(cur, w, h, DB4_DEC_LO, DB4_DEC_HI);
    const { A: LL, D: LH, h2 } = colsDwt(L, w2, h, DB4_DEC_LO, DB4_DEC_HI);
    const { A: HL, D: HH } = colsDwt(H, w2, h, DB4_DEC_LO, DB4_DEC_HI);
    levels.push({ LH, HL, HH, w, h, w2, h2 });
    cur = LL;
    w = w2;
    h = h2;
  }
  const finest = levels[0]!;
  const sigma = medianAbs(finest.LH) / 0.6745;
  const threshold = sigma * Math.sqrt(2 * Math.log(size * size));
  let ll = cur;
  for (let lv = level - 1; lv >= 0; lv -= 1) {
    const { LH, HL, HH, w: wO, h: hO, w2, h2 } = levels[lv]!;
    const L = colsIdwt(
      ll,
      softThreshold(LH, threshold),
      w2,
      h2,
      DB4_REC_LO,
      DB4_REC_HI,
      hO,
    );
    const H = colsIdwt(
      softThreshold(HL, threshold),
      softThreshold(HH, threshold),
      w2,
      h2,
      DB4_REC_LO,
      DB4_REC_HI,
      hO,
    );
    ll = rowsIdwt(L, H, w2, hO, DB4_REC_LO, DB4_REC_HI, wO);
  }
  return ll;
}

function fftRadix2(re: Float64Array, im: Float64Array): void {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i += 1) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      const tr = re[i]!;
      re[i] = re[j]!;
      re[j] = tr;
      const ti = im[i]!;
      im[i] = im[j]!;
      im[j] = ti;
    }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = (-2 * Math.PI) / len;
    const wpr = Math.cos(ang);
    const wpi = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let wr = 1;
      let wi = 0;
      for (let k = 0; k < len / 2; k += 1) {
        const a = i + k;
        const b = i + k + len / 2;
        const tr = re[b]! * wr - im[b]! * wi;
        const ti = re[b]! * wi + im[b]! * wr;
        re[b] = re[a]! - tr;
        im[b] = im[a]! - ti;
        re[a]! += tr;
        im[a]! += ti;
        const nwr = wr * wpr - wi * wpi;
        wi = wr * wpi + wi * wpr;
        wr = nwr;
      }
    }
  }
}

function fftLogMagnitude(gray: Float64Array, size: number): Float32Array {
  const re = new Float64Array(size * size);
  const im = new Float64Array(size * size);
  re.set(gray);
  const rr = new Float64Array(size);
  const ri = new Float64Array(size);
  for (let y = 0; y < size; y += 1) {
    for (let x = 0; x < size; x += 1) {
      rr[x] = re[y * size + x]!;
      ri[x] = im[y * size + x]!;
    }
    fftRadix2(rr, ri);
    for (let x = 0; x < size; x += 1) {
      re[y * size + x] = rr[x]!;
      im[y * size + x] = ri[x]!;
    }
  }
  const cr = new Float64Array(size);
  const ci = new Float64Array(size);
  for (let x = 0; x < size; x += 1) {
    for (let y = 0; y < size; y += 1) {
      cr[y] = re[y * size + x]!;
      ci[y] = im[y * size + x]!;
    }
    fftRadix2(cr, ci);
    for (let y = 0; y < size; y += 1) {
      re[y * size + x] = cr[y]!;
      im[y * size + x] = ci[y]!;
    }
  }
  const half = size / 2;
  const out = new Float32Array(size * size);
  let mn = Infinity;
  let mx = -Infinity;
  for (let y = 0; y < size; y += 1) {
    const sy = (y + half) % size;
    for (let x = 0; x < size; x += 1) {
      const sx = (x + half) % size;
      const idx = y * size + x;
      const v = Math.log1p(Math.hypot(re[idx]!, im[idx]!));
      out[sy * size + sx] = v;
      if (v < mn) mn = v;
      if (v > mx) mx = v;
    }
  }
  const range = mx - mn + 1e-8;
  for (let i = 0; i < out.length; i += 1) out[i] = (out[i]! - mn) / range;
  return out;
}

let carrierCache: Float32Array | undefined;
let carrierCacheSize = 0;

function carrierMask(size: number): Float32Array {
  if (carrierCache && carrierCacheSize === size) return carrierCache;
  const mask = new Float32Array(size * size);
  const c = size >> 1;
  for (const [fy, fx] of CARRIERS) {
    for (const [yy, xx] of [
      [c + fy, c + fx],
      [c - fy, c - fx],
    ] as const) {
      if (yy >= 0 && yy < size && xx >= 0 && xx < size) {
        mask[yy * size + xx] = 1;
      }
    }
  }
  carrierCache = mask;
  carrierCacheSize = size;
  return mask;
}

/** Resize ImageData to size×size (bilinear-ish via canvas when available). */
export async function resizeToSynthIdSquare(
  imageData: ImageData,
  size = SYNTHID_INPUT_SIZE,
): Promise<{ r: Uint8Array; g: Uint8Array; b: Uint8Array }> {
  const canvas = new OffscreenCanvas(size, size);
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  if (!ctx) throw new Error("OffscreenCanvas 2d unavailable for SynthID");
  const src = new OffscreenCanvas(imageData.width, imageData.height);
  const sctx = src.getContext("2d", { willReadFrequently: true });
  if (!sctx) throw new Error("OffscreenCanvas 2d unavailable for SynthID");
  sctx.putImageData(imageData, 0, 0);
  ctx.drawImage(src, 0, 0, size, size);
  const { data } = ctx.getImageData(0, 0, size, size);
  const n = size * size;
  const r = new Uint8Array(n);
  const g = new Uint8Array(n);
  const b = new Uint8Array(n);
  for (let i = 0, p = 0; i < n; i += 1, p += 4) {
    r[i] = data[p] ?? 0;
    g[i] = data[p + 1] ?? 0;
    b[i] = data[p + 2] ?? 0;
  }
  return { r, g, b };
}

/** Build NCHW Float32 (1×6×size×size) for OpenSynthID. */
export function buildSynthIdInput(
  rByte: Uint8Array,
  gByte: Uint8Array,
  bByte: Uint8Array,
  size = SYNTHID_INPUT_SIZE,
): Float32Array {
  const n = size * size;
  const out = new Float32Array(6 * n);
  const rF = new Float64Array(n);
  const gF = new Float64Array(n);
  const bF = new Float64Array(n);
  for (let i = 0; i < n; i += 1) {
    const r = rByte[i]! / 255;
    const g = gByte[i]! / 255;
    const b = bByte[i]! / 255;
    out[i] = r;
    out[n + i] = g;
    out[2 * n + i] = b;
    rF[i] = r;
    gF[i] = g;
    bF[i] = b;
  }
  const denR = waveletDenoise(rF, size);
  const denG = waveletDenoise(gF, size);
  const denB = waveletDenoise(bF, size);
  for (let i = 0; i < n; i += 1) {
    out[3 * n + i] =
      (rF[i]! - denR[i]! + (gF[i]! - denG[i]!) + (bF[i]! - denB[i]!)) / 3;
  }
  const gray = new Float64Array(n);
  for (let i = 0; i < n; i += 1) {
    const y = Math.round(
      0.299 * rByte[i]! + 0.587 * gByte[i]! + 0.114 * bByte[i]!,
    );
    gray[i] = Math.min(255, Math.max(0, y)) / 255;
  }
  out.set(fftLogMagnitude(gray, size), 4 * n);
  out.set(carrierMask(size), 5 * n);
  for (let i = 0; i < out.length; i += 1) {
    const v = out[i]!;
    if (Number.isNaN(v)) out[i] = 0;
    else if (v === Infinity) out[i] = 1;
    else if (v === -Infinity) out[i] = -1;
  }
  return out;
}
