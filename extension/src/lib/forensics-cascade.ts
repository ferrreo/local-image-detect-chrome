/**
 * Shared distilled → Community Forensics cascade gate.
 * Used by the Zig host, browser ORT-web, and Zig+ORT WASM paths so they
 * only pay for the second head when fusion can actually use it.
 */
export type ForensicsCascadeInput = {
  distilled: number;
  spectral: number;
  laplacianVariance: number;
  chromaFlatness: number;
};

/** Mirrors fusion.ts forensics gates. */
export function needsForensicsCascade(args: ForensicsCascadeInput): boolean {
  const d = args.distilled;
  const sp = args.spectral;
  // Only skip CF when distilled is already decisive on its own.
  if (d >= 0.88 && sp <= 0.4) return false;
  // Mid / elevated: ask CF (dual-mild-hold + strong CF paths).
  if (d >= 0.48 && d < 0.88) return true;
  const canCfBand =
    d >= 0.3 &&
    args.laplacianVariance >= 580 &&
    args.chromaFlatness >= 0.34 &&
    args.chromaFlatness <= 0.7;
  const canFlat =
    d >= 0.62 &&
    args.chromaFlatness >= 0.74 &&
    args.laplacianVariance >= 700;
  const canHighFlat =
    d >= 0.4 &&
    args.laplacianVariance >= 800 &&
    args.chromaFlatness >= 0.6 &&
    args.chromaFlatness <= 0.72 &&
    sp <= 0.38;
  return canCfBand || canFlat || canHighFlat;
}
