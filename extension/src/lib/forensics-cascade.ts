/**
 * Shared distilled → accurate-secondary cascade gate.
 * Lexica-class AI often scores near-zero on distilled; the old mid-band-only
 * gate never paid for Proofmark there. Accurate path now runs the secondary
 * unless distilled is already decisive alone.
 */
export type ForensicsCascadeInput = {
  distilled: number;
  spectral: number;
  laplacianVariance: number;
  chromaFlatness: number;
};

/** Skip accurate head only when distilled is already decisive on its own. */
export function needsForensicsCascade(args: ForensicsCascadeInput): boolean {
  const d = args.distilled;
  const sp = args.spectral;
  if (d >= 0.88 && sp <= 0.4) return false;
  return true;
}
