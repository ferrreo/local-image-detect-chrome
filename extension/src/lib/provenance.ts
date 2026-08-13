import { asAiConfidence, type AiConfidence } from "../shared/types";

const AI_SOFTWARE_PATTERNS: readonly RegExp[] = [
  /\bmidjourney\b/i,
  /\bstable[\s_-]?diffusion\b/i,
  /\bdall[\s.-]?e\b/i,
  /\bchatgpt\b/i,
  /\bopenai\b/i,
  /\badobe\s*firefly\b/i,
  /\bflux(\.1)?\b/i,
  /\bcomfyui\b/i,
  /\bautomatic1111\b/i,
  /\binvokeai\b/i,
  /\bnightcafe\b/i,
  /\bleonardo\.?ai\b/i,
  /\bideogram\b/i,
  /\bgrok\b/i,
  /\bgemini\b/i,
  /\bimagen\b/i,
  /\breve\b/i,
  /\bcivitai\b/i,
  /\brecraft\b/i,
  /\bai[\s_-]?generated\b/i,
  /\bgenerated\s+with\s+ai\b/i,
  /\btrainedalgorithmicmedia\b/i,
  /\bcompositedwithtrainedalgorithmicmedia\b/i,
];

const AI_PARAM_PATTERNS: readonly RegExp[] = [
  /\bsampler\s*[:=]/i,
  /\bcfg\s*scale\s*[:=]/i,
  /\bsteps\s*[:=]\s*\d+/i,
  /\bnegative\s*prompt\s*[:=]/i,
  /\btxt2img\b/i,
  /\bimg2img\b/i,
  /\bmodel\s*hash\s*[:=]/i,
  /\bclip\s*skip\s*[:=]/i,
  /digitalSourceType/i,
  /c2pa\.actions/i,
  /\bc2pa\b/i,
  // PNG tEXt/zTXt/iTXt key used by China AIGC labeling (and similar embeds).
  /ptEXtAIGC/i,
  /ContentProducer/i,
];

export type ProvenanceHit = {
  score: AiConfidence;
  detail: string;
  shortCircuit: boolean;
};

/** C2PA / JUMBF manifests are often appended near the end of WebP/JPEG files. */
const HEAD_SCAN_BYTES = 512_000;
const TAIL_SCAN_BYTES = 512_000;

function latin1Slice(bytes: Uint8Array, start: number, end: number): string {
  const lo = Math.max(0, start);
  const hi = Math.min(bytes.byteLength, end);
  let out = "";
  for (let i = lo; i < hi; i += 1) {
    out += String.fromCharCode(bytes[i] ?? 0);
  }
  return out;
}

function provenanceScanText(bytes: Uint8Array): string {
  if (bytes.byteLength <= HEAD_SCAN_BYTES + TAIL_SCAN_BYTES) {
    return latin1Slice(bytes, 0, bytes.byteLength);
  }
  const head = latin1Slice(bytes, 0, HEAD_SCAN_BYTES);
  const tail = latin1Slice(
    bytes,
    bytes.byteLength - TAIL_SCAN_BYTES,
    bytes.byteLength,
  );
  return `${head}\n${tail}`;
}

function firstMatch(
  text: string,
  patterns: readonly RegExp[],
): RegExp | undefined {
  for (const pattern of patterns) {
    if (pattern.test(text)) return pattern;
  }
  return undefined;
}

/**
 * High-precision provenance scan over raw image bytes.
 * Looks for EXIF/XMP/IPTC/C2PA strings and common generator fingerprints.
 * Does not parse full C2PA cryptographically; string presence is enough for a
 * strong local signal when generators embed declarations.
 */
export function analyzeProvenance(bytes: Uint8Array): ProvenanceHit {
  if (bytes.byteLength === 0) {
    return {
      score: asAiConfidence(0.5),
      detail: "empty-bytes",
      shortCircuit: false,
    };
  }

  const text = provenanceScanText(bytes);

  const software = firstMatch(text, AI_SOFTWARE_PATTERNS);
  if (software) {
    return {
      score: asAiConfidence(0.98),
      detail: `software:${software.source}`,
      shortCircuit: true,
    };
  }

  const params = firstMatch(text, AI_PARAM_PATTERNS);
  if (params) {
    return {
      score: asAiConfidence(0.93),
      detail: `params:${params.source}`,
      shortCircuit: true,
    };
  }

  // Weak hint: unusually large XMP/APP1 chunks often accompany generative tools.
  const xmpIndex = text.indexOf("http://ns.adobe.com/xap/1.0/");
  if (xmpIndex >= 0 && text.length - xmpIndex > 8_000) {
    return {
      score: asAiConfidence(0.62),
      detail: "large-xmp",
      shortCircuit: false,
    };
  }

  return {
    score: asAiConfidence(0.5),
    detail: "no-provenance",
    shortCircuit: false,
  };
}
