import { asAiConfidence, type AiConfidence } from "../shared/types";
import {
  AI_PARAM_PATTERNS,
  AI_SOFTWARE_PATTERNS,
  SYNTHID_SOFT_BINDING_PATTERNS,
} from "./ai-watermarks";

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
 * First-pass watermark / provenance scan over raw image bytes.
 *
 * Priority:
 * 1. SynthID Soft Binding / `c2pa.watermarked.unbound` (Google standard used
 *    by Gemini/Imagen and OpenAI ChatGPT image / API)
 * 2. Generator Software / XMP strings
 * 3. C2PA actions, SD parameter blocks, AIGC labels
 *
 * Does not cryptographically verify C2PA. Invisible SynthID *pixels* need the
 * OpenSynthID surrogate (or Google’s proprietary detector) when metadata is
 * stripped.
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

  const synthid = firstMatch(text, SYNTHID_SOFT_BINDING_PATTERNS);
  if (synthid) {
    return {
      score: asAiConfidence(0.99),
      detail: `watermark-synthid-meta:${synthid.source}`,
      shortCircuit: true,
    };
  }

  const software = firstMatch(text, AI_SOFTWARE_PATTERNS);
  if (software) {
    return {
      score: asAiConfidence(0.98),
      detail: `watermark-meta:software:${software.source}`,
      shortCircuit: true,
    };
  }

  const params = firstMatch(text, AI_PARAM_PATTERNS);
  if (params) {
    return {
      score: asAiConfidence(0.93),
      detail: `watermark-meta:params:${params.source}`,
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
