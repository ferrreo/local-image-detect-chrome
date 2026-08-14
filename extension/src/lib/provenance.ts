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

/**
 * Prefer textual / EXIF / XMP / C2PA containers over raw compressed pixels.
 * Whole-file latin1 scans false-positive common generator names inside IDAT.
 */
function metadataScanText(bytes: Uint8Array): string {
  const parts: string[] = [];
  const isPng =
    bytes[0] === 0x89 &&
    bytes[1] === 0x50 &&
    bytes[2] === 0x4e &&
    bytes[3] === 0x47;
  const isJpeg = bytes[0] === 0xff && bytes[1] === 0xd8;

  if (isPng) {
    // PNG: tEXt / iTXt / zTXt / eXIf / JUMBF chunk payloads.
    for (let i = 8; i + 12 < bytes.byteLength; ) {
      const len =
        ((bytes[i] ?? 0) << 24) |
        ((bytes[i + 1] ?? 0) << 16) |
        ((bytes[i + 2] ?? 0) << 8) |
        (bytes[i + 3] ?? 0);
      const type = latin1Slice(bytes, i + 4, i + 8);
      const dataStart = i + 8;
      const dataEnd = Math.min(bytes.byteLength, dataStart + Math.max(0, len));
      if (
        type === "tEXt" ||
        type === "iTXt" ||
        type === "zTXt" ||
        type === "eXIf" ||
        type === "caBX" ||
        type === "vpAd"
      ) {
        parts.push(latin1Slice(bytes, dataStart, dataEnd));
      }
      const next = dataEnd + 4;
      if (next <= i) break;
      i = next;
      if (type === "IEND") break;
    }
    // Trailing bytes after IEND often hold C2PA / test markers (always include).
    parts.push(latin1Slice(bytes, Math.max(0, bytes.byteLength - 16_000), bytes.byteLength));
    return parts.join("\n");
  }

  if (isJpeg) {
    let i = 2;
    while (i + 4 < bytes.byteLength) {
      if (bytes[i] !== 0xff) break;
      const marker = bytes[i + 1] ?? 0;
      if (marker === 0xd9 || marker === 0xda) break;
      const size = ((bytes[i + 2] ?? 0) << 8) | (bytes[i + 3] ?? 0);
      if (size < 2) break;
      if ((marker >= 0xe0 && marker <= 0xef) || marker === 0xfe) {
        parts.push(latin1Slice(bytes, i + 4, i + 2 + size));
      }
      i += 2 + size;
    }
    parts.push(latin1Slice(bytes, Math.max(0, bytes.byteLength - 16_000), bytes.byteLength));
    return parts.join("\n");
  }

  // WebP / unknown: head+tail only (no compressed pixel body if we can help it).
  return provenanceScanText(bytes);
}

function firstMatch(
  text: string,
  patterns: readonly RegExp[],
): RegExp | undefined {
  for (const pattern of patterns) {
    // RegExp with /g/ retains lastIndex — always test from start.
    pattern.lastIndex = 0;
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
 * 2. Generator Software / XMP strings (metadata containers only)
 * 3. C2PA actions, SD parameter blocks, AIGC labels (metadata containers only)
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

  // Hard SynthID / C2PA tokens are distinctive — full head+tail scan is OK.
  const full = provenanceScanText(bytes);
  const synthid = firstMatch(full, SYNTHID_SOFT_BINDING_PATTERNS);
  if (synthid) {
    return {
      score: asAiConfidence(0.99),
      detail: `watermark-synthid-meta:${synthid.source}`,
      shortCircuit: true,
    };
  }

  // Soft generator names ("openai", "canva", …) only in metadata containers.
  // Scanning compressed IDAT as latin1 false-positives charts / screenshots
  // and short-circuits to AI 98% with backend still "none".
  const meta = metadataScanText(bytes);

  const software = firstMatch(meta, AI_SOFTWARE_PATTERNS);
  if (software) {
    return {
      score: asAiConfidence(0.98),
      detail: `watermark-meta:software:${software.source}`,
      shortCircuit: true,
    };
  }

  const params = firstMatch(meta, AI_PARAM_PATTERNS);
  if (params) {
    return {
      score: asAiConfidence(0.93),
      detail: `watermark-meta:params:${params.source}`,
      shortCircuit: true,
    };
  }

  // Weak hint: unusually large XMP/APP1 chunks often accompany generative tools.
  const xmpIndex = meta.indexOf("http://ns.adobe.com/xap/1.0/");
  if (xmpIndex >= 0 && meta.length - xmpIndex > 8_000) {
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
