import { describe, expect, it } from "vitest";
import { analyzeProvenance } from "../../extension/src/lib/provenance";

function bytesFrom(text: string): Uint8Array {
  return new TextEncoder().encode(text);
}

describe("analyzeProvenance", () => {
  it("returns neutral score for ordinary bytes", () => {
    const hit = analyzeProvenance(bytesFrom("JFIF ordinary photograph metadata"));
    expect(hit.shortCircuit).toBe(false);
    expect(hit.score).toBe(0.5);
  });

  it("short-circuits on Midjourney software tags", () => {
    const hit = analyzeProvenance(
      bytesFrom("Exif\0\0Software\0Midjourney v6 output"),
    );
    expect(hit.shortCircuit).toBe(true);
    expect(hit.score).toBeGreaterThanOrEqual(0.9);
    expect(hit.detail).toMatch(/midjourney/i);
  });

  it("short-circuits on SynthID Soft Binding / c2pa.watermarked.unbound", () => {
    const hit = analyzeProvenance(
      bytesFrom(
        'c2pa.actions {"action":"c2pa.watermarked.unbound","softwareAgent":"OpenAI"}',
      ),
    );
    expect(hit.shortCircuit).toBe(true);
    expect(hit.score).toBeGreaterThanOrEqual(0.95);
    expect(hit.detail).toMatch(/synthid/i);
  });

  it("short-circuits on explicit SynthID Soft Binding URI", () => {
    const hit = analyzeProvenance(
      bytesFrom(
        "jumb softBinding com.google.synthid ContentCredentials Soft Binding",
      ),
    );
    expect(hit.shortCircuit).toBe(true);
    expect(hit.detail).toMatch(/synthid|soft/i);
  });

  it("short-circuits on Stable Diffusion parameter blocks", () => {
    const hit = analyzeProvenance(
      bytesFrom("Steps: 30, Sampler: Euler a, CFG scale: 7, Negative prompt: blurry"),
    );
    expect(hit.shortCircuit).toBe(true);
    expect(hit.score).toBeGreaterThanOrEqual(0.9);
  });

  it("short-circuits on C2PA digitalSourceType trainedAlgorithmicMedia", () => {
    const hit = analyzeProvenance(
      bytesFrom(
        'c2pa.actions {"digitalSourceType":"trainedAlgorithmicMedia"}',
      ),
    );
    expect(hit.shortCircuit).toBe(true);
    expect(hit.score).toBeGreaterThanOrEqual(0.9);
  });

  it("scans the file tail where C2PA manifests are often appended", () => {
    const head = new Uint8Array(600_000);
    head.fill(0x41); // 'A'
    const tail = bytesFrom(
      'jumb c2pa {"digitalSourceType":"trainedAlgorithmicMedia","claim_generator":"recraft"}',
    );
    const bytes = new Uint8Array(head.byteLength + tail.byteLength);
    bytes.set(head, 0);
    bytes.set(tail, head.byteLength);
    const hit = analyzeProvenance(bytes);
    expect(hit.shortCircuit).toBe(true);
    expect(hit.score).toBeGreaterThanOrEqual(0.9);
  });

  it("short-circuits on PNG ptEXtAIGC labeling chunks", () => {
    const hit = analyzeProvenance(
      bytesFrom('ptEXtAIGC{"Label":"1","ContentProducer":"vendor"}'),
    );
    expect(hit.shortCircuit).toBe(true);
    expect(hit.detail).toMatch(/AIGC|ContentProducer/i);
  });

  it("handles empty input", () => {
    const hit = analyzeProvenance(new Uint8Array());
    expect(hit.detail).toBe("empty-bytes");
  });
});
