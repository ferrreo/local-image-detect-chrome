import { describe, expect, it } from "vitest";
import { detectAiImage } from "../../extension/src/lib/pipeline";
import { readFileSync } from "node:fs";
import path from "node:path";

const fixtures = path.resolve("tests/fixtures/images");

function load(name: string): ArrayBuffer {
  const buf = readFileSync(path.join(fixtures, name));
  return buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
}

describe("detectAiImage pipeline", () => {
  it("short-circuits AI fixture with embedded Midjourney marker", async () => {
    const result = await detectAiImage({
      imageId: "ai1",
      bytes: load("ai_smooth_1.png"),
      stubVisual: true,
    });
    expect(result.label.kind).toBe("ai");
    expect(result.confidence).toBeGreaterThanOrEqual(0.9);
    expect(result.tiers.some((t) => t.tier === "provenance")).toBe(true);
  });

  it("returns a confidence for stub visual path on real-like fixture", async () => {
    const result = await detectAiImage({
      imageId: "real1",
      bytes: load("real_noise_1.png"),
      stubVisual: true,
    });
    expect(result.confidence).toBeGreaterThanOrEqual(0);
    expect(result.confidence).toBeLessThanOrEqual(1);
    expect(result.backend.kind).toBe("stub");
    expect(result.label.kind).not.toBe("error");
  });
});
