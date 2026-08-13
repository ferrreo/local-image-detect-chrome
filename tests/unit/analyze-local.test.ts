import { describe, expect, it } from "vitest";
import { analyzeLocalStub } from "../../extension/src/lib/analyze-local";
import { readFileSync } from "node:fs";
import path from "node:path";

const fixtures = path.resolve("tests/fixtures/images");

function load(name: string): ArrayBuffer {
  const buf = readFileSync(path.join(fixtures, name));
  return buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
}

describe("analyzeLocalStub", () => {
  it("flags provenance-marked AI fixture", async () => {
    const result = await analyzeLocalStub({
      imageId: "ai1",
      bytes: load("ai_smooth_1.png"),
    });
    expect(result.label.kind).toBe("ai");
    expect(result.confidence).toBeGreaterThanOrEqual(0.9);
  });

  it("returns stub backend for unmarked images", async () => {
    const result = await analyzeLocalStub({
      imageId: "real1",
      bytes: load("real_noise_1.png"),
    });
    expect(result.backend.kind).toBe("stub");
    expect(result.label.kind).not.toBe("error");
  });
});
