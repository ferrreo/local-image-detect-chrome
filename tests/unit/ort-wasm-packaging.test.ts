import { describe, expect, it } from "vitest";
import { existsSync } from "node:fs";
import path from "node:path";

/**
 * ORT's default import.meta.url resolver looks next to offscreen.js. We also
 * ship under dist/ort/ for chrome.runtime.getURL("ort/..."). Both must exist
 * after `npm run build`.
 */
describe("ORT wasm packaging", () => {
  const root = path.resolve("dist");
  const names = [
    "ort-wasm-simd-threaded.jsep.wasm",
    "ort-wasm-simd-threaded.wasm",
    "ort-wasm-simd-threaded.asyncify.wasm",
  ];

  it("ships wasm under dist/ort/ and beside offscreen.js", () => {
    expect(existsSync(path.join(root, "offscreen.js"))).toBe(true);
    for (const name of names) {
      expect(existsSync(path.join(root, "ort", name)), `ort/${name}`).toBe(
        true,
      );
      expect(existsSync(path.join(root, name)), name).toBe(true);
    }
  });
});
