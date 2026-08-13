import { describe, expect, it } from "vitest";
import { stubVisualClassify } from "../../extension/src/lib/visual-stub";

function image(
  width: number,
  height: number,
  fill: (i: number) => [number, number, number],
): ImageData {
  const data = new Uint8ClampedArray(width * height * 4);
  for (let i = 0; i < width * height; i += 1) {
    const [r, g, b] = fill(i);
    const p = i * 4;
    data[p] = r;
    data[p + 1] = g;
    data[p + 2] = b;
    data[p + 3] = 255;
  }
  return { data, width, height, colorSpace: "srgb" } as ImageData;
}

describe("stubVisualClassify", () => {
  it("scores flat images higher than colorful noisy ones", () => {
    const flat = image(32, 32, () => [120, 122, 121]);
    const vivid = image(32, 32, (i) => [
      (i * 17) % 256,
      (i * 41) % 256,
      (i * 73) % 256,
    ]);
    expect(stubVisualClassify(flat).score).toBeGreaterThan(
      stubVisualClassify(vivid).score,
    );
    expect(stubVisualClassify(flat).backend.kind).toBe("stub");
  });
});
