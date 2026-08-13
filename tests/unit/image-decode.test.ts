import { describe, expect, it } from "vitest";
import {
  guessMimeType,
  imageDataToNchwFloat32,
} from "../../extension/src/lib/image-decode";

describe("guessMimeType", () => {
  it("detects JPEG/PNG/WebP signatures", () => {
    expect(guessMimeType(Uint8Array.from([0xff, 0xd8, 0xff, 0xe0]))).toBe(
      "image/jpeg",
    );
    expect(
      guessMimeType(Uint8Array.from([0x89, 0x50, 0x4e, 0x47, 0, 0, 0, 0])),
    ).toBe("image/png");
    expect(
      guessMimeType(
        Uint8Array.from([
          0x52, 0x49, 0x46, 0x46, 0, 0, 0, 0, 0x57, 0x45, 0x42, 0x50,
        ]),
      ),
    ).toBe("image/webp");
  });
});

describe("imageDataToNchwFloat32", () => {
  it("emits planar RGB normalized with mean/std", () => {
    const data = new Uint8ClampedArray([255, 0, 0, 255, 0, 255, 0, 255]);
    const imageData = {
      data,
      width: 2,
      height: 1,
      colorSpace: "srgb",
    } as ImageData;
    const tensor = imageDataToNchwFloat32(imageData, [0, 0, 0], [1, 1, 1]);
    expect(tensor.length).toBe(6);
    expect(tensor[0]).toBeCloseTo(1);
    expect(tensor[1]).toBeCloseTo(0);
    expect(tensor[2]).toBeCloseTo(0);
    expect(tensor[3]).toBeCloseTo(1);
  });
});
