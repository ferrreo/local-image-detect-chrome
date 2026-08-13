import sharp from "sharp";

type RgbaImage = {
  width: number;
  height: number;
  data: Uint8ClampedArray;
};

async function decodeAnyImage(buffer: ArrayBuffer): Promise<RgbaImage> {
  const bytes = Buffer.from(buffer);
  // Trim trailing latin1 provenance markers after IEND for synthetic fixtures.
  let end = bytes.length;
  if (bytes[0] === 0x89 && bytes[1] === 0x50 && bytes[2] === 0x4e && bytes[3] === 0x47) {
    for (let i = 0; i < bytes.length - 8; i += 1) {
      if (
        bytes[i] === 0x49 &&
        bytes[i + 1] === 0x45 &&
        bytes[i + 2] === 0x4e &&
        bytes[i + 3] === 0x44
      ) {
        end = i + 8;
        break;
      }
    }
  }

  const { data, info } = await sharp(bytes.subarray(0, end))
    .ensureAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true });

  return {
    width: info.width,
    height: info.height,
    data: new Uint8ClampedArray(data.buffer, data.byteOffset, data.byteLength),
  };
}

class FakeImageBitmap {
  width: number;
  height: number;
  #data: Uint8ClampedArray;
  constructor(img: RgbaImage) {
    this.width = img.width;
    this.height = img.height;
    this.#data = img.data;
  }
  close() {}
  get data() {
    return this.#data;
  }
}

class FakeOffscreenCanvas {
  width: number;
  height: number;
  #pixels: Uint8ClampedArray;
  constructor(width: number, height: number) {
    this.width = width;
    this.height = height;
    this.#pixels = new Uint8ClampedArray(width * height * 4);
  }
  getContext(type: string) {
    if (type !== "2d") return null;
    const canvas = this;
    return {
      drawImage(
        bitmap: FakeImageBitmap,
        sx: number,
        sy: number,
        sw: number,
        sh: number,
        dx: number,
        dy: number,
        dw: number,
        dh: number,
      ) {
        for (let y = 0; y < dh; y += 1) {
          for (let x = 0; x < dw; x += 1) {
            const srcX = Math.min(
              bitmap.width - 1,
              Math.floor(sx + ((x + 0.5) * sw) / dw),
            );
            const srcY = Math.min(
              bitmap.height - 1,
              Math.floor(sy + ((y + 0.5) * sh) / dh),
            );
            const si = (srcY * bitmap.width + srcX) * 4;
            const di = ((dy + y) * canvas.width + (dx + x)) * 4;
            canvas.#pixels[di] = bitmap.data[si] ?? 0;
            canvas.#pixels[di + 1] = bitmap.data[si + 1] ?? 0;
            canvas.#pixels[di + 2] = bitmap.data[si + 2] ?? 0;
            canvas.#pixels[di + 3] = 255;
          }
        }
      },
      getImageData(x: number, y: number, w: number, h: number) {
        if (x === 0 && y === 0 && w === canvas.width && h === canvas.height) {
          return {
            data: canvas.#pixels,
            width: w,
            height: h,
            colorSpace: "srgb",
          };
        }
        const data = new Uint8ClampedArray(w * h * 4);
        for (let row = 0; row < h; row += 1) {
          const src = ((y + row) * canvas.width + x) * 4;
          data.set(canvas.#pixels.subarray(src, src + w * 4), row * w * 4);
        }
        return { data, width: w, height: h, colorSpace: "srgb" };
      },
    };
  }
}

export function installCanvasPolyfills(): void {
  globalThis.createImageBitmap = (async (
    image: ImageBitmapSource,
  ): Promise<ImageBitmap> => {
    if (!(image instanceof Blob)) {
      throw new Error("Test polyfill only supports Blob sources");
    }
    const buffer = await image.arrayBuffer();
    const rgba = await decodeAnyImage(buffer);
    return new FakeImageBitmap(rgba) as unknown as ImageBitmap;
  }) as typeof createImageBitmap;

  globalThis.OffscreenCanvas =
    FakeOffscreenCanvas as unknown as typeof OffscreenCanvas;
}

installCanvasPolyfills();
