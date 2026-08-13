import { inflateSync } from "node:zlib";

type RgbaImage = {
  width: number;
  height: number;
  data: Uint8ClampedArray;
};

function paeth(a: number, b: number, c: number): number {
  const p = a + b - c;
  const pa = Math.abs(p - a);
  const pb = Math.abs(p - b);
  const pc = Math.abs(p - c);
  if (pa <= pb && pa <= pc) return a;
  if (pb <= pc) return b;
  return c;
}

function decodePng(buffer: ArrayBuffer): RgbaImage {
  const bytes = new Uint8Array(buffer);
  if (
    bytes[0] !== 0x89 ||
    bytes[1] !== 0x50 ||
    bytes[2] !== 0x4e ||
    bytes[3] !== 0x47
  ) {
    throw new Error("Not a PNG");
  }

  let offset = 8;
  let width = 0;
  let height = 0;
  const idat: Uint8Array[] = [];

  while (offset + 8 <= bytes.length) {
    const len =
      ((bytes[offset] ?? 0) << 24) |
      ((bytes[offset + 1] ?? 0) << 16) |
      ((bytes[offset + 2] ?? 0) << 8) |
      (bytes[offset + 3] ?? 0);
    const type = String.fromCharCode(
      bytes[offset + 4] ?? 0,
      bytes[offset + 5] ?? 0,
      bytes[offset + 6] ?? 0,
      bytes[offset + 7] ?? 0,
    );
    const data = bytes.subarray(offset + 8, offset + 8 + len);
    if (type === "IHDR") {
      width =
        ((data[0] ?? 0) << 24) |
        ((data[1] ?? 0) << 16) |
        ((data[2] ?? 0) << 8) |
        (data[3] ?? 0);
      height =
        ((data[4] ?? 0) << 24) |
        ((data[5] ?? 0) << 16) |
        ((data[6] ?? 0) << 8) |
        (data[7] ?? 0);
    } else if (type === "IDAT") {
      idat.push(data);
    } else if (type === "IEND") {
      break;
    }
    offset += 12 + len;
  }

  const compressed = Buffer.concat(idat.map((c) => Buffer.from(c)));
  const raw = inflateSync(compressed);
  const stride = 1 + width * 4;
  const out = new Uint8ClampedArray(width * height * 4);
  let prev = new Uint8Array(width * 4);

  for (let y = 0; y < height; y += 1) {
    const rowOffset = y * stride;
    const filter = raw[rowOffset] ?? 0;
    const row = raw.subarray(rowOffset + 1, rowOffset + stride);
    const cur = new Uint8Array(width * 4);
    for (let i = 0; i < row.length; i += 1) {
      const x = row[i] ?? 0;
      const a = i >= 4 ? (cur[i - 4] ?? 0) : 0;
      const b = prev[i] ?? 0;
      const c = i >= 4 ? (prev[i - 4] ?? 0) : 0;
      let val = x;
      if (filter === 1) val = (x + a) & 255;
      else if (filter === 2) val = (x + b) & 255;
      else if (filter === 3) val = (x + Math.floor((a + b) / 2)) & 255;
      else if (filter === 4) val = (x + paeth(a, b, c)) & 255;
      cur[i] = val;
    }
    out.set(cur, y * width * 4);
    prev = cur;
  }

  return { width, height, data: out };
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
        // Nearest-neighbor resize/crop for tests.
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
  if (typeof globalThis.createImageBitmap !== "function") {
    const createImageBitmapPolyfill = async (
      image: ImageBitmapSource,
    ): Promise<ImageBitmap> => {
      if (!(image instanceof Blob)) {
        throw new Error("Test polyfill only supports Blob sources");
      }
      const buffer = await image.arrayBuffer();
      // Fixtures may append latin1 markers after IEND; trim to PNG stream.
      const bytes = new Uint8Array(buffer);
      let end = bytes.length;
      const marker = [0x49, 0x45, 0x4e, 0x44];
      for (let i = 0; i < bytes.length - 8; i += 1) {
        if (
          bytes[i] === marker[0] &&
          bytes[i + 1] === marker[1] &&
          bytes[i + 2] === marker[2] &&
          bytes[i + 3] === marker[3]
        ) {
          end = i + 8;
          break;
        }
      }
      const png = decodePng(buffer.slice(0, end));
      return new FakeImageBitmap(png) as unknown as ImageBitmap;
    };
    globalThis.createImageBitmap =
      createImageBitmapPolyfill as typeof createImageBitmap;
  }

  if (typeof globalThis.OffscreenCanvas !== "function") {
    globalThis.OffscreenCanvas =
      FakeOffscreenCanvas as unknown as typeof OffscreenCanvas;
  }
}

installCanvasPolyfills();
