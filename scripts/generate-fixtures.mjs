/**
 * Generate deterministic PNG fixtures for unit/integration tests.
 * "ai-like" images are smooth gradients; "real-like" images are noisy.
 */
import { deflateSync } from "node:zlib";
import { mkdirSync, writeFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const outDir = path.resolve(__dirname, "../tests/fixtures/images");
mkdirSync(outDir, { recursive: true });

function crc32(buf) {
  let c = ~0;
  for (let i = 0; i < buf.length; i += 1) {
    c ^= buf[i];
    for (let k = 0; k < 8; k += 1) {
      c = c & 1 ? (0xedb88320 ^ (c >>> 1)) : c >>> 1;
    }
  }
  return ~c >>> 0;
}

function chunk(type, data) {
  const typeBuf = Buffer.from(type, "ascii");
  const len = Buffer.alloc(4);
  len.writeUInt32BE(data.length, 0);
  const crc = Buffer.alloc(4);
  crc.writeUInt32BE(crc32(Buffer.concat([typeBuf, data])), 0);
  return Buffer.concat([len, typeBuf, data, crc]);
}

function png(size, rgbaFn) {
  const raw = Buffer.alloc((size * 4 + 1) * size);
  for (let y = 0; y < size; y += 1) {
    const rowStart = y * (size * 4 + 1);
    raw[rowStart] = 0;
    for (let x = 0; x < size; x += 1) {
      const [r, g, b, a] = rgbaFn(x, y, size);
      const i = rowStart + 1 + x * 4;
      raw[i] = r;
      raw[i + 1] = g;
      raw[i + 2] = b;
      raw[i + 3] = a;
    }
  }
  const signature = Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]);
  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(size, 0);
  ihdr.writeUInt32BE(size, 4);
  ihdr[8] = 8;
  ihdr[9] = 6;
  return Buffer.concat([
    signature,
    chunk("IHDR", ihdr),
    chunk("IDAT", deflateSync(raw)),
    chunk("IEND", Buffer.alloc(0)),
  ]);
}

function mulberry32(seed) {
  let t = seed >>> 0;
  return () => {
    t += 0x6d2b79f5;
    let r = Math.imul(t ^ (t >>> 15), 1 | t);
    r ^= r + Math.imul(r ^ (r >>> 7), 61 | r);
    return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
  };
}

function aiLike(seed) {
  return (x, y, size) => {
    // Near-flat generative-looking fields with only a soft radial falloff.
    const u = (x + 0.5) / size - 0.5;
    const v = (y + 0.5) / size - 0.5;
    const falloff = Math.hypot(u, v);
    const r = 150 + seed * 8 - falloff * 20;
    const g = 120 + seed * 5 - falloff * 12;
    const b = 170 - seed * 3 - falloff * 16;
    return [
      Math.max(0, Math.min(255, r)),
      Math.max(0, Math.min(255, g)),
      Math.max(0, Math.min(255, b)),
      255,
    ];
  };
}

function realLike(seed) {
  const rand = mulberry32(seed);
  return (x, y, size) => {
    // Spatially noisy, high-frequency texture with independent channel jitter.
    const n = (rand() - 0.5) * 160;
    const n2 = (rand() - 0.5) * 160;
    const n3 = (rand() - 0.5) * 160;
    const speck = ((x * 13 + y * 29 + seed * 7) % 17) * 6;
    return [
      Math.max(0, Math.min(255, 110 + n + speck)),
      Math.max(0, Math.min(255, 90 + n2)),
      Math.max(0, Math.min(255, 70 + n3 - speck * 0.5)),
      255,
    ];
  };
}

const files = [
  ["ai_smooth_1.png", aiLike(1), "ai"],
  ["ai_smooth_2.png", aiLike(2), "ai"],
  ["ai_smooth_3.png", aiLike(3), "ai"],
  ["real_noise_1.png", realLike(11), "real"],
  ["real_noise_2.png", realLike(22), "real"],
  ["real_noise_3.png", realLike(33), "real"],
];

const FIXTURE_SIDE = 320;

const index = [];
for (const [name, fn, label] of files) {
  const buf = png(FIXTURE_SIDE, fn);
  // Embed a fake Midjourney marker into one AI fixture for provenance tests.
  if (name === "ai_smooth_1.png") {
    const marker = Buffer.from("Software\0Midjourney", "latin1");
    const combined = Buffer.concat([buf, marker]);
    writeFileSync(path.join(outDir, name), combined);
  } else {
    writeFileSync(path.join(outDir, name), buf);
  }
  index.push({ file: name, label });
  console.log(`Wrote ${name}`);
}

writeFileSync(
  path.join(outDir, "index.json"),
  JSON.stringify({ images: index }, null, 2),
);

// Simple page used by Playwright integration tests.
const pageDir = path.resolve(__dirname, "../tests/fixtures/pages");
mkdirSync(pageDir, { recursive: true });
writeFileSync(
  path.join(pageDir, "gallery.html"),
  `<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>NeoPixel fixture gallery</title>
  <style>
    body { font-family: Georgia, serif; margin: 24px; background: #f3efe6; color: #222; }
    h1 { margin: 0 0 16px; }
    .grid { display: grid; grid-template-columns: repeat(3, 280px); gap: 16px; }
    figure { margin: 0; }
    img { width: 256px; height: 256px; display: block; background: #ddd; }
    figcaption { margin-top: 6px; font-size: 12px; }
  </style>
</head>
<body>
  <h1>Fixture gallery</h1>
  <div class="grid">
    ${index
      .map(
        (item) => `<figure>
      <img src="../images/${item.file}" alt="${item.label}" width="256" height="256" />
      <figcaption data-label="${item.label}">${item.file}</figcaption>
    </figure>`,
      )
      .join("\n")}
  </div>
</body>
</html>
`,
);

console.log("Fixtures ready.");
