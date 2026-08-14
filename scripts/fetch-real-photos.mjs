#!/usr/bin/env node
/**
 * Download openly usable real photographs for local eval balance.
 * Uses Lorem Picsum (photos from Unsplash) with stable IDs.
 * Incremental: skips IDs already present in real-index.json.
 */
import { createHash } from "node:crypto";
import {
  existsSync,
  mkdirSync,
  readFileSync,
  writeFileSync,
} from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(__dirname, "..");
const outRoot = path.join(root, "benchmark/openrouter/real");
const indexPath = path.join(root, "benchmark/openrouter/real-index.json");

// Stable Picsum photo IDs (Unsplash originals).
const REAL_PHOTOS = [
  { id: "picsum_10", picsumId: 10 },
  { id: "picsum_15", picsumId: 15 },
  { id: "picsum_28", picsumId: 28 },
  { id: "picsum_48", picsumId: 48 },
  { id: "picsum_64", picsumId: 64 },
  { id: "picsum_83", picsumId: 83 },
  { id: "picsum_96", picsumId: 96 },
  { id: "picsum_119", picsumId: 119 },
  { id: "picsum_129", picsumId: 129 },
  { id: "picsum_137", picsumId: 137 },
  { id: "picsum_152", picsumId: 152 },
  { id: "picsum_163", picsumId: 163 },
  { id: "picsum_176", picsumId: 176 },
  { id: "picsum_193", picsumId: 193 },
  { id: "picsum_201", picsumId: 201 },
  { id: "picsum_225", picsumId: 225 },
  { id: "picsum_237", picsumId: 237 },
  { id: "picsum_250", picsumId: 250 },
  { id: "picsum_274", picsumId: 274 },
  { id: "picsum_292", picsumId: 292 },
  { id: "picsum_314", picsumId: 314 },
  { id: "picsum_338", picsumId: 338 },
  { id: "picsum_349", picsumId: 349 },
  { id: "picsum_366", picsumId: 366 },
  { id: "picsum_399", picsumId: 399 },
  { id: "picsum_433", picsumId: 433 },
  { id: "picsum_453", picsumId: 453 },
  { id: "picsum_488", picsumId: 488 },
  { id: "picsum_525", picsumId: 525 },
  { id: "picsum_553", picsumId: 553 },
  { id: "picsum_582", picsumId: 582 },
  { id: "picsum_593", picsumId: 593 },
  { id: "picsum_628", picsumId: 628 },
  { id: "picsum_660", picsumId: 660 },
  { id: "picsum_718", picsumId: 718 },
  { id: "picsum_741", picsumId: 741 },
  { id: "picsum_783", picsumId: 783 },
  { id: "picsum_823", picsumId: 823 },
];

mkdirSync(outRoot, { recursive: true });
const existing = existsSync(indexPath)
  ? JSON.parse(readFileSync(indexPath, "utf8"))
  : { images: [] };
const have = new Set(existing.images.map((i) => i.id));
const images = [...existing.images];

for (const photo of REAL_PHOTOS) {
  if (have.has(photo.id)) {
    console.log(`skip ${photo.id}`);
    continue;
  }
  const url = `https://picsum.photos/id/${photo.picsumId}/768/768.jpg`;
  console.log(`fetch ${photo.id}`);
  const res = await fetch(url, {
    redirect: "follow",
    headers: {
      "User-Agent": "NeoPixelBenchmark/1.0",
      Accept: "image/jpeg,image/*;q=0.8,*/*;q=0.5",
    },
  });
  if (!res.ok) {
    console.warn(`  failed HTTP ${res.status}`);
    continue;
  }
  const buf = Buffer.from(await res.arrayBuffer());
  const name = `${photo.id}.jpg`;
  writeFileSync(path.join(outRoot, name), buf);
  const sha256 = createHash("sha256").update(buf).digest("hex");
  images.push({
    id: photo.id,
    file: `real/${name}`,
    label: "real",
    license: "Unsplash License (via Lorem Picsum)",
    sourceUrl: url,
    picsumId: photo.picsumId,
    sha256,
    bytes: buf.byteLength,
  });
  have.add(photo.id);
  console.log(`  ok ${name} (${buf.byteLength})`);
}

writeFileSync(
  indexPath,
  JSON.stringify({ updatedAt: new Date().toISOString(), images }, null, 2) +
    "\n",
);
console.log(`Wrote ${images.length} real photos → ${indexPath}`);
