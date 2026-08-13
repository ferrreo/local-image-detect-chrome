#!/usr/bin/env node
/**
 * Pull AI images from the Lexica.art public feed into the local eval corpus.
 *
 *   npm run fetch:lexica
 *   LEXICA_LIMIT=40 LEXICA_PAGES=3 npm run fetch:lexica
 *
 * Uses https://lexica.art/api/infinite-prompts (same source as the site feed).
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
const outDir = path.join(root, "benchmark/openrouter/ai/lexica__feed");
const indexPath = path.join(root, "benchmark/openrouter/index.json");
const registryPath = path.join(root, "benchmark/openrouter/lexica-feed.json");

const LIMIT = Math.max(1, Number(process.env.LEXICA_LIMIT ?? "48"));
const PAGES = Math.max(1, Number(process.env.LEXICA_PAGES ?? "4"));
const UA = "TruePixelBenchmark/1.0 (+local eval corpus)";

mkdirSync(outDir, { recursive: true });

const index = existsSync(indexPath)
  ? JSON.parse(readFileSync(indexPath, "utf8"))
  : { images: [] };
const haveIds = new Set(
  index.images
    .filter((i) => String(i.file || "").includes("lexica__feed/"))
    .map((i) => i.id),
);
const haveFiles = new Set(index.images.map((i) => i.file));

async function fetchFeedPage(cursor) {
  const url = new URL("https://lexica.art/api/infinite-prompts");
  url.searchParams.set("cursor", String(cursor));
  url.searchParams.set("model", "search");
  url.searchParams.set("searchMode", "images");
  const res = await fetch(url, {
    headers: {
      Accept: "application/json",
      "User-Agent": UA,
      Referer: "https://lexica.art/",
    },
  });
  if (!res.ok) {
    throw new Error(`Lexica feed HTTP ${res.status}`);
  }
  return res.json();
}

async function downloadImage(id) {
  // full_jpg is a stable JPEG CDN path used by Lexica.
  const url = `https://image.lexica.art/full_jpg/${id}`;
  const res = await fetch(url, {
    headers: {
      "User-Agent": UA,
      Referer: "https://lexica.art/",
      Accept: "image/jpeg,image/*;q=0.8,*/*;q=0.5",
    },
    redirect: "follow",
  });
  if (!res.ok) {
    throw new Error(`image HTTP ${res.status}`);
  }
  const buf = Buffer.from(await res.arrayBuffer());
  if (buf.byteLength < 1024) {
    throw new Error(`image too small (${buf.byteLength})`);
  }
  // JPEG magic
  if (!(buf[0] === 0xff && buf[1] === 0xd8)) {
    throw new Error("not jpeg");
  }
  return { buf, url };
}

const promptById = new Map();
const collected = [];
let cursor = 1;

for (let page = 0; page < PAGES && collected.length < LIMIT; page += 1) {
  console.log(`feed page cursor=${cursor}`);
  const data = await fetchFeedPage(cursor);
  for (const p of data.prompts ?? []) {
    promptById.set(p.id, p);
  }
  for (const img of data.images ?? []) {
    if (collected.length >= LIMIT) break;
    if (!img?.id || haveIds.has(img.id)) continue;
    collected.push(img);
    haveIds.add(img.id);
  }
  cursor = data.nextCursor ?? cursor + 50;
  if (!data.images?.length) break;
}

console.log(`Downloading up to ${collected.length} Lexica images…`);
const added = [];
for (const img of collected) {
  const file = `ai/lexica__feed/${img.id}.jpg`;
  const abs = path.join(root, "benchmark/openrouter", file);
  if (haveFiles.has(file) && existsSync(abs)) {
    console.log(`skip existing ${img.id}`);
    continue;
  }
  try {
    const { buf, url } = await downloadImage(img.id);
    writeFileSync(abs, buf);
    const prompt = promptById.get(img.promptid);
    const entry = {
      id: img.id,
      file,
      label: "ai",
      model: img.model_mode || prompt?.model || "lexica-feed",
      promptId: img.promptid ?? null,
      prompt: prompt?.cleanedPrompt || prompt?.prompt || null,
      width: img.width,
      height: img.height,
      sourceUrl: url,
      galleryUrl: `https://lexica.art?q=${img.id}`,
      hardCase: true,
      tags: ["lexica", "feed", "stable-diffusion-family"],
      sha256: createHash("sha256").update(buf).digest("hex"),
      bytes: buf.byteLength,
    };
    index.images.push(entry);
    added.push(entry);
    console.log(`ok ${img.id} (${buf.byteLength} bytes) model=${entry.model}`);
  } catch (error) {
    const msg = error instanceof Error ? error.message : String(error);
    console.warn(`fail ${img.id}: ${msg}`);
  }
}

index.updatedAt = new Date().toISOString();
index.count = index.images.length;
writeFileSync(indexPath, JSON.stringify(index, null, 2) + "\n");
writeFileSync(
  registryPath,
  JSON.stringify(
    {
      updatedAt: index.updatedAt,
      added: added.length,
      totalLexica: index.images.filter((i) =>
        String(i.file || "").includes("lexica__feed/"),
      ).length,
      images: added.map((i) => ({
        id: i.id,
        file: i.file,
        model: i.model,
        sha256: i.sha256,
      })),
    },
    null,
    2,
  ) + "\n",
);

console.log(
  `Added ${added.length} Lexica feed images → ${outDir} (corpus AI+real now ${index.images.length} AI-index entries)`,
);
