#!/usr/bin/env node
/**
 * Pull Lexica.art feed images with a frozen holdout + expandable train split.
 *
 * Lexica's infinite-prompts cursor is effectively unbounded at our scale, so we
 * can keep growing train without contaminating a sealed test set.
 *
 *   npm run fetch:lexica
 *   LEXICA_HOLDOUT=2000 LEXICA_TRAIN=15000 npm run fetch:lexica
 *   LEXICA_MODE=holdout LEXICA_HOLDOUT=500 npm run fetch:lexica
 *   LEXICA_MODE=train LEXICA_TRAIN=8000 npm run fetch:lexica
 *
 * Holdout → benchmark/openrouter/ai/lexica__holdout/ (+ index.json)
 * Train   → benchmark/distill-corpus/ai/lexica__train/
 *
 * IDs already in the holdout registry are never written to train.
 */
import { createHash } from "node:crypto";
import {
  existsSync,
  mkdirSync,
  readdirSync,
  readFileSync,
  writeFileSync,
} from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(__dirname, "..");
const openrouterRoot = path.join(root, "benchmark/openrouter");
const holdoutDir = path.join(openrouterRoot, "ai/lexica__holdout");
const legacyFeedDir = path.join(openrouterRoot, "ai/lexica__feed");
const trainDir = path.join(root, "benchmark/distill-corpus/ai/lexica__train");
const indexPath = path.join(openrouterRoot, "index.json");
const registryPath = path.join(openrouterRoot, "lexica-split.json");
const distillManifestPath = path.join(
  root,
  "benchmark/distill-corpus/manifest.json",
);

const MODE = String(process.env.LEXICA_MODE || "both").toLowerCase();
const HOLDOUT_TARGET = Math.max(
  0,
  Number(process.env.LEXICA_HOLDOUT ?? "2000") || 2000,
);
const TRAIN_TARGET = Math.max(
  0,
  Number(process.env.LEXICA_TRAIN ?? "15000") || 15000,
);
const MAX_PAGES = Math.max(
  1,
  Number(process.env.LEXICA_MAX_PAGES ?? "2000") || 2000,
);
const CONCURRENCY = Math.max(
  1,
  Number(process.env.LEXICA_CONCURRENCY ?? "8") || 8,
);
const PAGE_DELAY_MS = Math.max(
  0,
  Number(process.env.LEXICA_PAGE_DELAY_MS ?? "120") || 120,
);
const UA = "TruePixelBenchmark/1.0 (+local eval corpus)";

mkdirSync(holdoutDir, { recursive: true });
mkdirSync(trainDir, { recursive: true });

function loadJson(file, fallback) {
  if (!existsSync(file)) return fallback;
  return JSON.parse(readFileSync(file, "utf8"));
}

function sleep(ms) {
  return new Promise((r) => setTimeout(r, ms));
}

function listImages(dir) {
  if (!existsSync(dir)) return [];
  return readdirSync(dir).filter((n) => /\.(jpe?g|png|webp)$/i.test(n));
}

const index = loadJson(indexPath, { images: [] });
if (!Array.isArray(index.images)) index.images = [];

const registry = loadJson(registryPath, {
  version: 1,
  cursor: 1,
  holdoutIds: [],
  trainIds: [],
  failedIds: [],
});
const holdoutIds = new Set(registry.holdoutIds || []);
const trainIds = new Set(registry.trainIds || []);
const failedIds = new Set(registry.failedIds || []);
let cursor = Number(registry.cursor || 1) || 1;

// Seal every existing Lexica eval image as holdout (never train on these IDs).
for (const entry of index.images) {
  const file = String(entry.file || "");
  if (!file.includes("lexica__feed/") && !file.includes("lexica__holdout/")) {
    continue;
  }
  const id = entry.id || path.basename(file, path.extname(file));
  if (!id) continue;
  holdoutIds.add(id);
  entry.tags = Array.from(
    new Set([...(entry.tags || []), "lexica", "holdout", "frozen-test"]),
  );
  entry.hardCase = true;
  entry.split = "holdout";
}

for (const name of listImages(legacyFeedDir)) {
  holdoutIds.add(path.basename(name, path.extname(name)));
}
for (const name of listImages(holdoutDir)) {
  holdoutIds.add(path.basename(name, path.extname(name)));
}
for (const name of listImages(trainDir)) {
  const id = path.basename(name, path.extname(name));
  if (!holdoutIds.has(id)) trainIds.add(id);
}

async function fetchFeedPage(pageCursor) {
  const url = new URL("https://lexica.art/api/infinite-prompts");
  url.searchParams.set("cursor", String(pageCursor));
  url.searchParams.set("model", "search");
  url.searchParams.set("searchMode", "images");
  const res = await fetch(url, {
    headers: {
      Accept: "application/json",
      "User-Agent": UA,
      Referer: "https://lexica.art/",
    },
  });
  if (!res.ok) throw new Error(`Lexica feed HTTP ${res.status}`);
  return res.json();
}

async function downloadImage(id) {
  const url = `https://image.lexica.art/full_jpg/${id}`;
  const res = await fetch(url, {
    headers: {
      "User-Agent": UA,
      Referer: "https://lexica.art/",
      Accept: "image/jpeg,image/*;q=0.8,*/*;q=0.5",
    },
    redirect: "follow",
  });
  if (!res.ok) throw new Error(`image HTTP ${res.status}`);
  const buf = Buffer.from(await res.arrayBuffer());
  if (buf.byteLength < 1024) throw new Error(`image too small (${buf.byteLength})`);
  if (!(buf[0] === 0xff && buf[1] === 0xd8)) throw new Error("not jpeg");
  return { buf, url };
}

async function mapPool(items, limit, fn) {
  const out = new Array(items.length);
  let i = 0;
  async function worker() {
    while (i < items.length) {
      const idx = i;
      i += 1;
      out[idx] = await fn(items[idx], idx);
    }
  }
  await Promise.all(
    Array.from({ length: Math.min(limit, Math.max(1, items.length)) }, () =>
      worker(),
    ),
  );
  return out;
}

function persistRegistry() {
  registry.updatedAt = new Date().toISOString();
  registry.cursor = cursor;
  registry.holdoutIds = [...holdoutIds];
  registry.trainIds = [...trainIds];
  registry.failedIds = [...failedIds];
  registry.counts = {
    holdout: holdoutIds.size,
    train: trainIds.size,
    failed: failedIds.size,
  };
  writeFileSync(registryPath, JSON.stringify(registry, null, 2) + "\n");
}

function upsertOpenrouterEntry(entry) {
  const idx = index.images.findIndex((i) => i.id === entry.id);
  if (idx >= 0) index.images[idx] = { ...index.images[idx], ...entry };
  else index.images.push(entry);
}

function appendDistillManifest(records) {
  if (!records.length) return;
  const current = loadJson(distillManifestPath, {
    generatedAt: new Date().toISOString(),
    root: "benchmark/distill-corpus",
    counts: { total: 0, ai: 0, real: 0 },
    rows: [],
  });
  if (!Array.isArray(current.rows)) current.rows = [];
  const have = new Set(current.rows.map((r) => r.file));
  for (const row of records) {
    if (have.has(row.file)) continue;
    current.rows.push(row);
    have.add(row.file);
  }
  current.generatedAt = new Date().toISOString();
  current.counts = {
    total: current.rows.length,
    ai: current.rows.filter((r) => r.label === "ai").length,
    real: current.rows.filter((r) => r.label === "real").length,
  };
  current.note =
    "Training images. Lexica holdout lives under benchmark/openrouter/ai/lexica__holdout and must stay out of train.";
  mkdirSync(path.dirname(distillManifestPath), { recursive: true });
  writeFileSync(distillManifestPath, JSON.stringify(current, null, 2) + "\n");
}

const wantHoldout = MODE === "both" || MODE === "holdout";
const wantTrain = MODE === "both" || MODE === "train";
const pendingHoldout = [];
const pendingTrain = [];
const promptById = new Map();
let pages = 0;
let emptyPages = 0;

async function flushDownloads() {
  const holdoutBatch = pendingHoldout.splice(0, pendingHoldout.length);
  const trainBatch = pendingTrain.splice(0, pendingTrain.length);
  const distillRows = [];

  const holdoutResults = await mapPool(holdoutBatch, CONCURRENCY, async (img) => {
    try {
      const { buf, url } = await downloadImage(img.id);
      const file = `ai/lexica__holdout/${img.id}.jpg`;
      const abs = path.join(openrouterRoot, file);
      if (!existsSync(abs)) writeFileSync(abs, buf);
      const prompt = promptById.get(img.promptid);
      return {
        ok: true,
        entry: {
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
          tags: ["lexica", "holdout", "frozen-test", "stable-diffusion-family"],
          sha256: createHash("sha256").update(buf).digest("hex"),
          bytes: buf.byteLength,
          split: "holdout",
        },
      };
    } catch (error) {
      return {
        ok: false,
        id: img.id,
        error: error instanceof Error ? error.message : String(error),
      };
    }
  });

  let holdoutAdded = 0;
  for (const result of holdoutResults) {
    if (!result?.ok) {
      if (result?.id) failedIds.add(result.id);
      if (result?.error) {
        console.warn(`holdout fail ${result.id}: ${result.error}`);
      }
      continue;
    }
    upsertOpenrouterEntry(result.entry);
    holdoutIds.add(result.entry.id);
    holdoutAdded += 1;
  }

  const trainResults = await mapPool(trainBatch, CONCURRENCY, async (img) => {
    if (holdoutIds.has(img.id)) {
      return { ok: false, id: img.id, error: "holdout" };
    }
    try {
      const { buf, url } = await downloadImage(img.id);
      const abs = path.join(trainDir, `${img.id}.jpg`);
      if (!existsSync(abs)) writeFileSync(abs, buf);
      const digest = createHash("sha256").update(buf).digest("hex");
      return {
        ok: true,
        id: img.id,
        row: {
          file: `ai/lexica__train/${img.id}.jpg`,
          label: "ai",
          domain: "lexica__train",
          sha256: digest,
          width: img.width ?? null,
          height: img.height ?? null,
          source: "lexica.art/api/infinite-prompts",
          license: "scraped-eval-train",
          sourceUrl: url,
        },
      };
    } catch (error) {
      return {
        ok: false,
        id: img.id,
        error: error instanceof Error ? error.message : String(error),
      };
    }
  });

  let trainAdded = 0;
  for (const result of trainResults) {
    if (!result?.ok) {
      if (result?.id && result.error !== "holdout") failedIds.add(result.id);
      if (result?.error && result.error !== "holdout") {
        console.warn(`train fail ${result.id}: ${result.error}`);
      }
      continue;
    }
    trainIds.add(result.id);
    distillRows.push(result.row);
    trainAdded += 1;
  }

  if (holdoutAdded || trainAdded) {
    console.log(
      `flushed +${holdoutAdded} holdout (now ${holdoutIds.size})  ` +
        `+${trainAdded} train (now ${trainIds.size})`,
    );
  }
  appendDistillManifest(distillRows);
}

console.log(
  `Lexica split: mode=${MODE} holdout=${holdoutIds.size}/${HOLDOUT_TARGET} ` +
    `train=${trainIds.size}/${TRAIN_TARGET} cursor=${cursor}`,
);

while (pages < MAX_PAGES) {
  const needHoldout =
    wantHoldout && holdoutIds.size + pendingHoldout.length < HOLDOUT_TARGET;
  const needTrain =
    wantTrain && trainIds.size + pendingTrain.length < TRAIN_TARGET;
  if (!needHoldout && !needTrain) break;

  console.log(
    `feed page=${pages + 1} cursor=${cursor} ` +
      `queue holdout=${pendingHoldout.length} train=${pendingTrain.length}`,
  );
  let data;
  try {
    data = await fetchFeedPage(cursor);
  } catch (err) {
    console.warn(`feed error: ${err instanceof Error ? err.message : err}`);
    await sleep(1000 * Math.min(8, pages + 1));
    pages += 1;
    continue;
  }

  for (const p of data.prompts ?? []) promptById.set(p.id, p);
  const images = data.images ?? [];
  if (!images.length) {
    emptyPages += 1;
    if (emptyPages >= 3) break;
  } else {
    emptyPages = 0;
  }

  const pendingIds = new Set([
    ...pendingHoldout.map((x) => x.id),
    ...pendingTrain.map((x) => x.id),
  ]);

  for (const img of images) {
    if (!img?.id) continue;
    if (
      holdoutIds.has(img.id) ||
      trainIds.has(img.id) ||
      failedIds.has(img.id) ||
      pendingIds.has(img.id)
    ) {
      continue;
    }
    if (needHoldout && holdoutIds.size + pendingHoldout.length < HOLDOUT_TARGET) {
      pendingHoldout.push(img);
      pendingIds.add(img.id);
    } else if (
      needTrain &&
      trainIds.size + pendingTrain.length < TRAIN_TARGET
    ) {
      pendingTrain.push(img);
      pendingIds.add(img.id);
    }
  }

  cursor = data.nextCursor ?? cursor + 50;
  pages += 1;
  if (PAGE_DELAY_MS) await sleep(PAGE_DELAY_MS);

  if (
    pendingHoldout.length >= 40 ||
    pendingTrain.length >= 40 ||
    pages % 10 === 0
  ) {
    await flushDownloads();
    persistRegistry();
  }
}

await flushDownloads();
index.updatedAt = new Date().toISOString();
index.count = index.images.length;
writeFileSync(indexPath, JSON.stringify(index, null, 2) + "\n");
persistRegistry();

console.log(
  `Lexica done: holdout=${holdoutIds.size}/${HOLDOUT_TARGET} ` +
    `train=${trainIds.size}/${TRAIN_TARGET} cursor=${cursor} → ${registryPath}`,
);
