#!/usr/bin/env node
/**
 * Build a Proofmark-scale public training corpus under benchmark/distill-corpus/.
 *
 * Sources (images not committed):
 *   - Zitacron/real-vs-ai-corpus (CC BY-4.0) via datasets-server rows API
 *   - TheKernel01/Tiny-GenImage (CC BY-NC-SA-4.0) via parquet samples
 *
 * Lexica / hardcases stay in benchmark/openrouter and must remain holdout.
 *
 * Usage:
 *   node scripts/fetch-distill-corpus.mjs
 *   DISTILL_ZITACRON_PER_DOMAIN=400 DISTILL_TINY_GENIMAGE=800 node scripts/fetch-distill-corpus.mjs
 */
import { createHash } from "node:crypto";
import {
  createWriteStream,
  existsSync,
  mkdirSync,
  readFileSync,
  writeFileSync,
} from "node:fs";
import path from "node:path";
import { pipeline } from "node:stream/promises";
import { Readable } from "node:stream";
import { fileURLToPath } from "node:url";
import sharp from "sharp";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(__dirname, "..");
const outRoot = path.join(root, "benchmark/distill-corpus");

const ZITACRON_PER =
  Number(process.env.DISTILL_ZITACRON_PER_DOMAIN || "350") || 350;
const TINY_N = Number(process.env.DISTILL_TINY_GENIMAGE || "1000") || 1000;
const CONCURRENCY = Number(process.env.DISTILL_FETCH_CONCURRENCY || "6") || 6;

/** Stratified pulls from Zitacron/real-vs-ai-corpus (mirrors Proofmark domains). */
const ZITACRON_SPECS = [
  {
    key: "synthetic-characters",
    label: "ai",
    source: "AbstractPhil/synthetic-characters",
    start: 0,
    end: 140_000,
  },
  {
    key: "flux-reason",
    label: "ai",
    source: "LucasFang/FLUX-Reason-6M",
    start: 180_000,
    end: 6_000_000,
  },
  {
    key: "visual-logic",
    label: "real",
    source: "skylenage/DeepVision-103K",
    start: 150_000,
    end: 174_000,
  },
  {
    key: "laion-aesthetic",
    label: "real",
    source: "laion/laion2B-en-aesthetic",
    start: 6_050_000,
    end: 12_330_000,
  },
];

const exactHashes = new Set();
const manifest = [];

function sha256(buf) {
  return createHash("sha256").update(buf).digest("hex");
}

async function fetchWithRetry(url, attempts = 4) {
  let last;
  for (let i = 0; i < attempts; i += 1) {
    try {
      const res = await fetch(url, { redirect: "follow" });
      if (res.ok) return res;
      last = new Error(`HTTP ${res.status} ${url}`);
    } catch (err) {
      last = err;
    }
    await new Promise((r) => setTimeout(r, 500 * 2 ** i));
  }
  throw last;
}

async function fetchZitacronCandidates(spec) {
  const batchCount = 12;
  const batchLength = 100;
  const span = Math.max(1, spec.end - spec.start - batchLength);
  const starts = Array.from({ length: batchCount }, (_, index) =>
    Math.floor(spec.start + (span * index) / Math.max(1, batchCount - 1)),
  );
  const rows = [];
  for (const start of starts) {
    const params = new URLSearchParams({
      dataset: "Zitacron/real-vs-ai-corpus",
      config: "default",
      split: "train",
      offset: String(start),
      length: String(batchLength),
    });
    const res = await fetchWithRetry(
      `https://datasets-server.huggingface.co/rows?${params}`,
    );
    const payload = await res.json();
    for (const item of payload.rows || []) {
      const row = item.row;
      if (!row?.image?.src) continue;
      if (row.source_dataset !== spec.source) continue;
      const want = spec.label === "ai" ? 1 : 0;
      if (Number(row.label) !== want) continue;
      rows.push({ rowIndex: item.row_idx, src: row.image.src, row });
    }
  }
  return rows;
}

async function acceptImage(bytes, meta) {
  if (bytes.length < 1000 || bytes.length > 25_000_000) return null;
  const image = sharp(bytes);
  const info = await image.metadata();
  if (!info.width || !info.height) return null;
  if (Math.min(info.width, info.height) < 96) return null;
  const digest = sha256(bytes);
  if (exactHashes.has(digest)) return null;
  exactHashes.add(digest);
  return {
    bytes,
    digest,
    width: info.width,
    height: info.height,
    format: info.format || "jpeg",
    ...meta,
  };
}

async function writeAccepted(item, label, domain) {
  const ext =
    item.format === "png"
      ? "png"
      : item.format === "webp"
        ? "webp"
        : "jpg";
  const dir = path.join(outRoot, label, domain);
  mkdirSync(dir, { recursive: true });
  const file = `${item.rowIndex ?? item.digest.slice(0, 12)}.${ext}`;
  const rel = path.join(label, domain, file);
  const abs = path.join(outRoot, rel);
  if (!existsSync(abs)) writeFileSync(abs, item.bytes);
  const record = {
    file: rel.replaceAll("\\", "/"),
    label,
    domain,
    sha256: item.digest,
    width: item.width,
    height: item.height,
    source: item.source,
    license: item.license,
  };
  manifest.push(record);
  return record;
}

async function mapPool(items, limit, fn) {
  const out = [];
  let i = 0;
  async function worker() {
    while (i < items.length) {
      const idx = i;
      i += 1;
      out[idx] = await fn(items[idx], idx);
    }
  }
  await Promise.all(
    Array.from({ length: Math.min(limit, items.length) }, () => worker()),
  );
  return out;
}

async function fetchZitacron() {
  for (const spec of ZITACRON_SPECS) {
    console.log(`\nZitacron ${spec.key} (target ${ZITACRON_PER})`);
    const candidates = await fetchZitacronCandidates(spec);
    console.log(`  candidates ${candidates.length}`);
    let accepted = 0;
    for (let cursor = 0; cursor < candidates.length && accepted < ZITACRON_PER; ) {
      const batch = candidates.slice(cursor, cursor + CONCURRENCY);
      cursor += CONCURRENCY;
      const results = await mapPool(batch, CONCURRENCY, async (cand) => {
        try {
          const res = await fetchWithRetry(cand.src);
          const buf = Buffer.from(await res.arrayBuffer());
          return acceptImage(buf, {
            rowIndex: cand.rowIndex,
            source: spec.source,
            license: cand.row.source_license || "cc-by-4.0",
          });
        } catch {
          return null;
        }
      });
      for (const item of results) {
        if (!item || accepted >= ZITACRON_PER) continue;
        await writeAccepted(item, spec.label, `zitacron__${spec.key}`);
        accepted += 1;
      }
      process.stdout.write(`\r  accepted ${accepted}/${ZITACRON_PER}`);
    }
    process.stdout.write("\n");
  }
}

async function fetchTinyGenImage() {
  console.log(`\nTiny-GenImage sample (target ${TINY_N})`);
  // Use datasets-server rows on validation/train if available.
  const splits = ["validation", "train"];
  let accepted = 0;
  for (const split of splits) {
    if (accepted >= TINY_N) break;
    for (let offset = 0; offset < 50_000 && accepted < TINY_N; offset += 100) {
      const params = new URLSearchParams({
        dataset: "TheKernel01/Tiny-GenImage",
        config: "default",
        split,
        offset: String(offset),
        length: "100",
      });
      let payload;
      try {
        const res = await fetchWithRetry(
          `https://datasets-server.huggingface.co/rows?${params}`,
        );
        payload = await res.json();
      } catch (err) {
        console.warn(`  skip ${split}@${offset}: ${err}`);
        break;
      }
      const rows = payload.rows || [];
      if (!rows.length) break;
      const results = await mapPool(rows, CONCURRENCY, async (item) => {
        try {
          const row = item.row;
          const src = row?.image?.src;
          if (!src) return null;
          const labelNum = Number(row.label);
          // Tiny-GenImage: 0=real, 1=fake
          const label = labelNum === 1 ? "ai" : "real";
          const res = await fetchWithRetry(src);
          const buf = Buffer.from(await res.arrayBuffer());
          const domain =
            String(row.generator || row.source || row.model || "tiny-genimage")
              .toLowerCase()
              .replace(/[^a-z0-9._-]+/g, "-")
              .slice(0, 48) || "tiny-genimage";
          return acceptImage(buf, {
            rowIndex: item.row_idx,
            source: "TheKernel01/Tiny-GenImage",
            license: "cc-by-nc-sa-4.0",
            label,
            domain: `tiny__${domain}`,
          });
        } catch {
          return null;
        }
      });
      for (const item of results) {
        if (!item || accepted >= TINY_N) continue;
        await writeAccepted(item, item.label, item.domain);
        accepted += 1;
      }
      process.stdout.write(`\r  accepted ${accepted}/${TINY_N} (${split}@${offset})`);
      if (rows.length < 100) break;
    }
  }
  process.stdout.write("\n");
}

mkdirSync(outRoot, { recursive: true });
await fetchZitacron();
await fetchTinyGenImage();

const summary = {
  generatedAt: new Date().toISOString(),
  root: "benchmark/distill-corpus",
  counts: {
    total: manifest.length,
    ai: manifest.filter((m) => m.label === "ai").length,
    real: manifest.filter((m) => m.label === "real").length,
  },
  note: "Training images only. Keep benchmark/openrouter/ai/lexica__feed and hardcases as frozen holdout.",
  rows: manifest,
};
writeFileSync(
  path.join(outRoot, "manifest.json"),
  JSON.stringify(summary, null, 2) + "\n",
);
console.log(
  `\nDone: ${summary.counts.total} images (ai=${summary.counts.ai}, real=${summary.counts.real}) → ${outRoot}`,
);
