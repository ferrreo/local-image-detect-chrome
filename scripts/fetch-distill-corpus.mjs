#!/usr/bin/env node
/**
 * Build a ~50k public training corpus under benchmark/distill-corpus/.
 *
 * Sources (images not committed):
 *   - Zitacron/real-vs-ai-corpus (CC BY-4.0) via datasets-server rows API
 *   - TheKernel01/Tiny-GenImage (CC BY-NC-SA-4.0) via parquet samples
 *   - Lexica train split via npm run fetch:lexica (separate script)
 *
 * Lexica holdout + hardcases stay in benchmark/openrouter and must remain frozen.
 *
 * Usage:
 *   npm run fetch:distill-corpus
 *   DISTILL_TARGET_TOTAL=50000 npm run fetch:distill-corpus
 *   DISTILL_ZITACRON_PER_DOMAIN=8000 DISTILL_TINY_GENIMAGE=8000 npm run fetch:distill-corpus
 */
import { createHash } from "node:crypto";
import {
  existsSync,
  mkdirSync,
  readdirSync,
  readFileSync,
  renameSync,
  writeFileSync,
} from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import sharp from "sharp";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(__dirname, "..");
const outRoot = path.join(root, "benchmark/distill-corpus");
const manifestPath = path.join(outRoot, "manifest.json");

const TARGET_TOTAL =
  Number(process.env.DISTILL_TARGET_TOTAL || "50000") || 50000;
const ZITACRON_PER =
  Number(process.env.DISTILL_ZITACRON_PER_DOMAIN || "8000") || 8000;
const TINY_N = Number(process.env.DISTILL_TINY_GENIMAGE || "8000") || 8000;
const CONCURRENCY = Number(process.env.DISTILL_FETCH_CONCURRENCY || "10") || 10;
const ZITACRON_BATCHES =
  Number(process.env.DISTILL_ZITACRON_BATCHES || "120") || 120;

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

function loadExisting() {
  // Resume: adopt on-disk images + prior manifest rows.
  if (existsSync(manifestPath)) {
    try {
      const prev = JSON.parse(readFileSync(manifestPath, "utf8"));
      for (const row of prev.rows || []) {
        if (!row?.file || !row?.sha256) continue;
        const abs = path.join(outRoot, row.file);
        if (!existsSync(abs)) continue;
        exactHashes.add(row.sha256);
        manifest.push(row);
      }
    } catch (err) {
      console.warn(`manifest resume warn: ${err}`);
    }
  }
  for (const label of ["ai", "real"]) {
    const labelRoot = path.join(outRoot, label);
    if (!existsSync(labelRoot)) continue;
    for (const domain of readdirSync(labelRoot)) {
      const dir = path.join(labelRoot, domain);
      let names;
      try {
        names = readdirSync(dir);
      } catch {
        continue;
      }
      for (const name of names) {
        if (!/\.(jpe?g|png|webp)$/i.test(name)) continue;
        const rel = path.join(label, domain, name).replaceAll("\\", "/");
        if (manifest.some((m) => m.file === rel)) continue;
        const abs = path.join(outRoot, rel);
        try {
          const bytes = readFileSync(abs);
          const digest = sha256(bytes);
          if (exactHashes.has(digest)) continue;
          exactHashes.add(digest);
          manifest.push({
            file: rel,
            label,
            domain,
            sha256: digest,
            width: null,
            height: null,
            source: "resume-scan",
            license: "unknown",
          });
        } catch {
          // skip unreadable
        }
      }
    }
  }
  console.log(
    `resume: ${manifest.length} images already on disk ` +
      `(ai=${manifest.filter((m) => m.label === "ai").length}, ` +
      `real=${manifest.filter((m) => m.label === "real").length})`,
  );
}

async function fetchWithRetry(url, attempts = 4) {
  let last;
  for (let i = 0; i < attempts; i += 1) {
    try {
      const res = await fetch(url, { redirect: "follow" });
      if (res.ok) return res;
      last = new Error(`HTTP ${res.status} ${url}`);
      if (res.status === 429 || res.status >= 500) {
        await new Promise((r) => setTimeout(r, 800 * 2 ** i));
        continue;
      }
      break;
    } catch (err) {
      last = err;
      await new Promise((r) => setTimeout(r, 500 * 2 ** i));
    }
  }
  throw last;
}

async function fetchZitacronCandidates(spec) {
  const batchLength = 100;
  const span = Math.max(1, spec.end - spec.start - batchLength);
  const starts = Array.from({ length: ZITACRON_BATCHES }, (_, index) =>
    Math.floor(spec.start + (span * index) / Math.max(1, ZITACRON_BATCHES - 1)),
  );
  const rows = [];
  const seenIdx = new Set();
  let done = 0;
  // Keep listing concurrency modest — HF datasets-server 429s easily.
  await mapPool(starts, Math.min(4, CONCURRENCY), async (start) => {
    const params = new URLSearchParams({
      dataset: "Zitacron/real-vs-ai-corpus",
      config: "default",
      split: "train",
      offset: String(start),
      length: String(batchLength),
    });
    try {
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
        if (seenIdx.has(item.row_idx)) continue;
        seenIdx.add(item.row_idx);
        rows.push({ rowIndex: item.row_idx, src: row.image.src, row });
      }
    } catch (err) {
      console.warn(
        `\n  candidate offset ${start} failed: ${
          err instanceof Error ? err.message : err
        }`,
      );
    } finally {
      done += 1;
      if (done % 10 === 0 || done === starts.length) {
        process.stdout.write(
          `\r  listing offsets ${done}/${starts.length} (rows ${rows.length})`,
        );
      }
    }
  });
  process.stdout.write("\n");
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
  const rel = path.join(label, domain, file).replaceAll("\\", "/");
  const abs = path.join(outRoot, rel);
  if (!existsSync(abs)) writeFileSync(abs, item.bytes);
  const record = {
    file: rel,
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
    Array.from({ length: Math.min(limit, Math.max(1, items.length)) }, () =>
      worker(),
    ),
  );
  return out;
}

function countLabel(label) {
  return manifest.filter((m) => m.label === label).length;
}

function domainCount(domain) {
  return manifest.filter((m) => m.domain === domain).length;
}

function persistManifest() {
  const summary = {
    generatedAt: new Date().toISOString(),
    root: "benchmark/distill-corpus",
    targetTotal: TARGET_TOTAL,
    counts: {
      total: manifest.length,
      ai: countLabel("ai"),
      real: countLabel("real"),
    },
    note: "Training images only. Keep benchmark/openrouter/ai/lexica__holdout and hardcases as frozen holdout.",
    rows: manifest,
  };
  const tmp = `${manifestPath}.tmp-${process.pid}`;
  writeFileSync(tmp, JSON.stringify(summary) + "\n");
  renameSync(tmp, manifestPath);
  return summary;
}

async function fetchZitacron() {
  for (const spec of ZITACRON_SPECS) {
    if (manifest.length >= TARGET_TOTAL) break;
    const domain = `zitacron__${spec.key}`;
    const already = domainCount(domain);
    const target = ZITACRON_PER;
    if (already >= target) {
      console.log(`\nZitacron ${spec.key}: already ${already}/${target}, skip`);
      continue;
    }
    console.log(`\nZitacron ${spec.key} (have ${already}, target ${target})`);
    const candidates = await fetchZitacronCandidates(spec);
    console.log(`  candidates ${candidates.length}`);
    let accepted = already;
    // Prefer unseen row indexes.
    const haveIdx = new Set(
      manifest
        .filter((m) => m.domain === domain)
        .map((m) => path.basename(m.file, path.extname(m.file))),
    );
    const fresh = candidates.filter(
      (c) => !haveIdx.has(String(c.rowIndex)),
    );
    for (let cursor = 0; cursor < fresh.length && accepted < target; ) {
      if (manifest.length >= TARGET_TOTAL) break;
      const batch = fresh.slice(cursor, cursor + CONCURRENCY);
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
        if (!item || accepted >= target || manifest.length >= TARGET_TOTAL) {
          continue;
        }
        await writeAccepted(item, spec.label, domain);
        accepted += 1;
      }
      process.stdout.write(`\r  accepted ${accepted}/${target}`);
      if (accepted % 100 === 0) persistManifest();
    }
    process.stdout.write("\n");
    persistManifest();
  }
}

async function fetchTinyGenImage() {
  if (manifest.length >= TARGET_TOTAL) return;
  const already = manifest.filter((m) =>
    String(m.domain || "").startsWith("tiny__"),
  ).length;
  if (already >= TINY_N) {
    console.log(`\nTiny-GenImage: already ${already}/${TINY_N}, skip`);
    return;
  }
  console.log(`\nTiny-GenImage sample (have ${already}, target ${TINY_N})`);
  const splits = ["validation", "train"];
  let accepted = already;
  for (const split of splits) {
    if (accepted >= TINY_N || manifest.length >= TARGET_TOTAL) break;
    for (let offset = 0; offset < 200_000 && accepted < TINY_N; offset += 100) {
      if (manifest.length >= TARGET_TOTAL) break;
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
        if (!item || accepted >= TINY_N || manifest.length >= TARGET_TOTAL) {
          continue;
        }
        await writeAccepted(item, item.label, item.domain);
        accepted += 1;
      }
      process.stdout.write(
        `\r  accepted ${accepted}/${TINY_N} (${split}@${offset})`,
      );
      if (accepted % 100 === 0) persistManifest();
      if (rows.length < 100) break;
    }
  }
  process.stdout.write("\n");
  persistManifest();
}

mkdirSync(outRoot, { recursive: true });
loadExisting();
await fetchZitacron();
await fetchTinyGenImage();
const summary = persistManifest();
console.log(
  `\nDone: ${summary.counts.total} images (ai=${summary.counts.ai}, real=${summary.counts.real}) → ${outRoot}`,
);
console.log(
  `Target ${TARGET_TOTAL}. Lexica train/holdout: npm run fetch:lexica ` +
    `(LEXICA_TRAIN / LEXICA_HOLDOUT).`,
);
