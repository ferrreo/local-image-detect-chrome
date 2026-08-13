import { existsSync, readFileSync } from "node:fs";
import path from "node:path";

export function loadOpenRouterCorpus(root) {
  const base = path.join(root, "benchmark/openrouter");
  const aiPath = path.join(base, "index.json");
  const realPath = path.join(base, "real-index.json");
  if (!existsSync(aiPath) || !existsSync(realPath)) {
    throw new Error("benchmark/openrouter missing; fetch corpus first");
  }
  const ai = JSON.parse(readFileSync(aiPath, "utf8"));
  const real = JSON.parse(readFileSync(realPath, "utf8"));
  let images = [
    ...ai.images.map((i) => ({
      file: i.file,
      label: "ai",
      model: i.model ?? null,
      abs: path.join(base, i.file),
    })),
    ...real.images.map((i) => ({
      file: i.file,
      label: "real",
      model: null,
      abs: path.join(base, i.file),
    })),
  ];
  const limit = Number(process.env.EVAL_SUITE_LIMIT ?? "0");
  if (Number.isFinite(limit) && limit > 0) {
    // Keep balance: half AI / half real when possible.
    const aiImgs = images.filter((i) => i.label === "ai");
    const realImgs = images.filter((i) => i.label === "real");
    const each = Math.max(1, Math.floor(limit / 2));
    images = [...aiImgs.slice(0, each), ...realImgs.slice(0, each)].slice(
      0,
      limit,
    );
  }
  return { base, images };
}

export function balancedAccuracy(c) {
  const tpr = c.tp + c.fn === 0 ? 0 : c.tp / (c.tp + c.fn);
  const tnr = c.tn + c.fp === 0 ? 0 : c.tn / (c.tn + c.fp);
  return (tpr + tnr) / 2;
}

export function scoreRows(rows, threshold = 0.65) {
  let tp = 0;
  let tn = 0;
  let fp = 0;
  let fn = 0;
  let sum = 0;
  for (const row of rows) {
    sum += row.totalMs;
    const predAi = row.confidence >= threshold;
    const actualAi = row.label === "ai";
    if (actualAi && predAi) tp += 1;
    else if (!actualAi && !predAi) tn += 1;
    else if (!actualAi && predAi) fp += 1;
    else fn += 1;
  }
  const confusion = { tp, tn, fp, fn };
  return {
    balancedAccuracy: balancedAccuracy(confusion),
    confusion,
    timing: {
      avgTotalMs: rows.length ? sum / rows.length : 0,
      count: rows.length,
    },
  };
}
