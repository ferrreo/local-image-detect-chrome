#!/usr/bin/env node
/**
 * Orchestrate a ~50k train corpus + frozen Lexica holdout.
 *
 *   npm run fetch:corpus50k
 *
 * Order:
 *   1. Zitacron + Tiny-GenImage → benchmark/distill-corpus (~35k)
 *   2. Lexica holdout (2k) + Lexica train (15k)
 *
 * Resume-safe; re-running only fills deficits.
 */
import { spawn } from "node:child_process";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(__dirname, "..");

function run(cmd, args, env = {}) {
  return new Promise((resolve, reject) => {
    console.log(`\n$ ${cmd} ${args.join(" ")}`);
    const child = spawn(cmd, args, {
      cwd: root,
      stdio: "inherit",
      env: { ...process.env, ...env },
    });
    child.on("exit", (code) => {
      if (code === 0) resolve();
      else reject(new Error(`${cmd} exited ${code}`));
    });
  });
}

const distillEnv = {
  DISTILL_TARGET_TOTAL: process.env.DISTILL_TARGET_TOTAL || "35000",
  DISTILL_ZITACRON_PER_DOMAIN:
    process.env.DISTILL_ZITACRON_PER_DOMAIN || "7000",
  DISTILL_TINY_GENIMAGE: process.env.DISTILL_TINY_GENIMAGE || "7000",
  DISTILL_ZITACRON_BATCHES: process.env.DISTILL_ZITACRON_BATCHES || "140",
  DISTILL_FETCH_CONCURRENCY: process.env.DISTILL_FETCH_CONCURRENCY || "10",
};

const lexicaEnv = {
  LEXICA_MODE: process.env.LEXICA_MODE || "both",
  LEXICA_HOLDOUT: process.env.LEXICA_HOLDOUT || "2000",
  LEXICA_TRAIN: process.env.LEXICA_TRAIN || "15000",
  LEXICA_MAX_PAGES: process.env.LEXICA_MAX_PAGES || "2500",
  LEXICA_CONCURRENCY: process.env.LEXICA_CONCURRENCY || "8",
};

console.log("Building ~50k distill corpus + Lexica holdout/train split");
console.log("distill env", distillEnv);
console.log("lexica env", lexicaEnv);

await run("node", ["scripts/fetch-distill-corpus.mjs"], distillEnv);
await run("node", ["scripts/fetch-lexica-images.mjs"], lexicaEnv);

console.log("\nCorpus50k fetch finished.");
