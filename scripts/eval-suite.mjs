#!/usr/bin/env node
/**
 * Full offline eval suite for local PC + CI.
 *
 * Host matrix (Node CPU + Zig EP prefs) and browser matrix
 * (unpacked extension in Playwright Chromium: WASM / WebGPU).
 *
 * Usage:
 *   npm run setup:models && npm run setup:ort && npm run build:zig
 *   npm run build
 *   npm run eval:suite                 # full local suite
 *   EVAL_SUITE_LIMIT=8 npm run eval:suite:ci
 *
 * Env:
 *   EVAL_SUITE_HOST=0           skip host modes
 *   EVAL_SUITE_BROWSER=0        skip Playwright extension modes
 *   EVAL_SUITE_HOST_MODES=...   comma list
 *   EVAL_SUITE_BROWSER_PROVIDERS=wasm,webgpu
 *   EVAL_SUITE_LIMIT=N          balanced subset of corpus
 */
import { spawnSync } from "node:child_process";
import { existsSync, mkdirSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(__dirname, "..");
const outDir = path.join(root, "benchmark/eval-suite");
mkdirSync(outDir, { recursive: true });

function run(cmd, args, env = {}) {
  console.log(`\n$ ${cmd} ${args.join(" ")}`);
  const res = spawnSync(cmd, args, {
    cwd: root,
    stdio: "inherit",
    env: { ...process.env, ...env },
    shell: process.platform === "win32",
  });
  if (res.status !== 0) {
    throw new Error(`${cmd} exited ${res.status}`);
  }
}

const runHost = process.env.EVAL_SUITE_HOST !== "0";
const runBrowser = process.env.EVAL_SUITE_BROWSER !== "0";

if (!existsSync(path.join(root, "models/ai-image-detect-distilled/model_fp16.onnx"))) {
  run("node", ["scripts/setup-models.mjs"]);
}

if (runHost) {
  if (
    (process.env.EVAL_SUITE_HOST_MODES ?? "zig").includes("zig") &&
    !existsSync(path.join(root, "native/zig-infer/zig-out/bin/truepixel-infer"))
  ) {
    try {
      run("node", ["scripts/build-zig.mjs"]);
    } catch (error) {
      console.warn("Zig build failed; zig modes will skip.", error);
    }
  }
  run(process.execPath, ["scripts/eval-suite-host.mjs"]);
}

if (runBrowser) {
  run("node", ["scripts/generate-icons.mjs"]);
  run("node", ["scripts/generate-fixtures.mjs"]);
  run("npm", ["run", "build"]);
  if (!existsSync(path.join(root, "dist/models"))) {
    throw new Error("dist/models missing after build — setup:models first");
  }
  const pw = [
    "playwright",
    "test",
    "tests/eval/eval-suite.spec.ts",
    "--reporter=line",
  ];
  const useXvfb =
    process.env.EVAL_SUITE_XVFB === "1" ||
    (process.env.CI === "1" && process.platform === "linux");
  if (useXvfb) {
    run("xvfb-run", ["--auto-servernum", "npx", ...pw], {
      EVAL_SUITE_BROWSER_PROVIDERS:
        process.env.EVAL_SUITE_BROWSER_PROVIDERS ?? "wasm",
    });
  } else {
    run("npx", pw, {
      EVAL_SUITE_BROWSER_PROVIDERS:
        process.env.EVAL_SUITE_BROWSER_PROVIDERS ?? "wasm,webgpu",
    });
  }
}

run(process.execPath, ["scripts/eval-suite-report.mjs"]);
console.log(`\nOpen ${path.join(outDir, "index.html")} for the summary table.`);
console.log(
  "Live extension page: chrome-extension://<id>/eval.html?corpus=http://127.0.0.1:<port>",
);
