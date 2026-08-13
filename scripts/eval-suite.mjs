#!/usr/bin/env node
/**
 * Full offline eval suite for local PC + CI.
 *
 * Host matrix (Node CPU + Zig EP prefs) and browser matrix
 * (unpacked extension in Playwright Chromium: WASM / WebGPU).
 *
 * Usage:
 *   npm ci
 *   npm run setup:models && npm run setup:ort && npm run build:zig
 *   npm run eval:suite
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
import { createRequire } from "node:module";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(__dirname, "..");
const require = createRequire(import.meta.url);
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
    throw new Error(`${cmd} ${args.join(" ")} exited ${res.status}`);
  }
}

function hasEsbuild() {
  try {
    require.resolve("esbuild/package.json");
    return true;
  } catch {
    return false;
  }
}

function ensureNpmDeps() {
  if (hasEsbuild()) return;
  console.log(
    "\nnode_modules incomplete (esbuild missing). Running npm ci once…",
  );
  run("npm", ["ci"]);
  if (!hasEsbuild()) {
    throw new Error(
      "esbuild still missing after npm ci. From the repo root run: npm ci",
    );
  }
}

const runHost = process.env.EVAL_SUITE_HOST !== "0";
const runBrowser = process.env.EVAL_SUITE_BROWSER !== "0";

ensureNpmDeps();

if (!existsSync(path.join(root, "models/ai-image-detect-distilled/model_fp16.onnx"))) {
  run("node", ["scripts/setup-models.mjs"]);
}

if (runHost) {
  const hostModes = process.env.EVAL_SUITE_HOST_MODES ?? "zig";
  if (
    hostModes.includes("zig") &&
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
  // `npm run build` already runs prepare:assets via prebuild — don't duplicate.
  run("npm", ["run", "build"]);
  if (!existsSync(path.join(root, "dist/models"))) {
    throw new Error("dist/models missing after build — run npm run setup:models");
  }
  if (!existsSync(path.join(root, "dist/eval.html"))) {
    throw new Error("dist/eval.html missing after build");
  }
  const pw = [
    "playwright",
    "test",
    "--project=eval",
    "--reporter=line",
  ];
  // Browser suite uses Chromium --headless=new (see tests/eval/eval-suite.spec.ts).
  // EVAL_SUITE_HEADED=1 opens a window; EVAL_SUITE_XVFB=1 still wraps for old CI images.
  const useXvfb = process.env.EVAL_SUITE_XVFB === "1";
  const env = {
    EVAL_SUITE_BROWSER_PROVIDERS:
      process.env.EVAL_SUITE_BROWSER_PROVIDERS ??
      (process.env.CI === "1" ? "wasm" : "wasm,webgpu"),
  };
  if (useXvfb) {
    run("xvfb-run", ["--auto-servernum", "npx", ...pw], env);
  } else {
    run("npx", pw, env);
  }
}

run(process.execPath, ["scripts/eval-suite-report.mjs"]);
console.log(`\nOpen ${path.join(outDir, "index.html")} for the summary table.`);
console.log(
  "Live extension page: chrome-extension://<id>/eval.html?corpus=http://127.0.0.1:<port>",
);
