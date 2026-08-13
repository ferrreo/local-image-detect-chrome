#!/usr/bin/env node
/**
 * Build / vendor onnxruntime-web from microsoft/onnxruntime#29599
 * (preferredMatmulAccumulatorPrecision for MatMulNBits on WebGPU).
 *
 * Produces:
 *   vendor/onnxruntime-web/   — drop-in package (dist + package.json)
 *   native/ort-web-pr29599/active.json
 *
 * First run is slow (emscripten jsep wasm). Re-runs reuse artifacts unless
 * TRUEPIXEL_FORCE_ORT_WEB_PR=1.
 *
 *   npm run setup:ort-web-pr29599
 */
import {
  existsSync,
  mkdirSync,
  writeFileSync,
  rmSync,
  cpSync,
  readFileSync,
  readdirSync,
  copyFileSync,
} from "node:fs";
import { execFileSync, spawnSync } from "node:child_process";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(__dirname, "..");
const srcRoot = path.join(root, "native/ort-web-pr29599-src");
const ortSrc = path.join(srcRoot, "onnxruntime");
const stage = path.join(root, "native/ort-web-pr29599");
const vendor = path.join(root, "vendor/onnxruntime-web");
const force = process.env.TRUEPIXEL_FORCE_ORT_WEB_PR === "1";
const prRef = process.env.TRUEPIXEL_ORT_PR_REF ?? "pull/29599/head";

const marker = path.join(stage, "active.json");

function run(cmd, args, opts = {}) {
  console.log(`$ ${cmd} ${args.join(" ")}`);
  execFileSync(cmd, args, { stdio: "inherit", ...opts });
}

function ensureNinja() {
  const which = spawnSync("ninja", ["--version"], { encoding: "utf8" });
  if (which.status === 0) return;
  run("pip3", ["install", "--user", "ninja"]);
}

/** ORT cmake requires node+npm from the same install (not /exec-daemon/node). */
function ensureNodeForOrt() {
  delete process.env.npm_config_prefix;
  const nvmNode = "/home/ubuntu/.nvm/versions/node/v22.22.2/bin";
  if (existsSync(path.join(nvmNode, "node")) && existsSync(path.join(nvmNode, "npm"))) {
    process.env.PATH = `${nvmNode}:${process.env.PATH ?? ""}`;
    process.env.NODE_EXECUTABLE = path.join(nvmNode, "node");
    process.env.NPM_CLI = path.join(nvmNode, "npm");
    return;
  }
  const which = spawnSync("bash", ["-lc", "type -P node"], { encoding: "utf8" });
  const nodePath = (which.stdout ?? "").trim();
  if (!nodePath) throw new Error("node not found for ORT cmake");
  process.env.NODE_EXECUTABLE = nodePath;
}

function ensureEmsdk(ort) {
  const emsdk = path.join(ort, "cmake/external/emsdk");
  if (!existsSync(path.join(emsdk, "emsdk"))) {
    throw new Error(`emsdk missing under ${emsdk}`);
  }
  run("./emsdk", ["install", "latest"], { cwd: emsdk });
  run("./emsdk", ["activate", "latest"], { cwd: emsdk });
  const envFile = path.join(emsdk, "emsdk_env.sh");
  const printed = execFileSync(
    "bash",
    ["-lc", `unset npm_config_prefix; source '${envFile}' && env`],
    { encoding: "utf8" },
  );
  const keepPath = process.env.PATH ?? "";
  for (const line of printed.split("\n")) {
    const i = line.indexOf("=");
    if (i <= 0) continue;
    const k = line.slice(0, i);
    const v = line.slice(i + 1);
    if (k === "PATH") {
      // Prepend emsdk, keep our node-first PATH.
      process.env.PATH = `${v}:${keepPath}`;
    } else if (
      k === "EMSDK" ||
      k === "EMSDK_NODE" ||
      k.startsWith("EM_") ||
      k.includes("EMSCRIPTEN")
    ) {
      process.env[k] = v;
    }
  }
  ensureNodeForOrt();
}

if (existsSync(path.join(vendor, "dist/ort.webgpu.bundle.min.mjs")) && !force) {
  console.log(`Vendored ort-web already present: ${vendor}`);
  console.log(`Marker: ${marker}`);
  process.exit(0);
}

mkdirSync(srcRoot, { recursive: true });
mkdirSync(stage, { recursive: true });
ensureNinja();
ensureNodeForOrt();

if (!existsSync(path.join(ortSrc, ".git"))) {
  run(
    "git",
    ["clone", "--filter=blob:none", "https://github.com/microsoft/onnxruntime.git", ortSrc],
    { cwd: srcRoot },
  );
}

run("git", ["fetch", "origin", prRef], { cwd: ortSrc });
run("git", ["checkout", "-f", "FETCH_HEAD"], { cwd: ortSrc });
run("git", ["submodule", "sync", "--recursive"], { cwd: ortSrc });
run(
  "git",
  ["submodule", "update", "--init", "--recursive", "--depth", "1"],
  { cwd: ortSrc },
);

const sha = execFileSync("git", ["rev-parse", "HEAD"], {
  cwd: ortSrc,
  encoding: "utf8",
}).trim();
console.log(`Building onnxruntime-web from ${sha} (${prRef})`);

ensureEmsdk(ortSrc);

const buildSh = path.join(ortSrc, "build.sh");
const common = [
  "--config",
  "Release",
  "--build_wasm",
  "--skip_tests",
  "--enable_wasm_simd",
  "--enable_wasm_threads",
  "--disable_wasm_exception_catching",
  "--disable_rtti",
  "--parallel",
  // cmake find_program(NODE) needs a node+npm co-located install.
  "--cmake_extra_defines",
  `NODE_EXECUTABLE=${process.env.NODE_EXECUTABLE}`,
];

// ort.webgpu.* loads asyncify.wasm (native WebGPU EP) — that is where
// preferredMatmulAccumulatorPrecision from #29599 lives. Plain wasm = CPU EP.
run("bash", [buildSh, ...common], { cwd: ortSrc });
run("bash", [buildSh, ...common, "--use_webgpu"], { cwd: ortSrc });

const wasmOut = path.join(ortSrc, "build/Linux/Release");
const needed = [
  "ort-wasm-simd-threaded.wasm",
  "ort-wasm-simd-threaded.mjs",
  "ort-wasm-simd-threaded.asyncify.wasm",
  "ort-wasm-simd-threaded.asyncify.mjs",
];
for (const f of needed) {
  const p = path.join(wasmOut, f);
  if (!existsSync(p)) {
    const found = execFileSync(
      "bash",
      ["-lc", `find '${ortSrc}/build' -name '${f}' | head -1`],
      { encoding: "utf8" },
    ).trim();
    if (!found) throw new Error(`Missing build artifact ${f}`);
    mkdirSync(wasmOut, { recursive: true });
    copyFileSync(found, path.join(wasmOut, f));
  }
}

run("npm", ["ci"], { cwd: path.join(ortSrc, "js") });
run("npm", ["ci"], { cwd: path.join(ortSrc, "js/common") });
run("npm", ["ci"], { cwd: path.join(ortSrc, "js/web") });

const webDist = path.join(ortSrc, "js/web/dist");
mkdirSync(webDist, { recursive: true });
for (const f of needed) {
  copyFileSync(path.join(wasmOut, f), path.join(webDist, f));
}

run("npm", ["run", "build"], { cwd: path.join(ortSrc, "js/web") });

rmSync(vendor, { recursive: true, force: true });
mkdirSync(path.dirname(vendor), { recursive: true });
// Minimal package: package.json + dist + types + lib (for resolution).
cpSync(path.join(ortSrc, "js/web/dist"), path.join(vendor, "dist"), {
  recursive: true,
});
for (const f of ["package.json", "types.d.ts", "README.md"]) {
  const src = path.join(ortSrc, "js/web", f);
  if (existsSync(src)) copyFileSync(src, path.join(vendor, f));
}
if (existsSync(path.join(ortSrc, "js/web/lib"))) {
  cpSync(path.join(ortSrc, "js/web/lib"), path.join(vendor, "lib"), {
    recursive: true,
  });
}

// Point package name stays onnxruntime-web; stamp provenance.
const pkg = JSON.parse(readFileSync(path.join(vendor, "package.json"), "utf8"));
pkg.name = "onnxruntime-web";
pkg.truepixel = {
  source: "microsoft/onnxruntime#29599",
  sha,
  preferredMatmulAccumulatorPrecision: true,
};
writeFileSync(path.join(vendor, "package.json"), JSON.stringify(pkg, null, 2) + "\n");

writeFileSync(
  marker,
  JSON.stringify(
    {
      pr: 29599,
      ref: prRef,
      sha,
      vendor,
      builtAt: new Date().toISOString(),
      artifacts: readdirSync(path.join(vendor, "dist")).filter((f) =>
        f.startsWith("ort-wasm"),
      ),
    },
    null,
    2,
  ) + "\n",
);

console.log(`Vendored onnxruntime-web → ${vendor}`);
console.log(`Wrote ${marker}`);
console.log("Next: npm install && npm run build");
