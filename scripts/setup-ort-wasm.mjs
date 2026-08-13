#!/usr/bin/env node
/**
 * Build / stage ONNX Runtime WebAssembly static library for Zig linking.
 *
 * Produces:
 *   native/ort-wasm/lib/libonnxruntime_webassembly.a
 *   native/ort-wasm/include/onnxruntime_c_api.h (+ cxx headers)
 *   native/ort-wasm/active.json
 *
 * First run clones ORT + builds with emscripten (slow). Re-runs reuse the .a.
 *
 *   npm run setup:ort-wasm
 *   TRUEPIXEL_FORCE_ORT_WASM=1 npm run setup:ort-wasm   # rebuild
 */
import {
  existsSync,
  mkdirSync,
  copyFileSync,
  writeFileSync,
  rmSync,
  cpSync,
  readFileSync,
} from "node:fs";
import { execFileSync, spawnSync } from "node:child_process";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(__dirname, "..");
const version = process.env.TRUEPIXEL_ORT_WASM_VERSION ?? "1.22.0";
const srcRoot = path.join(root, "native/ort-wasm-src");
const ortSrc = path.join(srcRoot, "onnxruntime");
const stage = path.join(root, "native/ort-wasm");
const libOut = path.join(stage, "lib", "libonnxruntime_webassembly.a");
const force = process.env.TRUEPIXEL_FORCE_ORT_WASM === "1";

function run(cmd, args, opts = {}) {
  console.log(`$ ${cmd} ${args.join(" ")}`);
  execFileSync(cmd, args, { stdio: "inherit", ...opts });
}

function ensureNinja() {
  const which = spawnSync("ninja", ["--version"], { encoding: "utf8" });
  if (which.status === 0) return;
  run("pip3", ["install", "--user", "ninja"]);
}

if (existsSync(libOut) && !force) {
  console.log(`ORT wasm static lib already present: ${libOut}`);
  process.exit(0);
}

mkdirSync(srcRoot, { recursive: true });
ensureNinja();

if (!existsSync(path.join(ortSrc, ".git"))) {
  run(
    "git",
    [
      "clone",
      "--depth",
      "1",
      "--branch",
      `v${version}`,
      "https://github.com/microsoft/onnxruntime.git",
      ortSrc,
    ],
    { cwd: srcRoot },
  );
}

run("git", ["submodule", "sync", "--recursive"], { cwd: ortSrc });
run(
  "git",
  ["submodule", "update", "--init", "--recursive", "--depth", "1"],
  { cwd: ortSrc },
);

// GitLab regenerates Eigen archives; pinned SHA1 in deps.txt can go stale.
const depsTxt = path.join(ortSrc, "cmake/deps.txt");
if (existsSync(depsTxt)) {
  let deps = readFileSync(depsTxt, "utf8");
  const stale =
    "eigen;https://gitlab.com/libeigen/eigen/-/archive/1d8b82b0740839c0de7f1242a3585e3390ff5f33/eigen-1d8b82b0740839c0de7f1242a3585e3390ff5f33.zip;5ea4d05e62d7f954a46b3213f9b2535bdd866803";
  const fixed =
    "eigen;https://gitlab.com/libeigen/eigen/-/archive/1d8b82b0740839c0de7f1242a3585e3390ff5f33/eigen-1d8b82b0740839c0de7f1242a3585e3390ff5f33.zip;51982be81bbe52572b54180454df11a3ece9a934";
  if (deps.includes(stale)) {
    writeFileSync(depsTxt, deps.replace(stale, fixed));
    console.log("Patched cmake/deps.txt Eigen SHA1 for current GitLab archive");
  }
}

const emsdk = path.join(ortSrc, "cmake/external/emsdk");
if (!existsSync(path.join(emsdk, "emsdk"))) {
  throw new Error(`emsdk missing at ${emsdk}`);
}
run("./emsdk", ["install", "latest"], { cwd: emsdk });
run("./emsdk", ["activate", "latest"], { cwd: emsdk });

const envFile = path.join(emsdk, "emsdk_env.sh");
// ORT ≤1.22 single-thread WASM needs MLFloat16 helpers (upstream d4076dc).
const mlasi = path.join(ortSrc, "onnxruntime/core/mlas/lib/mlasi.h");
if (existsSync(mlasi)) {
  let src = readFileSync(mlasi, "utf8");
  if (!src.includes("constexpr static MLFloat16 FromBits")) {
    src = src.replace(
      `explicit MLFloat16(float ff) : val(MLAS_Float2Half(ff)) {}

    float ToFloat() const { return MLAS_Half2Float(val); }`,
      `explicit MLFloat16(float ff) : val(MLAS_Float2Half(ff)) {}
    constexpr static MLFloat16 FromBits(uint16_t x) noexcept { return MLFloat16(x); }

    MLFloat16 Abs() const noexcept {
        return MLFloat16(static_cast<uint16_t>(val & ~kSignMask));
    }
    bool IsNaN() const noexcept {
        return Abs().val > kPositiveInfinityBits;
    }
    bool IsNegative() const noexcept {
        return static_cast<int16_t>(val) < 0;
    }
    MLFloat16 Negate() const {
        return MLFloat16(IsNaN() ? val : static_cast<uint16_t>(val ^ kSignMask));
    }
    static constexpr uint16_t kSignMask = 0x8000U;
    static constexpr uint16_t kPositiveInfinityBits = 0x7C00U;

    float ToFloat() const { return MLAS_Half2Float(val); }`,
    );
    writeFileSync(mlasi, src);
    console.log("Patched mlasi.h MLFloat16 helpers for single-thread WASM");
  }
}

const buildSh = `
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
source "${envFile}"
cd "${ortSrc}"
./build.sh --config Release --build_wasm_static_lib --enable_wasm_simd \\
  --skip_tests --disable_wasm_exception_catching --disable_rtti --parallel
`;
run("bash", ["-lc", buildSh]);

const built = path.join(
  ortSrc,
  "build/Linux/Release/libonnxruntime_webassembly.a",
);
if (!existsSync(built)) {
  // Some ORT versions put it under build/Release
  const alt = path.join(ortSrc, "build/Release/libonnxruntime_webassembly.a");
  if (!existsSync(alt)) {
    throw new Error(`libonnxruntime_webassembly.a not found after build`);
  }
  mkdirSync(path.dirname(libOut), { recursive: true });
  copyFileSync(alt, libOut);
} else {
  mkdirSync(path.dirname(libOut), { recursive: true });
  copyFileSync(built, libOut);
}

const includeSrc = path.join(
  ortSrc,
  "include/onnxruntime/core/session",
);
const includeDst = path.join(stage, "include");
mkdirSync(includeDst, { recursive: true });
for (const name of [
  "onnxruntime_c_api.h",
  "onnxruntime_cxx_api.h",
  "onnxruntime_cxx_inline.h",
]) {
  const src = path.join(includeSrc, name);
  if (existsSync(src)) copyFileSync(src, path.join(includeDst, name));
}

writeFileSync(
  path.join(stage, "active.json"),
  JSON.stringify(
    {
      version,
      lib: "lib/libonnxruntime_webassembly.a",
      simd: true,
      threads: false,
      builtAt: new Date().toISOString(),
    },
    null,
    2,
  ) + "\n",
);

console.log(`Staged ORT wasm static lib → ${libOut}`);
console.log(`Headers → ${includeDst}`);
