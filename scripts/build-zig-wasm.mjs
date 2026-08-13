#!/usr/bin/env node
/**
 * Link Zig SIMD preprocess + C ORT bridge + libonnxruntime_webassembly.a via emcc.
 *
 * Prerequisites:
 *   npm run setup:ort-wasm
 *
 * Output:
 *   native/zig-infer/zig-out/wasm/truepixel_infer.js
 *   native/zig-infer/zig-out/wasm/truepixel_infer.wasm
 */
import { execFileSync, spawnSync } from "node:child_process";
import {
  existsSync,
  mkdirSync,
  readFileSync,
  writeFileSync,
  rmSync,
} from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(__dirname, "..");
const zigDir = path.join(root, "native/zig-infer");
const ortWasm = path.join(root, "native/ort-wasm");
const libA = path.join(ortWasm, "lib/libonnxruntime_webassembly.a");
const includeDir = path.join(ortWasm, "include");
const outDir = path.join(zigDir, "zig-out/wasm");
const zigObj = path.join(outDir, "wasm_api.o");

function findZig() {
  if (process.env.ZIG) return process.env.ZIG;
  const which = spawnSync("zig", ["version"], { encoding: "utf8" });
  if (which.status === 0 && which.stdout.trim().startsWith("0.16")) {
    return "zig";
  }
  for (const c of [
    "/tmp/zig-x86_64-linux-0.16.0/zig",
    path.join(root, "tools/zig/zig"),
  ]) {
    if (existsSync(c)) return c;
  }
  throw new Error("Zig 0.16 required for wasm ORT link");
}

function findEmcc() {
  if (process.env.EMCC) return process.env.EMCC;
  const candidates = [
    path.join(
      root,
      "native/ort-wasm-src/onnxruntime/cmake/external/emsdk/upstream/emscripten/emcc",
    ),
    "emcc",
  ];
  for (const c of candidates) {
    if (c !== "emcc" && !existsSync(c)) continue;
    const r = spawnSync(c, ["-v"], { encoding: "utf8" });
    if (r.status === 0 || (r.stderr && String(r.stderr).includes("emcc"))) {
      return c;
    }
  }
  throw new Error(
    "emcc not found. Run npm run setup:ort-wasm (installs emsdk) or set EMCC=",
  );
}

if (!existsSync(libA)) {
  console.log("ORT wasm static lib missing; running setup:ort-wasm…");
  execFileSync(process.execPath, [path.join(root, "scripts/setup-ort-wasm.mjs")], {
    stdio: "inherit",
    cwd: root,
  });
}
if (!existsSync(libA)) {
  throw new Error(`Still missing ${libA}`);
}
if (!existsSync(path.join(includeDir, "onnxruntime_c_api.h"))) {
  throw new Error(`Missing ORT headers under ${includeDir}`);
}

mkdirSync(outDir, { recursive: true });
const zig = findZig();
const emcc = findEmcc();
console.log(`zig=${zig}`);
console.log(`emcc=${emcc}`);

// Zig freestanding object: SIMD preprocess only (no emscripten stdlib needed).
execFileSync(
  zig,
  [
    "build-obj",
    path.join(zigDir, "src/wasm_api.zig"),
    `-femit-bin=${zigObj}`,
    "-target",
    "wasm32-freestanding",
    "-O",
    "ReleaseFast",
    "-mcpu=generic+simd128",
    "-fno-entry",
  ],
  { stdio: "inherit", cwd: zigDir },
);

const exports = [
  "_tp_abi_version",
  "_tp_has_ort_session",
  "_tp_malloc",
  "_tp_free",
  "_tp_session_create",
  "_tp_session_free",
  "_tp_session_run",
  "_tp_rgb_to_nchw_half",
  "_tp_rgba_resize_nchw",
  "_tp_softmax2",
].join(",");

const jsOut = path.join(outDir, "truepixel_infer.js");
const wasmOut = path.join(outDir, "truepixel_infer.wasm");
rmSync(jsOut, { force: true });
rmSync(wasmOut, { force: true });

execFileSync(
  emcc,
  [
    path.join(zigDir, "src/wasm_ort_bridge.c"),
    zigObj,
    libA,
    `-I${includeDir}`,
    "-o",
    jsOut,
    "-O3",
    "-msimd128",
    "-s",
    "MODULARIZE=1",
    "-s",
    "EXPORT_ES6=1",
    "-s",
    "EXPORT_NAME=createTruepixelInfer",
    "-s",
    "ALLOW_MEMORY_GROWTH=1",
    "-s",
    "INITIAL_MEMORY=67108864",
    "-s",
    "STACK_SIZE=5242880",
    "-s",
    `EXPORTED_FUNCTIONS=${exports}`,
    "-s",
    "EXPORTED_RUNTIME_METHODS=HEAPU8,HEAPF32,UTF8ToString",
    "-s",
    "ENVIRONMENT=web,worker",
    "-s",
    "FILESYSTEM=0",
    "-s",
    "ASSERTIONS=0",
    "-Wno-limited-postlink-optimizations",
  ],
  { stdio: "inherit", cwd: zigDir },
);

if (!existsSync(wasmOut)) {
  throw new Error(`emcc did not produce ${wasmOut}`);
}

writeFileSync(
  path.join(outDir, "build.json"),
  JSON.stringify(
    {
      builtAt: new Date().toISOString(),
      ortWasm: existsSync(path.join(ortWasm, "active.json"))
        ? JSON.parse(readFileSync(path.join(ortWasm, "active.json"), "utf8"))
        : null,
      wasmBytes: readFileSync(wasmOut).byteLength,
    },
    null,
    2,
  ) + "\n",
);

console.log(`Built Zig+ORT WASM → ${jsOut}`);
console.log(`                   → ${wasmOut}`);
