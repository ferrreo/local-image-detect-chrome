import * as esbuild from "esbuild";
import {
  cpSync,
  mkdirSync,
  rmSync,
  existsSync,
  readFileSync,
  writeFileSync,
  readdirSync,
} from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(__dirname, "..");
const outdir = path.join(root, "dist");
const watch = process.argv.includes("--watch");

const entryPoints = {
  background: path.join(root, "extension/src/background/service-worker.ts"),
  content: path.join(root, "extension/src/content/content.ts"),
  offscreen: path.join(root, "extension/src/offscreen/offscreen.ts"),
  popup: path.join(root, "extension/src/popup/popup.ts"),
  options: path.join(root, "extension/src/options/options.ts"),
  eval: path.join(root, "extension/src/eval/eval.ts"),
};

function copyStatic() {
  mkdirSync(outdir, { recursive: true });
  cpSync(
    path.join(root, "extension/manifest.json"),
    path.join(outdir, "manifest.json"),
  );
  cpSync(
    path.join(root, "extension/src/offscreen/offscreen.html"),
    path.join(outdir, "offscreen.html"),
  );
  cpSync(
    path.join(root, "extension/src/popup/popup.html"),
    path.join(outdir, "popup.html"),
  );
  cpSync(
    path.join(root, "extension/src/popup/popup.css"),
    path.join(outdir, "popup.css"),
  );
  cpSync(
    path.join(root, "extension/src/options/options.html"),
    path.join(outdir, "options.html"),
  );
  cpSync(
    path.join(root, "extension/src/options/options.css"),
    path.join(outdir, "options.css"),
  );
  cpSync(
    path.join(root, "extension/src/eval/eval.html"),
    path.join(outdir, "eval.html"),
  );
  cpSync(
    path.join(root, "extension/src/eval/eval.css"),
    path.join(outdir, "eval.css"),
  );
  cpSync(
    path.join(root, "extension/src/content/overlay.css"),
    path.join(outdir, "overlay.css"),
  );
  cpSync(path.join(root, "extension/icons"), path.join(outdir, "icons"), {
    recursive: true,
  });

  // Bundle onnxruntime WASM assets for extension pages.
  // Prefer vendored ORT #29599 build when present (MatMulNBits f32 accumulators).
  const vendorOrt = path.join(root, "vendor/onnxruntime-web/dist");
  const npmOrt = path.join(root, "node_modules/onnxruntime-web/dist");
  const ortPkg = existsSync(
    path.join(vendorOrt, "ort.webgpu.bundle.min.mjs"),
  )
    ? vendorOrt
    : npmOrt;
  const ortOut = path.join(outdir, "ort");
  mkdirSync(ortOut, { recursive: true });
  if (existsSync(ortPkg)) {
    for (const file of readdirSync(ortPkg)) {
      if (
        file.startsWith("ort-wasm") &&
        (file.endsWith(".wasm") || file.endsWith(".mjs"))
      ) {
        cpSync(path.join(ortPkg, file), path.join(ortOut, file));
      }
    }
  }

  // Optional: seed packaged models if present (still one-time download path is primary).
  const modelsSrc = path.join(root, "models");
  if (existsSync(modelsSrc)) {
    cpSync(modelsSrc, path.join(outdir, "models"), { recursive: true });
  }

  // Zig+ORT WASM (linked libonnxruntime_webassembly.a) when built.
  const zigWasmSrc = path.join(root, "native/zig-infer/zig-out/wasm");
  if (
    existsSync(path.join(zigWasmSrc, "truepixel_infer.wasm")) &&
    existsSync(path.join(zigWasmSrc, "truepixel_infer.js"))
  ) {
    const zigWasmOut = path.join(outdir, "wasm");
    mkdirSync(zigWasmOut, { recursive: true });
    cpSync(
      path.join(zigWasmSrc, "truepixel_infer.wasm"),
      path.join(zigWasmOut, "truepixel_infer.wasm"),
    );
    cpSync(
      path.join(zigWasmSrc, "truepixel_infer.js"),
      path.join(zigWasmOut, "truepixel_infer.js"),
    );
  }

  // Stamp build metadata.
  writeFileSync(
    path.join(outdir, "build.json"),
    JSON.stringify(
      {
        builtAt: new Date().toISOString(),
        stubDefault: process.env.TRUEPIXEL_STUB_INFERENCE === "1",
      },
      null,
      2,
    ),
  );

  // Keep a copy of manifest description for sanity.
  const manifest = JSON.parse(
    readFileSync(path.join(outdir, "manifest.json"), "utf8"),
  );
  if (manifest.manifest_version !== 3) {
    throw new Error("manifest_version must be 3");
  }
}

function ortWebAlias() {
  const vendorEntry = path.join(
    root,
    "vendor/onnxruntime-web/dist/ort.webgpu.bundle.min.mjs",
  );
  if (!existsSync(vendorEntry)) return {};
  console.log("Using vendored onnxruntime-web (PR #29599)");
  return {
    "onnxruntime-web": path.join(root, "vendor/onnxruntime-web"),
    "onnxruntime-web/webgpu": vendorEntry,
  };
}

async function buildOnce() {
  rmSync(outdir, { recursive: true, force: true });
  mkdirSync(outdir, { recursive: true });
  const alias = ortWebAlias();

  // Background/content/popup/options must stay free of onnxruntime-web.
  await esbuild.build({
    entryPoints: {
      background: entryPoints.background,
      content: entryPoints.content,
      popup: entryPoints.popup,
      options: entryPoints.options,
      eval: entryPoints.eval,
    },
    outdir,
    entryNames: "[name]",
    bundle: true,
    format: "esm",
    platform: "browser",
    target: ["chrome121"],
    sourcemap: true,
    logLevel: "info",
    external: ["onnxruntime-web"],
  });

  // Offscreen document owns WebGPU/WASM ONNX inference.
  await esbuild.build({
    entryPoints: { offscreen: entryPoints.offscreen },
    outdir,
    entryNames: "[name]",
    bundle: true,
    format: "esm",
    platform: "browser",
    target: ["chrome121"],
    sourcemap: true,
    logLevel: "info",
    loader: { ".wasm": "file" },
    ...(Object.keys(alias).length
      ? { alias }
      : {}),
  });

  copyStatic();
  console.log(`Built extension → ${outdir}`);
}

if (watch) {
  copyStatic();
  const ctx = await esbuild.context({
    entryPoints,
    outdir,
    entryNames: "[name]",
    bundle: true,
    format: "esm",
    platform: "browser",
    target: ["chrome121"],
    sourcemap: true,
    logLevel: "info",
  });
  await ctx.watch();
  console.log("Watching…");
} else {
  await buildOnce();
}
