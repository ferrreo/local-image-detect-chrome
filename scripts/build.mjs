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
    path.join(root, "extension/src/content/overlay.css"),
    path.join(outdir, "overlay.css"),
  );
  cpSync(path.join(root, "extension/icons"), path.join(outdir, "icons"), {
    recursive: true,
  });

  // Bundle onnxruntime WASM assets for extension pages.
  const ortPkg = path.join(root, "node_modules/onnxruntime-web/dist");
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

async function buildOnce() {
  rmSync(outdir, { recursive: true, force: true });
  mkdirSync(outdir, { recursive: true });

  // Background/content/popup/options must stay free of onnxruntime-web.
  await esbuild.build({
    entryPoints: {
      background: entryPoints.background,
      content: entryPoints.content,
      popup: entryPoints.popup,
      options: entryPoints.options,
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
