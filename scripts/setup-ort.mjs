#!/usr/bin/env node
/**
 * Download ONNX Runtime (+ optional CUDA package) and the native WebGPU EP plugin.
 *
 *   npm run setup:ort                              # CPU ORT + WebGPU plugin (default)
 *   TRUEPIXEL_ORT_VARIANT=gpu npm run setup:ort    # CUDA ORT + WebGPU plugin
 *   TRUEPIXEL_ORT_VARIANT=auto npm run setup:ort   # CUDA if nvidia-smi, else CPU
 *   TRUEPIXEL_SKIP_WEBGPU_EP=1                     # skip plugin download
 *
 * Native cross-vendor GPU = WebGPU plugin (Dawn → Vulkan on Linux). There is no
 * separate ORT "Vulkan" EP. Browser WebGPU stays on onnxruntime-web.
 */
import {
  createWriteStream,
  existsSync,
  mkdirSync,
  rmSync,
  writeFileSync,
  copyFileSync,
  readdirSync,
} from "node:fs";
import { pipeline } from "node:stream/promises";
import { execFileSync, spawnSync } from "node:child_process";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { Readable } from "node:stream";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(__dirname, "..");
const version = process.env.TRUEPIXEL_ORT_VERSION ?? "1.29.0";
const webgpuEpVersion = process.env.TRUEPIXEL_WEBGPU_EP_VERSION ?? "0.2.1";
const ortDir = path.join(root, "native/ort");

function detectVariant() {
  const forced = (process.env.TRUEPIXEL_ORT_VARIANT ?? "cpu").toLowerCase();
  if (forced === "gpu" || forced === "cuda") return "gpu";
  if (forced === "cpu") return "cpu";
  if (forced === "auto") {
    const nv = spawnSync("nvidia-smi", ["-L"], { encoding: "utf8" });
    if (nv.status === 0 && (nv.stdout ?? "").includes("GPU")) return "gpu";
    return "cpu";
  }
  return "cpu";
}

function gpuPackageName(ver) {
  // 1.29+ ships cuda12/cuda13 suffixes; 1.24.x used -gpu-.
  const majorMinor = ver.split(".").map(Number);
  if (majorMinor[0] > 1 || (majorMinor[0] === 1 && majorMinor[1] >= 29)) {
    const cuda = process.env.TRUEPIXEL_ORT_CUDA ?? "12";
    return `onnxruntime-linux-x64-gpu_cuda${cuda}-${ver}`;
  }
  return `onnxruntime-linux-x64-gpu-${ver}`;
}

async function download(url, dest) {
  console.log(`Downloading ${url} …`);
  const res = await fetch(url);
  if (!res.ok) throw new Error(`download failed: ${res.status} ${res.statusText} (${url})`);
  await pipeline(Readable.fromWeb(res.body), createWriteStream(dest));
}

function findSoVersion(libDir) {
  const files = readdirSync(libDir);
  const ver = files.find((f) => /^libonnxruntime\.so\.\d+\.\d+\.\d+$/.test(f));
  if (!ver) return null;
  return ver.replace("libonnxruntime.so.", "");
}

const variant = detectVariant();
const name =
  variant === "gpu" ? gpuPackageName(version) : `onnxruntime-linux-x64-${version}`;
const dest = path.join(ortDir, name);
const marker = path.join(dest, "lib", "libonnxruntime.so");
const activeLink = path.join(ortDir, "active");
const webgpuDestDir = path.join(ortDir, "webgpu-ep");
const webgpuSoName = "libonnxruntime_providers_webgpu.so";
const webgpuSo = path.join(webgpuDestDir, webgpuSoName);

mkdirSync(ortDir, { recursive: true });

if (existsSync(marker) && process.env.TRUEPIXEL_FORCE_ORT !== "1") {
  console.log(`ORT already present: ${dest}`);
} else {
  const url = `https://github.com/microsoft/onnxruntime/releases/download/v${version}/${name}.tgz`;
  const tgz = path.join(ortDir, `${name}.tgz`);

  await download(url, tgz);

  if (existsSync(dest)) rmSync(dest, { recursive: true, force: true });
  console.log(`Extracting to ${dest} …`);
  execFileSync("tar", ["-xzf", tgz, "-C", ortDir], { stdio: "inherit" });
  rmSync(tgz, { force: true });

  if (!existsSync(marker)) {
    throw new Error(`ORT extract missing ${marker}`);
  }
}

const skipWebgpu = process.env.TRUEPIXEL_SKIP_WEBGPU_EP === "1";
if (!skipWebgpu) {
  if (existsSync(webgpuSo) && process.env.TRUEPIXEL_FORCE_ORT !== "1") {
    console.log(`WebGPU EP already present: ${webgpuSo}`);
  } else {
    mkdirSync(webgpuDestDir, { recursive: true });
    const nupkg = path.join(
      ortDir,
      `Microsoft.ML.OnnxRuntime.EP.WebGpu.${webgpuEpVersion}.nupkg`,
    );
    const nugetUrl =
      `https://api.nuget.org/v3-flatcontainer/microsoft.ml.onnxruntime.ep.webgpu/` +
      `${webgpuEpVersion}/microsoft.ml.onnxruntime.ep.webgpu.${webgpuEpVersion}.nupkg`;
    await download(nugetUrl, nupkg);

    const extractDir = path.join(ortDir, `webgpu-ep-extract-${webgpuEpVersion}`);
    if (existsSync(extractDir)) rmSync(extractDir, { recursive: true, force: true });
    mkdirSync(extractDir, { recursive: true });
    // nupkg is zip; prefer unzip, fall back to Python zipfile.
    const unzip = spawnSync("unzip", ["-qo", nupkg, "-d", extractDir], {
      encoding: "utf8",
    });
    if (unzip.status !== 0) {
      execFileSync(
        "python3",
        ["-c", "import sys,zipfile; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])", nupkg, extractDir],
        { stdio: "inherit" },
      );
    }

    const nested = path.join(
      extractDir,
      "runtimes/linux-x64/native",
      webgpuSoName,
    );
    if (!existsSync(nested)) {
      throw new Error(`WebGPU EP nupkg missing ${nested}`);
    }
    copyFileSync(nested, webgpuSo);
    rmSync(extractDir, { recursive: true, force: true });
    rmSync(nupkg, { force: true });
    console.log(`WebGPU EP ready: ${webgpuSo}`);
  }

  // Also stage next to ORT libs so RPATH / $ORIGIN/../lib finds it.
  copyFileSync(webgpuSo, path.join(dest, "lib", webgpuSoName));
}

rmSync(activeLink, { recursive: true, force: true });
execFileSync("ln", ["-sfn", name, activeLink], { cwd: ortDir });

const soVersion = findSoVersion(path.join(dest, "lib")) ?? version;
const active = {
  variant,
  version,
  soVersion,
  name,
  path: dest,
  webgpuEpVersion: skipWebgpu ? null : webgpuEpVersion,
  webgpuEpPath: skipWebgpu ? null : webgpuSo,
};
writeFileSync(path.join(ortDir, "active.json"), JSON.stringify(active, null, 2) + "\n");

console.log(`ORT ready (${variant}): ${dest}`);
if (!skipWebgpu) {
  console.log(
    "WebGPU plugin staged (Dawn→Vulkan on Linux). Zig registers it at runtime.",
  );
}
if (variant === "gpu") {
  console.log("CUDA package selected — needs a matching CUDA toolkit at runtime.");
} else if (skipWebgpu) {
  console.log("CPU-only (WebGPU plugin skipped).");
}
