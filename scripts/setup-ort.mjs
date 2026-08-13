#!/usr/bin/env node
/**
 * Download ONNX Runtime 1.22.0 Linux x64 (shared libs) into native/ort/.
 * The Zig host links these at build time and copies them next to the binary.
 */
import { createWriteStream, existsSync, mkdirSync, rmSync } from "node:fs";
import { pipeline } from "node:stream/promises";
import { createGunzip } from "node:zlib";
import { execFileSync } from "node:child_process";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { Readable } from "node:stream";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(__dirname, "..");
const version = "1.22.0";
const name = `onnxruntime-linux-x64-${version}`;
const dest = path.join(root, "native/ort", name);
const marker = path.join(dest, "lib", "libonnxruntime.so");

if (existsSync(marker) && process.env.TRUEPIXEL_FORCE_ORT !== "1") {
  console.log(`ORT already present: ${dest}`);
  process.exit(0);
}

const url = `https://github.com/microsoft/onnxruntime/releases/download/v${version}/${name}.tgz`;
const tmpDir = path.join(root, "native/ort");
mkdirSync(tmpDir, { recursive: true });
const tgz = path.join(tmpDir, `${name}.tgz`);

console.log(`Downloading ${url} …`);
const res = await fetch(url);
if (!res.ok) throw new Error(`download failed: ${res.status} ${res.statusText}`);
await pipeline(Readable.fromWeb(res.body), createWriteStream(tgz));

if (existsSync(dest)) rmSync(dest, { recursive: true, force: true });
console.log(`Extracting to ${dest} …`);
execFileSync("tar", ["-xzf", tgz, "-C", tmpDir], { stdio: "inherit" });
rmSync(tgz, { force: true });

if (!existsSync(marker)) {
  throw new Error(`ORT extract missing ${marker}`);
}
console.log("ORT ready.");
