#!/usr/bin/env node
/**
 * Build native/zig-infer (ReleaseFast, x86_64_v3). Ensures ORT is present first.
 */
import { execFileSync, spawnSync } from "node:child_process";
import { existsSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(__dirname, "..");
const zigDir = path.join(root, "native/zig-infer");

execFileSync(process.execPath, [path.join(root, "scripts/setup-ort.mjs")], {
  stdio: "inherit",
  cwd: root,
});

function findZig() {
  if (process.env.ZIG) return process.env.ZIG;
  const which = spawnSync("zig", ["version"], { encoding: "utf8" });
  if (which.status === 0 && which.stdout.trim().startsWith("0.16")) {
    return "zig";
  }
  const candidates = [
    "/tmp/zig-x86_64-linux-0.16.0/zig",
    path.join(root, "tools/zig/zig"),
  ];
  for (const c of candidates) {
    if (existsSync(c)) return c;
  }
  throw new Error(
    "Zig 0.16 required. Install from https://ziglang.org/download/ and ensure `zig` is on PATH, or set ZIG=...",
  );
}

const zig = findZig();
const ver = execFileSync(zig, ["version"], { encoding: "utf8" }).trim();
console.log(`Using ${zig} (${ver})`);
execFileSync(zig, ["build", "-Doptimize=ReleaseFast"], {
  stdio: "inherit",
  cwd: zigDir,
});
console.log(`Built ${path.join(zigDir, "zig-out/bin/truepixel-infer")}`);
