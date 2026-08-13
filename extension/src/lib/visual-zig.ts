/**
 * Zig+ORT native visual backend (x86_64_v3 / ORT MLAS AVX2+).
 * Spawns a long-lived `truepixel-infer` process and speaks JSONL.
 * Browser runtime still uses onnxruntime-web; this path is for Node eval / host tools.
 */
import { spawn, type ChildProcessWithoutNullStreams } from "node:child_process";
import { createInterface } from "node:readline";
import { existsSync } from "node:fs";
import path from "node:path";
import { asAiConfidence, type InferenceBackend } from "../shared/types";
import type { VisualClassification } from "./visual-stub";
export { needsForensicsCascade } from "./forensics-cascade";

export type ZigInferRequest = {
  /** Absolute or repo-relative image path. */
  imagePath: string;
  runDistilled?: boolean;
  runForensics?: boolean;
};

export type ZigInferResult = VisualClassification & {
  secondaryScore?: ReturnType<typeof asAiConfidence>;
  distilledMs: number;
  forensicsMs: number;
  ranDistilled: boolean;
  ranForensics: boolean;
  inferMs: number;
  preprocessMs: number;
  preferEp?: string;
  distilledEp?: string;
  forensicsEp?: string;
};

type Pending = {
  line: string;
  resolve: (value: ZigInferResult) => void;
  reject: (err: Error) => void;
};

let child: ChildProcessWithoutNullStreams | undefined;
let ready: Promise<void> | undefined;
let queue: Pending[] = [];
let inFlight = false;

export function zigInferBinaryExists(): boolean {
  try {
    resolveBinary();
    return true;
  } catch {
    return false;
  }
}

function resolveBinary(): string {
  const candidates = [
    process.env.TRUEPIXEL_ZIG_INFER,
    path.resolve("native/zig-infer/zig-out/bin/truepixel-infer"),
    path.resolve(
      process.cwd(),
      "native/zig-infer/zig-out/bin/truepixel-infer",
    ),
  ].filter(Boolean) as string[];
  for (const candidate of candidates) {
    if (existsSync(candidate)) return candidate;
  }
  throw new Error(
    "truepixel-infer missing. Build with: npm run build:zig",
  );
}

function ensureChild(): Promise<void> {
  if (ready) return ready;
  ready = new Promise((resolve, reject) => {
    const bin = resolveBinary();
    child = spawn(bin, [], {
      stdio: ["pipe", "pipe", "pipe"],
      env: {
        ...process.env,
        TRUEPIXEL_ROOT: process.env.TRUEPIXEL_ROOT ?? process.cwd(),
      },
    });
    const rl = createInterface({ input: child.stdout });
    let settled = false;

    const finish = (msg: Record<string, unknown>) => {
      const pending = queue.shift();
      inFlight = false;
      if (!pending) {
        flush();
        return;
      }
      if (msg.ok !== true && msg.event !== "warm" && msg.event !== "pong") {
        pending.reject(
          new Error(
            `zig-infer failed: ${String(msg.error ?? "unknown")} ${String(msg.detail ?? "")}`,
          ),
        );
        flush();
        return;
      }
      if (msg.event === "warm" || msg.event === "pong") {
        pending.resolve({
          score: asAiConfidence(0.5),
          backend: { kind: "wasm" },
          detail: msg.event === "warm" ? "zig-warm" : "zig-pong",
          distilledMs: 0,
          forensicsMs: 0,
          ranDistilled: false,
          ranForensics: false,
          inferMs: Number(msg.ms ?? 0),
          preprocessMs: 0,
          preferEp: String(msg.preferEp ?? "auto"),
          distilledEp: String(msg.distilledEp ?? "unknown"),
          forensicsEp: String(msg.forensicsEp ?? "unknown"),
        });
        flush();
        return;
      }
      const distilled = Number(msg.distilled ?? 0.5);
      const forensics = Number(msg.forensics ?? 0.5);
      const distilledMs = Number(msg.distilledMs ?? 0);
      const forensicsMs = Number(msg.forensicsMs ?? 0);
      const ranDistilled = Boolean(msg.ranDistilled);
      const ranForensics = Boolean(msg.ranForensics);
      const backend: InferenceBackend = { kind: "wasm" };
      pending.resolve({
        score: asAiConfidence(distilled),
        ...(ranForensics
          ? { secondaryScore: asAiConfidence(forensics) }
          : {}),
        backend,
        detail: `zig:distilled=${distilled.toFixed(3)}${
          ranForensics ? `,forensics=${forensics.toFixed(3)}` : ""
        }`,
        distilledMs,
        forensicsMs,
        ranDistilled,
        ranForensics,
        inferMs: distilledMs + forensicsMs,
        preprocessMs: 0,
      });
      flush();
    };

    rl.on("line", (line) => {
      let msg: Record<string, unknown>;
      try {
        msg = JSON.parse(line) as Record<string, unknown>;
      } catch {
        return;
      }
      if (!settled && msg.event === "ready") {
        settled = true;
        resolve();
        return;
      }
      finish(msg);
    });

    child.stderr.on("data", (chunk: Buffer) => {
      process.stderr.write(`[zig-infer] ${chunk.toString()}`);
    });
    child.on("exit", (code) => {
      ready = undefined;
      child = undefined;
      inFlight = false;
      const err = new Error(`zig-infer exited (${code})`);
      while (queue.length) queue.shift()?.reject(err);
      if (!settled) reject(err);
    });
  });
  return ready;
}

function flush() {
  if (inFlight || !child || queue.length === 0) return;
  inFlight = true;
  child.stdin.write(`${queue[0]!.line}\n`);
}

function send(line: string): Promise<ZigInferResult> {
  return ensureChild().then(
    () =>
      new Promise<ZigInferResult>((resolve, reject) => {
        queue.push({ line, resolve, reject });
        flush();
      }),
  );
}

export type ZigPreferEp = "WebGPU" | "Vulkan" | "XNNPACK" | "CPU";

export async function warmVisualZig(preferEp?: ZigPreferEp): Promise<{
  preferEp: string;
  distilledEp: string;
  forensicsEp: string;
  ms: number;
}> {
  const payload = preferEp
    ? JSON.stringify({ cmd: "warm", preferEp })
    : '{"cmd":"warm"}';
  const result = await send(payload);
  return {
    preferEp: result.preferEp ?? preferEp ?? "auto",
    distilledEp: result.distilledEp ?? "unknown",
    forensicsEp: result.forensicsEp ?? "unknown",
    ms: result.inferMs,
  };
}

export async function classifyZigVisual(
  args: ZigInferRequest,
): Promise<ZigInferResult> {
  const runDistilled = args.runDistilled !== false;
  const runForensics = args.runForensics === true;
  const models: string[] = [];
  if (runDistilled) models.push("distilled");
  if (runForensics) models.push("forensics");
  if (models.length === 0) models.push("distilled");

  const payload = JSON.stringify({
    cmd: "infer",
    path: args.imagePath,
    models,
  });
  return send(payload);
}

export async function shutdownVisualZig(): Promise<void> {
  if (!child) return;
  const proc = child;
  child = undefined;
  ready = undefined;
  queue = [];
  inFlight = false;
  await new Promise<void>((resolve) => {
    const done = () => resolve();
    proc.once("exit", done);
    try {
      proc.stdin.write('{"cmd":"quit"}\n');
    } catch {
      /* ignore */
    }
    // Hard kill if quit doesn't land promptly (avoids stuck GPU EP probes).
    setTimeout(() => {
      try {
        proc.kill("SIGKILL");
      } catch {
        /* ignore */
      }
    }, 1500);
  });
}
