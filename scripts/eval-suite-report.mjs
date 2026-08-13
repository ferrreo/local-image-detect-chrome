#!/usr/bin/env node
/** Merge host + browser JSON into latest.json and a standalone HTML report. */
import { existsSync, readFileSync, writeFileSync, mkdirSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(__dirname, "..");
const outDir = path.join(root, "benchmark/eval-suite");
mkdirSync(outDir, { recursive: true });

function readJson(name) {
  const p = path.join(outDir, name);
  if (!existsSync(p)) return null;
  return JSON.parse(readFileSync(p, "utf8"));
}

const host = readJson("host-latest.json");
const browser = readJson("browser-latest.json");
const results = [
  ...(host?.results ?? []),
  ...(browser?.results ?? []),
];

const latest = {
  generatedAt: new Date().toISOString(),
  host: host ?? null,
  browser: browser ?? null,
  summary: results.map((r) => ({
    mode: r.mode,
    skipped: Boolean(r.skipped),
    error: r.error ?? null,
    ba: r.balancedAccuracy ?? null,
    tp: r.confusion?.tp ?? null,
    tn: r.confusion?.tn ?? null,
    fp: r.confusion?.fp ?? null,
    fn: r.confusion?.fn ?? null,
    avgMs: r.timing?.avgTotalMs ?? null,
    preferEp: r.preferEp ?? null,
    actualEp: r.distilledEp ?? r.providerActual ?? null,
    gpuAvailable: r.gpuAvailable ?? null,
    engine: r.engine ?? null,
    runtime: r.runtime ?? null,
  })),
  results,
};

writeFileSync(
  path.join(outDir, "latest.json"),
  JSON.stringify(latest, null, 2) + "\n",
);

const rowsHtml = latest.summary
  .map((s) => {
    if (s.skipped) {
      return `<tr class="skip"><td>${s.mode}</td><td colspan="8">skipped${s.error ? `: ${s.error}` : ""}</td></tr>`;
    }
    if (s.error) {
      return `<tr class="err"><td>${s.mode}</td><td colspan="8">error: ${s.error}</td></tr>`;
    }
    return `<tr>
      <td>${s.mode}</td>
      <td>${s.engine ?? ""}</td>
      <td>${s.preferEp ?? ""}</td>
      <td>${s.actualEp ?? ""}</td>
      <td>${s.ba != null ? (s.ba * 100).toFixed(1) + "%" : ""}</td>
      <td>${s.tp}/${s.tn}/${s.fp}/${s.fn}</td>
      <td>${s.avgMs != null ? s.avgMs.toFixed(1) : ""}</td>
      <td>${s.gpuAvailable == null ? "" : s.gpuAvailable ? "yes" : "no"}</td>
      <td>${s.runtime ?? ""}</td>
    </tr>`;
  })
  .join("\n");

const html = `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>TruePixel eval suite</title>
<style>
:root { --ink:#18201c; --muted:#5b655e; --line:#cdc6b8; --ok:#1f6b3a; --bad:#8b2e2e; }
body { margin:0; font-family:"IBM Plex Sans",Segoe UI,sans-serif; color:var(--ink);
  background:linear-gradient(180deg,#efebe1,#e7e0d3); }
main { max-width:1100px; margin:0 auto; padding:2rem 1.25rem 4rem; }
h1 { margin:0; font-size:clamp(2.2rem,5vw,3.4rem); letter-spacing:-.04em; }
.lede { color:var(--muted); max-width:40rem; }
table { width:100%; border-collapse:collapse; margin-top:1.5rem; background:rgba(255,253,248,.75); }
th,td { border-bottom:1px solid var(--line); padding:.55rem .5rem; text-align:left; font-size:.9rem; }
th { font-size:.72rem; text-transform:uppercase; letter-spacing:.04em; color:var(--muted); }
.skip td, .err td { color:var(--bad); }
code { font-family:"IBM Plex Mono",ui-monospace,monospace; font-size:.85em; }
</style>
</head>
<body>
<main>
  <h1>TruePixel</h1>
  <p class="lede">Offline eval suite — Node/Zig host CPU+GPU prefs and Chromium extension WebGPU/WASM via Playwright.</p>
  <p>Generated <code>${latest.generatedAt}</code>. Live runner: load unpacked <code>dist/</code> → <code>chrome-extension://&lt;id&gt;/eval.html</code>.</p>
  <table>
    <thead>
      <tr>
        <th>mode</th><th>engine</th><th>prefer</th><th>actual</th>
        <th>BA</th><th>tp/tn/fp/fn</th><th>avg ms</th><th>gpu</th><th>runtime</th>
      </tr>
    </thead>
    <tbody>
      ${rowsHtml}
    </tbody>
  </table>
</main>
</body>
</html>
`;

writeFileSync(path.join(outDir, "index.html"), html);
console.log(`Wrote ${path.join(outDir, "latest.json")}`);
console.log(`Wrote ${path.join(outDir, "index.html")}`);
