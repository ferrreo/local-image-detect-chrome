import type {
  AnalyzeImageResponse,
  DetectionResult,
  ExtensionResponse,
  GetStatusResponse,
  ResetVisualResponse,
  SetupModelsResponse,
  VisualProvider,
} from "../shared/types";

type CorpusImage = {
  file: string;
  label: "ai" | "real";
  model: string | null;
};

type Row = {
  file: string;
  label: "ai" | "real";
  model: string | null;
  confidence: number;
  predicted: "ai" | "real";
  correct: boolean;
  backend: string;
  totalMs: number;
  elapsedMs: number;
  tiers: string;
  error?: string;
};

type SuiteResult = {
  kind: "truepixel-browser-eval";
  generatedAt: string;
  corpusBase: string;
  providerRequested: VisualProvider["kind"];
  providerActual: string;
  gpuAvailable: boolean;
  threshold: number;
  balancedAccuracy: number;
  confusion: { tp: number; tn: number; fp: number; fn: number };
  timing: { avgTotalMs: number; count: number };
  rows: Row[];
};

function qs<T extends HTMLElement>(id: string): T {
  const el = document.getElementById(id);
  if (!el) throw new Error(`#${id} missing`);
  return el as T;
}

const corpusBaseInput = qs<HTMLInputElement>("corpusBase");
const providerSelect = qs<HTMLSelectElement>("provider");
const thresholdInput = qs<HTMLInputElement>("threshold");
const setupBtn = qs<HTMLButtonElement>("setupBtn");
const runBtn = qs<HTMLButtonElement>("runBtn");
const statusLine = qs<HTMLElement>("statusLine");
const metaLine = qs<HTMLElement>("metaLine");
const rowsBody = qs<HTMLElement>("rowsBody");
const summaryRow = qs<HTMLElement>("summaryRow");
const suiteResultEl = qs<HTMLElement>("suite-result");
const doneEl = qs<HTMLElement>("done");

function params() {
  return new URLSearchParams(location.search);
}

function setStatus(text: string) {
  statusLine.textContent = text;
}

async function send<T extends ExtensionResponse>(
  message: Record<string, unknown>,
): Promise<T> {
  const response = (await chrome.runtime.sendMessage(message)) as T;
  if (
    typeof response === "object" &&
    response !== null &&
    "kind" in response &&
    response.kind === "error"
  ) {
    throw new Error(response.message);
  }
  return response;
}

async function loadCorpus(base: string, limit = 0): Promise<CorpusImage[]> {
  const root = base.replace(/\/$/, "");
  const ai = await (await fetch(`${root}/index.json`)).json();
  const real = await (await fetch(`${root}/real-index.json`)).json();
  let images: CorpusImage[] = [
    ...ai.images.map((i: { file: string; model?: string }) => ({
      file: i.file,
      label: "ai" as const,
      model: i.model ?? null,
    })),
    ...real.images.map((i: { file: string }) => ({
      file: i.file,
      label: "real" as const,
      model: null,
    })),
  ];
  if (Number.isFinite(limit) && limit > 0) {
    const aiImgs = images.filter((i) => i.label === "ai");
    const realImgs = images.filter((i) => i.label === "real");
    const each = Math.max(1, Math.floor(limit / 2));
    images = [...aiImgs.slice(0, each), ...realImgs.slice(0, each)].slice(
      0,
      limit,
    );
  }
  return images;
}

function balancedAccuracy(c: {
  tp: number;
  tn: number;
  fp: number;
  fn: number;
}): number {
  const tpr = c.tp + c.fn === 0 ? 0 : c.tp / (c.tp + c.fn);
  const tnr = c.tn + c.fp === 0 ? 0 : c.tn / (c.tn + c.fp);
  return (tpr + tnr) / 2;
}

function renderSummary(result: SuiteResult) {
  const c = result.confusion;
  summaryRow.innerHTML = `
    <td>${(result.balancedAccuracy * 100).toFixed(1)}%</td>
    <td>${c.tp}</td>
    <td>${c.tn}</td>
    <td>${c.fp}</td>
    <td>${c.fn}</td>
    <td>${result.timing.avgTotalMs.toFixed(1)}</td>
    <td>${result.providerActual}</td>
    <td>${result.gpuAvailable ? "yes" : "no"}</td>
  `;
}

function appendRow(row: Row) {
  const tr = document.createElement("tr");
  tr.innerHTML = `
    <td class="${row.correct ? "ok" : "miss"}">${row.correct ? "ok" : "MISS"}</td>
    <td>${row.file}</td>
    <td>${row.label}</td>
    <td>${row.confidence.toFixed(3)}</td>
    <td>${row.predicted}</td>
    <td>${row.totalMs.toFixed(1)}</td>
    <td>${row.backend}</td>
    <td>${row.tiers}${row.error ? ` err=${row.error}` : ""}</td>
  `;
  rowsBody.appendChild(tr);
}

async function setupModels() {
  setStatus("Setting up models (packaged → cache, else download)…");
  const res = await send<SetupModelsResponse>({
    kind: "setup-models",
    requestId: crypto.randomUUID(),
  });
  if (res.status.kind !== "ready") {
    throw new Error(
      res.status.kind === "error"
        ? res.status.message
        : `Models not ready: ${res.status.kind}`,
    );
  }
  setStatus(`Models ready (${res.status.bytes} bytes, ${res.status.version})`);
}

async function configureRuntime(
  provider: VisualProvider["kind"],
  threshold: number,
) {
  await send({
    kind: "set-options",
    requestId: crypto.randomUUID(),
    stubInference: false,
    visualProvider: provider,
    threshold,
    autoScan: false,
  });
  const reset = await send<ResetVisualResponse>({
    kind: "reset-visual",
    requestId: crypto.randomUUID(),
    warm: true,
  });
  return reset;
}

async function analyzeOne(
  corpusBase: string,
  item: CorpusImage,
): Promise<{ result: DetectionResult; totalMs: number }> {
  // Prefer analyze-image + http(s) URL so the SW/offscreen path matches production.
  const url = `${corpusBase.replace(/\/$/, "")}/${item.file}`;
  const t0 = performance.now();
  const res = await send<AnalyzeImageResponse>({
    kind: "analyze-image",
    requestId: crypto.randomUUID(),
    imageId: item.file,
    src: url,
    width: 0,
    height: 0,
  });
  return { result: res.result, totalMs: performance.now() - t0 };
}

async function runEval() {
  doneEl.dataset.state = "running";
  doneEl.hidden = true;
  rowsBody.innerHTML = "";
  runBtn.disabled = true;
  setupBtn.disabled = true;

  try {
    const corpusBase = corpusBaseInput.value.trim();
    const provider = providerSelect.value as VisualProvider["kind"];
    const threshold = Number(thresholdInput.value);
    if (!corpusBase) throw new Error("Corpus base URL required");
    if (!Number.isFinite(threshold)) throw new Error("Bad threshold");

    await setupModels();
    const reset = await configureRuntime(provider, threshold);
    metaLine.textContent = `provider requested=${provider} actual=${reset.backend.kind} gpu=${reset.gpuAvailable}`;

    const limit = Number(params().get("limit") ?? "0");
    const images = await loadCorpus(corpusBase, limit);
    setStatus(`Running ${images.length} images…`);

    const rows: Row[] = [];
    let tp = 0;
    let tn = 0;
    let fp = 0;
    let fn = 0;
    let sum = 0;

    for (let i = 0; i < images.length; i += 1) {
      const item = images[i]!;
      setStatus(`(${i + 1}/${images.length}) ${item.file}`);
      try {
        const { result, totalMs } = await analyzeOne(corpusBase, item);
        if (result.label.kind === "error") {
          throw new Error(result.label.message);
        }
        const predicted: "ai" | "real" =
          result.confidence >= threshold ? "ai" : "real";
        const actualAi = item.label === "ai";
        const predictedAi = predicted === "ai";
        if (actualAi && predictedAi) tp += 1;
        else if (!actualAi && !predictedAi) tn += 1;
        else if (!actualAi && predictedAi) fp += 1;
        else fn += 1;
        sum += totalMs;
        const row: Row = {
          file: item.file,
          label: item.label,
          model: item.model,
          confidence: result.confidence,
          predicted,
          correct: predictedAi === actualAi,
          backend: result.backend.kind,
          totalMs,
          elapsedMs: result.elapsedMs,
          tiers: result.tiers
            .map((t) => `${t.tier}:${Number(t.aiScore).toFixed(3)}`)
            .join("|"),
        };
        rows.push(row);
        appendRow(row);
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        fn += item.label === "ai" ? 1 : 0;
        fp += item.label === "real" ? 1 : 0;
        const row: Row = {
          file: item.file,
          label: item.label,
          model: item.model,
          confidence: 0.5,
          predicted: "real",
          correct: false,
          backend: "none",
          totalMs: 0,
          elapsedMs: 0,
          tiers: "",
          error: message,
        };
        rows.push(row);
        appendRow(row);
      }
    }

    const confusion = { tp, tn, fp, fn };
    const suite: SuiteResult = {
      kind: "truepixel-browser-eval",
      generatedAt: new Date().toISOString(),
      corpusBase,
      providerRequested: provider,
      providerActual: reset.backend.kind,
      gpuAvailable: reset.gpuAvailable,
      threshold,
      balancedAccuracy: balancedAccuracy(confusion),
      confusion,
      timing: {
        avgTotalMs: rows.length ? sum / rows.length : 0,
        count: rows.length,
      },
      rows,
    };

    renderSummary(suite);
    suiteResultEl.textContent = JSON.stringify(suite);
    suiteResultEl.removeAttribute("hidden");
    doneEl.dataset.state = "done";
    doneEl.removeAttribute("hidden");
    doneEl.setAttribute("data-ba", String(suite.balancedAccuracy));
    setStatus(
      `Done BA=${(suite.balancedAccuracy * 100).toFixed(1)}% avg=${suite.timing.avgTotalMs.toFixed(1)}ms`,
    );

    const status = await send<GetStatusResponse>({
      kind: "get-status",
      requestId: crypto.randomUUID(),
    });
    metaLine.textContent = `provider requested=${provider} actual=${suite.providerActual} statusBackend=${status.backend.kind} gpu=${suite.gpuAvailable}`;
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    setStatus(`Error: ${message}`);
    doneEl.dataset.state = "error";
    doneEl.dataset.error = message;
    doneEl.removeAttribute("hidden");
    suiteResultEl.textContent = JSON.stringify({
      kind: "truepixel-browser-eval-error",
      message,
    });
    suiteResultEl.removeAttribute("hidden");
  } finally {
    runBtn.disabled = false;
    setupBtn.disabled = false;
  }
}

function init() {
  const p = params();
  corpusBaseInput.value =
    p.get("corpus") ?? "http://127.0.0.1:4173/benchmark/openrouter";
  providerSelect.value = p.get("provider") ?? "wasm";
  if (p.get("threshold")) thresholdInput.value = p.get("threshold")!;

  setupBtn.addEventListener("click", () => {
    void setupModels().catch((error: unknown) => {
      setStatus(error instanceof Error ? error.message : String(error));
    });
  });
  runBtn.addEventListener("click", () => {
    void runEval();
  });

  if (p.get("autorun") === "1") {
    void runEval();
  }
}

init();
