import { newRequestId } from "../shared/messages";
import type {
  AiConcealMode,
  GetStatusResponse,
  SetupModelsResponse,
} from "../shared/types";
import { parseAiConcealMode } from "../shared/types";

const modelStatusEl = document.querySelector("#modelStatus");
const backendStatusEl = document.querySelector("#backendStatus");
const thresholdStatusEl = document.querySelector("#thresholdStatus");
const messageEl = document.querySelector("#message");
const setupBtn = document.querySelector<HTMLButtonElement>("#setupBtn");
const rescanBtn = document.querySelector<HTMLButtonElement>("#rescanBtn");
const autoScanEl = document.querySelector<HTMLInputElement>("#autoScan");
const aiConcealEl = document.querySelector<HTMLSelectElement>("#aiConceal");
const extVersionEl = document.querySelector("#extVersion");

if (extVersionEl instanceof HTMLElement) {
  const manifest = chrome.runtime.getManifest();
  extVersionEl.textContent = `v${manifest.version}`;
}

function setMessage(text: string, kind: "ok" | "error" | "" = ""): void {
  if (!(messageEl instanceof HTMLElement)) return;
  messageEl.textContent = text;
  messageEl.classList.remove("ok", "error");
  if (kind) messageEl.classList.add(kind);
}

function formatModels(status: GetStatusResponse["models"]): string {
  switch (status.kind) {
    case "ready":
      return `ready (${status.version})`;
    case "missing":
      return "not downloaded";
    case "downloading":
      return `downloading ${Math.round(status.progress * 100)}%`;
    case "error":
      return `error: ${status.message}`;
    default: {
      const _exhaustive: never = status;
      return String(_exhaustive);
    }
  }
}

async function refresh(): Promise<void> {
  const response = (await chrome.runtime.sendMessage({
    kind: "get-status",
    requestId: newRequestId(),
  })) as GetStatusResponse;

  if (modelStatusEl) modelStatusEl.textContent = formatModels(response.models);
  if (backendStatusEl) backendStatusEl.textContent = response.backend.kind;
  if (thresholdStatusEl) {
    const aiPct = (response.threshold * 100).toFixed(2);
    const realPct = ((response.realThreshold ?? 0.4099) * 100).toFixed(2);
    thresholdStatusEl.textContent = `AI ≥${aiPct}% · Real ≤${realPct}%`;
  }
  if (autoScanEl) autoScanEl.checked = response.autoScan;
  if (aiConcealEl) {
    aiConcealEl.value = parseAiConcealMode(response.aiConceal);
  }

  if (setupBtn) {
    setupBtn.disabled = response.models.kind === "ready";
    setupBtn.textContent =
      response.models.kind === "ready" ? "Models ready" : "Download models";
  }
}

setupBtn?.addEventListener("click", () => {
  void (async () => {
    if (setupBtn) setupBtn.disabled = true;
    setMessage("Downloading model weights (one-time)…");
    const response = (await chrome.runtime.sendMessage({
      kind: "setup-models",
      requestId: newRequestId(),
    })) as SetupModelsResponse;
    if (response.status.kind === "ready") {
      setMessage("Models installed. Scanning stays on-device.", "ok");
    } else if (response.status.kind === "error") {
      setMessage(response.status.message, "error");
    }
    await refresh();
  })();
});

rescanBtn?.addEventListener("click", () => {
  void (async () => {
    const [tab] = await chrome.tabs.query({
      active: true,
      currentWindow: true,
    });
    if (tab?.id === undefined) {
      setMessage("No active tab.", "error");
      return;
    }
    const url = tab.url ?? "";
    if (
      url.startsWith("chrome://") ||
      url.startsWith("chrome-extension://") ||
      url.startsWith("edge://") ||
      url.startsWith("about:") ||
      url.startsWith("devtools://")
    ) {
      setMessage("Can't scan this page type (open a normal website).", "error");
      return;
    }
    try {
      await chrome.tabs.sendMessage(tab.id, { kind: "neopixel-rescan" });
      setMessage("Rescan requested.", "ok");
    } catch {
      setMessage(
        "No scanner on this tab — reload the page after installing/updating.",
        "error",
      );
    }
  })();
});

autoScanEl?.addEventListener("change", () => {
  void chrome.runtime.sendMessage({
    kind: "set-options",
    requestId: newRequestId(),
    autoScan: autoScanEl.checked,
  });
});

aiConcealEl?.addEventListener("change", () => {
  const aiConceal: AiConcealMode = parseAiConcealMode(aiConcealEl.value);
  void chrome.runtime.sendMessage({
    kind: "set-options",
    requestId: newRequestId(),
    aiConceal,
  });
  setMessage(
    aiConceal === "none"
      ? "AI images show badge only."
      : `AI images will ${aiConceal}. Click badge to reveal.`,
    "ok",
  );
});

void refresh().catch((error: unknown) => {
  setMessage(error instanceof Error ? error.message : String(error), "error");
});
