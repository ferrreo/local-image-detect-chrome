import { newRequestId } from "../shared/messages";
import type {
  GetStatusResponse,
  SetupModelsResponse,
} from "../shared/types";

const modelStatusEl = document.querySelector("#modelStatus");
const backendStatusEl = document.querySelector("#backendStatus");
const thresholdStatusEl = document.querySelector("#thresholdStatus");
const messageEl = document.querySelector("#message");
const setupBtn = document.querySelector<HTMLButtonElement>("#setupBtn");
const rescanBtn = document.querySelector<HTMLButtonElement>("#rescanBtn");
const autoScanEl = document.querySelector<HTMLInputElement>("#autoScan");

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
    thresholdStatusEl.textContent = `${Math.round(response.threshold * 100)}%`;
  }
  if (autoScanEl) autoScanEl.checked = response.autoScan;

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
    await chrome.tabs.sendMessage(tab.id, { kind: "truepixel-rescan" });
    setMessage("Rescan requested.", "ok");
  })();
});

autoScanEl?.addEventListener("change", () => {
  void chrome.runtime.sendMessage({
    kind: "set-options",
    requestId: newRequestId(),
    autoScan: autoScanEl.checked,
  });
});

void refresh().catch((error: unknown) => {
  setMessage(error instanceof Error ? error.message : String(error), "error");
});
