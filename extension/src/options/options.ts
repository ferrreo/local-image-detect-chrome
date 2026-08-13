import { newRequestId } from "../shared/messages";
import type { GetStatusResponse } from "../shared/types";
import { parseAiConcealMode } from "../shared/types";

const form = document.querySelector<HTMLFormElement>("#form");
const thresholdEl = document.querySelector<HTMLInputElement>("#threshold");
const autoScanEl = document.querySelector<HTMLInputElement>("#autoScan");
const aiConcealEl = document.querySelector<HTMLSelectElement>("#aiConceal");
const debugEl = document.querySelector<HTMLInputElement>("#debug");
const savedEl = document.querySelector<HTMLElement>("#saved");

async function load(): Promise<void> {
  const response = (await chrome.runtime.sendMessage({
    kind: "get-status",
    requestId: newRequestId(),
  })) as GetStatusResponse;

  if (thresholdEl) thresholdEl.value = String(response.threshold);
  if (autoScanEl) autoScanEl.checked = response.autoScan;
  if (aiConcealEl) {
    aiConcealEl.value = parseAiConcealMode(response.aiConceal);
  }

  const stored = await chrome.storage.local.get(["options"]);
  const options = stored.options;
  if (
    typeof options === "object" &&
    options !== null &&
    "debug" in options &&
    debugEl
  ) {
    debugEl.checked = Boolean((options as { debug?: boolean }).debug);
  }
}

form?.addEventListener("submit", (event) => {
  event.preventDefault();
  void (async () => {
    const threshold = Number(thresholdEl?.value ?? 0.65);
    await chrome.runtime.sendMessage({
      kind: "set-options",
      requestId: newRequestId(),
      threshold,
      autoScan: autoScanEl?.checked ?? true,
      debug: debugEl?.checked ?? false,
      aiConceal: parseAiConcealMode(aiConcealEl?.value),
    });
    if (savedEl) {
      savedEl.hidden = false;
      window.setTimeout(() => {
        savedEl.hidden = true;
      }, 1500);
    }
  })();
});

void load();
