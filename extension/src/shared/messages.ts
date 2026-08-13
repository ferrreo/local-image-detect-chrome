import type { ExtensionRequest, ExtensionResponse } from "./types";

export function isExtensionResponse(
  value: unknown,
): value is ExtensionResponse {
  if (typeof value !== "object" || value === null) return false;
  if (!("kind" in value) || typeof value.kind !== "string") return false;
  if (!("requestId" in value) || typeof value.requestId !== "string") {
    return false;
  }
  return true;
}

export async function sendExtensionRequest<T extends ExtensionResponse>(
  request: ExtensionRequest,
): Promise<T> {
  const response: unknown = await chrome.runtime.sendMessage(request);
  if (!isExtensionResponse(response)) {
    throw new Error("Invalid extension response");
  }
  if (response.kind === "error") {
    throw new Error(response.message);
  }
  if (response.requestId !== request.requestId) {
    throw new Error("Mismatched requestId in extension response");
  }
  return response as T;
}

export function newRequestId(): string {
  if (typeof crypto !== "undefined" && "randomUUID" in crypto) {
    return crypto.randomUUID();
  }
  return `req_${Date.now()}_${Math.random().toString(16).slice(2)}`;
}
