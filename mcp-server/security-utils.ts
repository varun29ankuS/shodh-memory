export function isLocalHostFromUrl(apiUrl: string): boolean {
  try {
    const url = new URL(apiUrl);
    const host = url.hostname;
    return host === "127.0.0.1" || host === "localhost" || host === "::1" || host === "0.0.0.0";
  } catch {
    return false;
  }
}

export function shouldWarnInsecureApiUrl(apiUrl: string, allowHttpEnv?: string): boolean {
  return !isLocalHostFromUrl(apiUrl) && apiUrl.startsWith("http://") && allowHttpEnv !== "true";
}

export function serializeAndValidateBody(body: object, maxLength: number): { ok: true; serialized: string } | { ok: false; error: string } {
  const serialized = JSON.stringify(body);
  if (serialized.length > maxLength) {
    return { ok: false, error: `Request body exceeds maximum length of ${maxLength} characters` };
  }
  return { ok: true, serialized };
}

export function nextReconnectDelay(currentDelayMs: number, maxDelayMs: number): number {
  const safeCurrent = Math.max(currentDelayMs, 1000);
  return Math.min(safeCurrent * 2, maxDelayMs);
}