import { createHmac, randomBytes, randomUUID, timingSafeEqual } from "node:crypto";
import { spawn, type ChildProcessWithoutNullStreams } from "node:child_process";
import { createConnection, type Socket } from "node:net";
import { TextDecoder } from "node:util";

export const IPC_PROTOCOL_VERSION = 2;
export const IPC_MAX_FRAME_BYTES = 8 * 1024 * 1024;

const RESPONSE_FIELDS = ["auth", "body", "id", "status", "v"];
const utf8Decoder = new TextDecoder("utf-8", { fatal: true });
const WINDOWS_HELPER_LIMIT = 4;
const NONCE_BYTES = 32;
const PROBE_RESPONSE_DOMAIN = Buffer.from("shodh-ipc-v2/server-proof", "utf8");
const REQUEST_DOMAIN = Buffer.from("shodh-ipc-v2/request", "utf8");
const RESPONSE_DOMAIN = Buffer.from("shodh-ipc-v2/response", "utf8");
let activeWindowsHelpers = 0;
const windowsHelperWaiters: Array<() => void> = [];

export interface WindowsIpcHelper {
  command: string;
  args: string[];
}

interface IpcRequestEnvelope {
  v: number;
  id: string;
  auth: string;
  challenge: string;
  method: string;
  path: string;
  body: unknown;
}

interface IpcResponseEnvelope {
  v: number;
  id: string;
  auth: string;
  status: number;
  body: unknown;
}

export class IpcApiError extends Error {
  readonly status: number;
  readonly body: unknown;

  constructor(status: number, body: unknown) {
    super(`API error ${status}: ${formatBody(body)}`);
    this.name = "IpcApiError";
    this.status = status;
    this.body = body;
  }
}

/** One-request-per-connection client for Shodh's authenticated local IPC protocol. */
export class ShodhIpcClient {
  readonly endpoint: string;
  readonly #apiKey: string;
  readonly #windowsHelper?: WindowsIpcHelper;
  #serverNonce: Buffer | null = null;

  constructor(endpoint: string, apiKey: string, windowsHelper?: WindowsIpcHelper) {
    if (!endpoint) {
      throw new Error("Shodh IPC endpoint must not be empty");
    }
    // A `\\.\pipe\` endpoint is a Windows named pipe. On any other platform it is
    // a single-component relative path that createConnection cannot open; fail loudly
    // here rather than forwarding a bogus path to an auto-spawned server that then
    // dies silently. (This catches the cross-platform quickstart being copied as-is.)
    if (process.platform !== "win32" && endpoint.startsWith("\\\\.\\pipe\\")) {
      throw new Error(
        `Shodh IPC endpoint "${endpoint}" is a Windows named pipe, but this platform ` +
          `is ${process.platform}. Set SHODH_IPC_ENDPOINT to a Unix socket path.`,
      );
    }
    if (process.platform === "win32" && !windowsHelper) {
      throw new Error("Shodh IPC on Windows requires the bundled Rust pipe helper");
    }
    this.endpoint = endpoint;
    this.#apiKey = apiKey;
    this.#windowsHelper = windowsHelper;
  }

  async request<T>(
    path: string,
    method = "GET",
    body: unknown = null,
    timeoutMs = 10_000,
  ): Promise<T> {
    if (!Number.isFinite(timeoutMs) || timeoutMs <= 0) {
      throw new Error("Shodh IPC timeout must be a positive number");
    }

    const expiresAt = Date.now() + timeoutMs;
    if (method === "GET" && path === "/health") {
      return await this.#probe(expiresAt) as T;
    }
    const normalizedBody = normalizeBody(body ?? null);
    const bodyJson = JSON.stringify(normalizedBody);
    const id = randomUUID();
    const sizingNonce = this.#serverNonce ?? Buffer.alloc(NONCE_BYTES);
    const sizingAuth = requestAuth(
      this.#apiKey,
      sizingNonce,
      id,
      method,
      path,
      Buffer.from(bodyJson, "utf8"),
    );
    encodeRequest({
      v: IPC_PROTOCOL_VERSION,
      id,
      auth: sizingAuth,
      challenge: "",
      method,
      path,
      body: normalizedBody,
    });
    if (!this.#serverNonce) {
      await this.#probe(expiresAt);
    }

    const auth = requestAuth(
      this.#apiKey,
      this.#serverNonce!,
      id,
      method,
      path,
      Buffer.from(bodyJson, "utf8"),
    );
    const request: IpcRequestEnvelope = {
      v: IPC_PROTOCOL_VERSION,
      id,
      auth,
      challenge: "",
      method,
      path,
      body: normalizedBody,
    };
    const encoded = encodeRequest(request);

    const response = await exchange(
      this.endpoint,
      encoded,
      remainingMs(expiresAt),
      this.#windowsHelper,
    );
    const envelope = decodeResponse(response, id);
    verifyResponse(this.#apiKey, this.#serverNonce!, auth, envelope);
    if (envelope.status < 200 || envelope.status >= 300) {
      throw new IpcApiError(envelope.status, envelope.body);
    }
    return envelope.body as T;
  }

  async #probe(expiresAt: number): Promise<unknown> {
    const id = randomUUID();
    const challenge = randomBytes(NONCE_BYTES).toString("hex");
    const request: IpcRequestEnvelope = {
      v: IPC_PROTOCOL_VERSION,
      id,
      auth: "",
      challenge,
      method: "GET",
      path: "/health",
      body: null,
    };
    const response = await exchange(
      this.endpoint,
      encodeRequest(request),
      remainingMs(expiresAt),
      this.#windowsHelper,
    );
    const envelope = decodeResponse(response, id);
    this.#serverNonce = verifyProbeResponse(this.#apiKey, challenge, envelope);
    if (envelope.status < 200 || envelope.status >= 300) {
      throw new IpcApiError(envelope.status, envelope.body);
    }
    return envelope.body;
  }
}

function normalizeBody(body: unknown): unknown {
  try {
    const encoded = JSON.stringify(body);
    if (encoded === undefined) return null;
    return JSON.parse(encoded) as unknown;
  } catch (error) {
    throw new Error(`Failed to encode Shodh IPC request body: ${errorMessage(error)}`);
  }
}

function remainingMs(expiresAt: number): number {
  const remaining = expiresAt - Date.now();
  if (remaining <= 0) {
    throw new Error("Shodh IPC request timed out before the exchange completed");
  }
  return remaining;
}

function requestAuth(
  key: string,
  nonce: Buffer,
  id: string,
  method: string,
  path: string,
  body: Buffer,
): string {
  const version = Buffer.alloc(2);
  version.writeUInt16BE(IPC_PROTOCOL_VERSION);
  const proof = macHex(key, REQUEST_DOMAIN, [
    nonce,
    version,
    Buffer.from(id, "utf8"),
    Buffer.from(method, "utf8"),
    Buffer.from(path, "utf8"),
    body,
  ]);
  return `request:${nonce.toString("hex")}:${proof}`;
}

function verifyProbeResponse(
  key: string,
  challenge: string,
  response: IpcResponseEnvelope,
): Buffer {
  const [kind, nonceHex, proofs, ...trailing] = response.auth.split(":");
  if (kind !== "probe" || !nonceHex || proofs === undefined || trailing.length > 0) {
    throw new Error("Shodh IPC health response did not authenticate the server");
  }
  const nonce = decodeNonce(nonceHex, "server nonce");
  const status = Buffer.alloc(2);
  status.writeUInt16BE(response.status);
  const expected = macHex(key, PROBE_RESPONSE_DOMAIN, [
    Buffer.from(response.id, "utf8"),
    Buffer.from(challenge, "utf8"),
    nonce,
    status,
  ]);
  if (!proofs.split(",").some((proof) => constantTimeHexEqual(proof, expected))) {
    throw new Error("Shodh IPC health response was not signed by the configured server");
  }
  return nonce;
}

function verifyResponse(
  key: string,
  nonce: Buffer,
  requestProof: string,
  response: IpcResponseEnvelope,
): void {
  const proof = response.auth.startsWith("response:")
    ? response.auth.slice("response:".length)
    : "";
  const status = Buffer.alloc(2);
  status.writeUInt16BE(response.status);
  const valid = verifyMac(key, RESPONSE_DOMAIN, [
    nonce,
    Buffer.from(requestProof, "utf8"),
    Buffer.from(response.id, "utf8"),
    status,
  ], proof);
  if (!valid) {
    throw new Error("Shodh IPC response server proof was invalid");
  }
}

function macHex(key: string, domain: Buffer, fields: Buffer[]): string {
  const mac = createHmac("sha256", key);
  updateMacField(mac, domain);
  for (const field of fields) updateMacField(mac, field);
  return mac.digest("hex");
}

function verifyMac(key: string, domain: Buffer, fields: Buffer[], proof: string): boolean {
  const expected = macHex(key, domain, fields);
  return constantTimeHexEqual(proof, expected);
}

function updateMacField(mac: ReturnType<typeof createHmac>, field: Buffer): void {
  const length = Buffer.alloc(8);
  length.writeBigUInt64BE(BigInt(field.byteLength));
  mac.update(length);
  mac.update(field);
}

function constantTimeHexEqual(actual: string, expected: string): boolean {
  if (!/^[0-9a-f]+$/i.test(actual) || actual.length !== expected.length) return false;
  return timingSafeEqual(Buffer.from(actual, "hex"), Buffer.from(expected, "hex"));
}

function decodeNonce(encoded: string, label: string): Buffer {
  if (!/^[0-9a-f]+$/i.test(encoded) || encoded.length !== NONCE_BYTES * 2) {
    throw new Error(`Shodh IPC response carried an invalid ${label}`);
  }
  return Buffer.from(encoded, "hex");
}

function encodeRequest(request: IpcRequestEnvelope): Buffer {
  let json: string;
  try {
    json = JSON.stringify(request);
  } catch (error) {
    throw new Error(`Failed to encode Shodh IPC request: ${errorMessage(error)}`);
  }

  const encoded = Buffer.from(`${json}\n`, "utf8");
  if (encoded.byteLength > IPC_MAX_FRAME_BYTES) {
    throw new Error(`Shodh IPC request exceeds the ${IPC_MAX_FRAME_BYTES}-byte frame limit`);
  }
  return encoded;
}

function exchange(
  endpoint: string,
  encoded: Buffer,
  timeoutMs: number,
  windowsHelper?: WindowsIpcHelper,
): Promise<Buffer> {
  return process.platform === "win32"
    ? exchangeWindowsPipe(endpoint, encoded, timeoutMs, windowsHelper!)
    : exchangeUnixSocket(endpoint, encoded, timeoutMs);
}

function exchangeWindowsPipe(
  endpoint: string,
  encoded: Buffer,
  timeoutMs: number,
  windowsHelper: WindowsIpcHelper,
): Promise<Buffer> {
  return new Promise((resolve, reject) => {
    const expiresAt = Date.now() + timeoutMs;
    let settled = false;
    let response = Buffer.alloc(0);
    let diagnostics = Buffer.alloc(0);
    let helper: ChildProcessWithoutNullStreams | null = null;
    let releaseHelper: (() => void) | null = null;

    const finish = (error?: Error) => {
      if (settled) return;
      settled = true;
      clearTimeout(deadline);
      if (helper && helper.exitCode === null && helper.signalCode === null) helper.kill();
      if (error) {
        reject(error);
      } else {
        const newline = response.indexOf(0x0a);
        if (newline === -1) {
          reject(new Error("Shodh IPC connection closed before a complete response frame arrived"));
        } else if (newline !== response.byteLength - 1) {
          reject(new Error("Shodh IPC response contained data after its newline-delimited frame"));
        } else {
          resolve(response.subarray(0, newline));
        }
      }
    };

    const deadline = setTimeout(() => {
      finish(new Error(`Shodh IPC request timed out after ${timeoutMs} ms`));
    }, timeoutMs);

    acquireWindowsHelper().then((release) => {
      releaseHelper = () => {
        if (!releaseHelper) return;
        releaseHelper = null;
        release();
      };
      if (settled || Date.now() >= expiresAt) {
        releaseHelper();
        finish(new Error(`Shodh IPC request timed out after ${timeoutMs} ms`));
        return;
      }
      let runningHelper: ChildProcessWithoutNullStreams;
      try {
        // The bundled Rust helper opens with SECURITY_IDENTIFICATION and verifies
        // the server account before it writes the authenticated request proof.
        runningHelper = spawn(windowsHelper.command, [
          ...windowsHelper.args,
          "--ipc-endpoint",
          endpoint,
          "--timeout-ms",
          String(Math.max(1, Math.ceil(expiresAt - Date.now()))),
        ], {
          stdio: ["pipe", "pipe", "pipe"],
          windowsHide: true,
        }) as ChildProcessWithoutNullStreams;
        helper = runningHelper;
      } catch (error) {
        releaseHelper();
        finish(new Error(`Failed to start Shodh IPC helper: ${errorMessage(error)}`));
        return;
      }

      runningHelper.stdout.on("data", (chunk: Buffer) => {
        if (settled) return;
        if (response.byteLength + chunk.byteLength > IPC_MAX_FRAME_BYTES) {
          finish(new Error(`Shodh IPC response exceeds the ${IPC_MAX_FRAME_BYTES}-byte frame limit`));
        } else {
          response = Buffer.concat([response, chunk]);
        }
      });
      runningHelper.stderr.on("data", (chunk: Buffer) => {
        if (diagnostics.byteLength < 4096) {
          diagnostics = Buffer.concat([diagnostics, chunk.subarray(0, 4096 - diagnostics.byteLength)]);
        }
      });
      runningHelper.once("error", (error) => {
        releaseHelper?.();
        finish(new Error(`Failed to start Shodh IPC helper: ${error.message}`));
      });
      runningHelper.once("close", (code) => {
        releaseHelper?.();
        if (settled) return;
        if (code !== 0) {
          const detail = diagnostics.toString("utf8").trim();
          if (detail === "FRAME_TOO_LARGE") {
            finish(new Error(`Shodh IPC response exceeds the ${IPC_MAX_FRAME_BYTES}-byte frame limit`));
          } else {
            finish(new Error(`Shodh IPC helper failed${detail ? `: ${detail}` : ""}`));
          }
        } else {
          finish();
        }
      });

      runningHelper.stdin.on("error", (error) => {
        finish(new Error(`Failed to write Shodh IPC request: ${error.message}`));
      });
      runningHelper.stdin.end(encoded);
    });
  });
}

function acquireWindowsHelper(): Promise<() => void> {
  return new Promise((resolve) => {
    const acquire = () => {
      activeWindowsHelpers += 1;
      let released = false;
      resolve(() => {
        if (released) return;
        released = true;
        activeWindowsHelpers -= 1;
        windowsHelperWaiters.shift()?.();
      });
    };
    if (activeWindowsHelpers < WINDOWS_HELPER_LIMIT) acquire();
    else windowsHelperWaiters.push(acquire);
  });
}

function exchangeUnixSocket(
  endpoint: string,
  encoded: Buffer,
  timeoutMs: number,
): Promise<Buffer> {
  return new Promise((resolve, reject) => {
    let socket: Socket | null = null;
    let settled = false;
    let response = Buffer.alloc(0);
    const controller = new AbortController();

    const finish = (error?: Error, frame?: Buffer) => {
      if (settled) return;
      settled = true;
      clearTimeout(deadline);
      controller.abort();
      socket?.destroy();
      if (error) reject(error);
      else resolve(frame!);
    };

    const deadline = setTimeout(() => {
      finish(new Error(`Shodh IPC request timed out after ${timeoutMs} ms`));
    }, timeoutMs);

    try {
      socket = createConnection({ path: endpoint, signal: controller.signal });
      socket.on("data", (chunk: Buffer) => {
        if (settled) return;
        if (response.byteLength + chunk.byteLength > IPC_MAX_FRAME_BYTES) {
          finish(new Error(`Shodh IPC response exceeds the ${IPC_MAX_FRAME_BYTES}-byte frame limit`));
          return;
        }
        response = Buffer.concat([response, chunk]);
        const newline = response.indexOf(0x0a);
        if (newline !== -1 && newline !== response.byteLength - 1) {
          finish(new Error("Shodh IPC response contained data after its newline-delimited frame"));
        }
      });
      socket.once("connect", () => {
        socket?.write(encoded, (error) => {
          if (error) finish(new Error(`Failed to write Shodh IPC request: ${error.message}`));
        });
      });
      socket.once("end", () => {
        if (settled) return;
        const newline = response.indexOf(0x0a);
        if (newline === -1) {
          finish(new Error("Shodh IPC connection closed before a complete response frame arrived"));
        } else {
          finish(undefined, response.subarray(0, newline));
        }
      });
      socket.once("error", (error) => {
        finish(new Error(`Shodh IPC connection failed: ${error.message}`));
      });
    } catch (error) {
      finish(new Error(`Failed to connect to Shodh IPC: ${errorMessage(error)}`));
    }
  });
}

function decodeResponse(frame: Buffer, expectedId: string): IpcResponseEnvelope {
  let text: string;
  try {
    text = utf8Decoder.decode(frame);
  } catch {
    throw new Error("Shodh IPC response was not valid UTF-8");
  }

  let value: unknown;
  try {
    value = JSON.parse(text);
  } catch (error) {
    throw new Error(`Shodh IPC response was not valid JSON: ${errorMessage(error)}`);
  }

  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new Error("Shodh IPC response must be a JSON object");
  }
  const record = value as Record<string, unknown>;
  const fields = Object.keys(record).sort();
  if (fields.length !== RESPONSE_FIELDS.length || fields.some((field, index) => field !== RESPONSE_FIELDS[index])) {
    throw new Error("Shodh IPC response fields did not match the version two envelope");
  }
  if (record.v !== IPC_PROTOCOL_VERSION) {
    throw new Error(`Unsupported Shodh IPC response version ${String(record.v)}`);
  }
  // An empty id is how the server reports a protocol error it hit before (or while)
  // parsing the request envelope — it has no correlation id to echo. Treat it as a
  // server error to surface via the status check below, NOT as an id mismatch, so
  // the real status/message reaches the caller instead of being masked.
  if (typeof record.id !== "string" || (record.id !== "" && record.id !== expectedId)) {
    throw new Error("Shodh IPC response request ID did not match");
  }
  if (typeof record.auth !== "string") {
    throw new Error("Shodh IPC response authentication proof was invalid");
  }
  if (!Number.isInteger(record.status) || (record.status as number) < 100 || (record.status as number) > 599) {
    throw new Error("Shodh IPC response status was invalid");
  }

  return record as unknown as IpcResponseEnvelope;
}

function formatBody(body: unknown): string {
  if (typeof body === "string") return body;
  try {
    return JSON.stringify(body) ?? "null";
  } catch {
    return "unreadable error response";
  }
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
