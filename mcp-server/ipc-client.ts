import { randomUUID } from "node:crypto";
import { spawn, type ChildProcessWithoutNullStreams } from "node:child_process";
import { createConnection, type Socket } from "node:net";
import { TextDecoder } from "node:util";

export const IPC_PROTOCOL_VERSION = 1;
export const IPC_MAX_FRAME_BYTES = 8 * 1024 * 1024;

const RESPONSE_FIELDS = ["body", "id", "status", "v"];
const utf8Decoder = new TextDecoder("utf-8", { fatal: true });
const WINDOWS_HELPER_LIMIT = 4;
let activeWindowsHelpers = 0;
const windowsHelperWaiters: Array<() => void> = [];
const WINDOWS_PIPE_HELPER = String.raw`
const fs = require("node:fs");
const endpoint = process.argv[1];
const request = fs.readFileSync(0);
let descriptor;
try {
  for (;;) {
    try {
      descriptor = fs.openSync(endpoint, fs.constants.O_RDWR);
      break;
    } catch (error) {
      if (error.code !== "EBUSY" && error.code !== "EAGAIN") throw error;
      Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, 10);
    }
  }
  let offset = 0;
  while (offset < request.length) {
    const written = fs.writeSync(descriptor, request, offset, request.length - offset);
    if (written === 0) throw new Error("zero-byte write");
    offset += written;
  }
  const chunks = [];
  let total = 0;
  for (;;) {
    const chunk = Buffer.allocUnsafe(8192);
    let count;
    try {
      count = fs.readSync(descriptor, chunk, 0, chunk.length, null);
    } catch (error) {
      if (error.code === "EPIPE") break;
      throw error;
    }
    if (count === 0) break;
    total += count;
    if (total > ${IPC_MAX_FRAME_BYTES}) throw new Error("FRAME_TOO_LARGE");
    chunks.push(chunk.subarray(0, count));
  }
  fs.writeFileSync(1, Buffer.concat(chunks, total));
} catch (error) {
  fs.writeFileSync(2, String(error && error.message ? error.message : error));
  process.exitCode = 1;
} finally {
  if (descriptor !== undefined) {
    try { fs.closeSync(descriptor); } catch {}
  }
}`;

interface IpcRequestEnvelope {
  v: number;
  id: string;
  auth: string;
  method: string;
  path: string;
  body: unknown;
}

interface IpcResponseEnvelope {
  v: number;
  id: string;
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

  constructor(endpoint: string, apiKey: string) {
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
    this.endpoint = endpoint;
    this.#apiKey = apiKey;
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

    const id = randomUUID();
    const request: IpcRequestEnvelope = {
      v: IPC_PROTOCOL_VERSION,
      id,
      auth: this.#apiKey,
      method,
      path,
      body: body ?? null,
    };
    const encoded = encodeRequest(request);

    const response = await exchange(this.endpoint, encoded, timeoutMs);
    const envelope = decodeResponse(response, id);
    if (envelope.status < 200 || envelope.status >= 300) {
      throw new IpcApiError(envelope.status, envelope.body);
    }
    return envelope.body as T;
  }
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

function exchange(endpoint: string, encoded: Buffer, timeoutMs: number): Promise<Buffer> {
  return process.platform === "win32"
    ? exchangeWindowsPipe(endpoint, encoded, timeoutMs)
    : exchangeUnixSocket(endpoint, encoded, timeoutMs);
}

function exchangeWindowsPipe(
  endpoint: string,
  encoded: Buffer,
  timeoutMs: number,
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
        // Node's named-pipe connector enters an uncancellable 30-second
        // WaitNamedPipe. Isolating synchronous pipe I/O in a killable, bounded
        // helper keeps the public deadline real without a native addon.
        runningHelper = spawn(process.execPath, ["-e", WINDOWS_PIPE_HELPER, endpoint], {
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
    throw new Error("Shodh IPC response fields did not match the version one envelope");
  }
  if (record.v !== IPC_PROTOCOL_VERSION) {
    throw new Error(`Unsupported Shodh IPC response version ${String(record.v)}`);
  }
  // An empty id is how the server reports a protocol error it hit before (or while)
  // parsing the request envelope — it has no correlation id to echo. Treat it as a
  // server error to surface via the status check below, NOT as an id mismatch, so
  // the real status/message reaches the caller instead of being masked.
  if (record.id !== "" && record.id !== expectedId) {
    throw new Error("Shodh IPC response request ID did not match");
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
