import { createHmac, randomUUID } from "node:crypto";
import { existsSync, rmSync } from "node:fs";
import { createServer, type Server, type Socket } from "node:net";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import {
  IPC_MAX_FRAME_BYTES,
  IPC_PROTOCOL_VERSION,
  IpcApiError,
  ShodhIpcClient,
  type WindowsIpcHelper,
} from "../ipc-client";

const NONCE = Buffer.alloc(32, 0x5a);
const PROBE_RESPONSE_DOMAIN = Buffer.from("shodh-ipc-v2/server-proof", "utf8");
const REQUEST_DOMAIN = Buffer.from("shodh-ipc-v2/request", "utf8");
const RESPONSE_DOMAIN = Buffer.from("shodh-ipc-v2/response", "utf8");
const WINDOWS_TEST_HELPER = String.raw`
const fs = require("node:fs");
const endpointIndex = process.argv.indexOf("--ipc-endpoint");
const endpoint = process.argv[endpointIndex + 1];
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
  fs.writeSync(descriptor, request);
  const chunks = [];
  for (;;) {
    const chunk = Buffer.allocUnsafe(8192);
    let count;
    try { count = fs.readSync(descriptor, chunk, 0, chunk.length, null); }
    catch (error) { if (error.code === "EPIPE") break; throw error; }
    if (count === 0) break;
    chunks.push(chunk.subarray(0, count));
  }
  fs.writeFileSync(1, Buffer.concat(chunks));
} catch (error) {
  fs.writeFileSync(2, String(error && error.message ? error.message : error));
  process.exitCode = 1;
} finally {
  if (descriptor !== undefined) try { fs.closeSync(descriptor); } catch {}
}`;

const servers: Server[] = [];
const unixEndpoints: string[] = [];

afterEach(async () => {
  await Promise.all(servers.splice(0).map((server) => closeServer(server)));
  for (const endpoint of unixEndpoints.splice(0)) {
    if (existsSync(endpoint)) rmSync(endpoint, { force: true });
  }
});

describe("ShodhIpcClient", () => {
  it("authenticates the endpoint without transmitting the API key", async () => {
    let captured: RequestRecord | undefined;
    const endpoint = await listen((socket, request) => {
      captured = request;
      socket.end(encodeResponse(signedResponse(request, "secret-key", 200, { memories: [] })));
    }, "secret-key");

    const client = makeClient(endpoint, "secret-key");
    await expect(client.request("/api/recall", "POST", { query: "hello" }, 1_000))
      .resolves.toEqual({ memories: [] });
    expect(captured).toMatchObject({
      v: IPC_PROTOCOL_VERSION,
      challenge: "",
      method: "POST",
      path: "/api/recall",
      body: { query: "hello" },
    });
    expect(captured?.auth).toMatch(/^request:[0-9a-f]{64}:[0-9a-f]{64}$/);
    expect(requestProofIsValid(captured!, "secret-key")).toBe(true);
    expect(JSON.stringify(captured)).not.toContain("secret-key");
    expect(captured?.id).toMatch(/^[0-9a-f-]{36}$/);
  });

  it("maps signed non-success statuses to API-compatible errors", async () => {
    const endpoint = await listen((socket, request) => {
      socket.end(encodeResponse(signedResponse(
        request,
        "wrong-key",
        401,
        { code: "INVALID_API_KEY", message: "Invalid API key" },
      )));
    }, "wrong-key");

    const client = makeClient(endpoint, "wrong-key");
    const error = await client.request("/api/users", "GET", null, 1_000).catch((cause) => cause);
    expect(error).toBeInstanceOf(IpcApiError);
    expect(error).toMatchObject({ status: 401 });
    expect((error as Error).message).toContain("INVALID_API_KEY");
  });

  it("rejects a health response without a valid server proof", async () => {
    const endpoint = await listen((socket, request) => {
      socket.end(encodeResponse({
        v: IPC_PROTOCOL_VERSION,
        id: request.id,
        auth: "",
        status: 200,
        body: { ok: true },
      }));
    }, "key", false);

    await expect(makeClient(endpoint, "key").request("/health", "GET", null, 1_000))
      .rejects.toThrow("did not authenticate the server");
  });

  it("rejects a response with the wrong protocol version", async () => {
    const endpoint = await listen((socket, request) => {
      socket.end(encodeResponse({
        v: IPC_PROTOCOL_VERSION + 1,
        id: request.id,
        auth: "",
        status: 200,
        body: {},
      }));
    }, "key", false);

    await expect(makeClient(endpoint, "key").request("/health", "GET", null, 1_000))
      .rejects.toThrow("Unsupported Shodh IPC response version");
  });

  it("rejects a response whose request ID does not match", async () => {
    const endpoint = await listen((socket) => {
      socket.end(encodeResponse({
        v: IPC_PROTOCOL_VERSION,
        id: randomUUID(),
        auth: "",
        status: 200,
        body: {},
      }));
    }, "key", false);

    await expect(makeClient(endpoint, "key").request("/health", "GET", null, 1_000))
      .rejects.toThrow("request ID did not match");
  });

  it("rejects fields outside the strict version two response envelope", async () => {
    const endpoint = await listen((socket, request) => {
      socket.end(JSON.stringify({
        v: IPC_PROTOCOL_VERSION,
        id: request.id,
        auth: "",
        status: 200,
        body: {},
        extra: true,
      }) + "\n");
    }, "key", false);

    await expect(makeClient(endpoint, "key").request("/health", "GET", null, 1_000))
      .rejects.toThrow("fields did not match");
  });

  it("waits for close and rejects a second response frame", async () => {
    const endpoint = await listen((socket, request) => {
      socket.write(encodeResponse(signedProbeResponse(request, "key")));
      setTimeout(() => socket.end("{}\n"), 10);
    }, "key", false);

    await expect(makeClient(endpoint, "key").request("/health", "GET", null, 1_000))
      .rejects.toThrow("data after");
  });

  it("rejects invalid UTF-8 responses", async () => {
    const endpoint = await listen((socket) => {
      socket.end(Buffer.from([0xc3, 0x28, 0x0a]));
    }, "key", false);

    await expect(makeClient(endpoint, "key").request("/health", "GET", null, 1_000))
      .rejects.toThrow("not valid UTF-8");
  });

  it("rejects response frames over eight MiB", async () => {
    const endpoint = await listen((socket) => {
      socket.end(Buffer.alloc(IPC_MAX_FRAME_BYTES + 1, 0x20));
    }, "key", false);

    await expect(makeClient(endpoint, "key").request("/health", "GET", null, 2_000))
      .rejects.toThrow("response exceeds");
  });

  it("enforces one total connect and probe deadline", async () => {
    const endpoint = await listen(() => {
      // Keep the accepted connection open without replying.
    }, "key", false);

    await expect(makeClient(endpoint, "key").request("/health", "GET", null, 30))
      .rejects.toThrow("timed out");
  });

  it("rejects oversized requests before opening a connection", async () => {
    const client = makeClient(testEndpoint(), "key");
    const oversized = "x".repeat(IPC_MAX_FRAME_BYTES);
    await expect(client.request("/api/remember", "POST", { content: oversized }, 1_000))
      .rejects.toThrow("request exceeds");
  });
});

interface RequestRecord {
  v: number;
  id: string;
  auth: string;
  challenge: string;
  method: string;
  path: string;
  body: unknown;
}

interface ResponseRecord {
  v: number;
  id: string;
  auth: string;
  status: number;
  body: unknown;
}

async function listen(
  onRequest: (socket: Socket, request: RequestRecord) => void,
  apiKey: string,
  autoProbe = true,
): Promise<string> {
  const endpoint = testEndpoint();
  const server = createServer((socket) => {
    let frame = Buffer.alloc(0);
    socket.on("data", (chunk: Buffer) => {
      frame = Buffer.concat([frame, chunk]);
      const newline = frame.indexOf(0x0a);
      if (newline === -1) return;
      const request = JSON.parse(frame.subarray(0, newline).toString("utf8")) as RequestRecord;
      if (autoProbe && request.method === "GET" && request.path === "/health") {
        socket.end(encodeResponse(signedProbeResponse(request, apiKey)));
      } else {
        onRequest(socket, request);
      }
    });
  });
  servers.push(server);
  if (process.platform !== "win32") unixEndpoints.push(endpoint);
  await new Promise<void>((resolve, reject) => {
    server.once("error", reject);
    server.listen(endpoint, resolve);
  });
  return endpoint;
}

function signedProbeResponse(request: RequestRecord, key: string): ResponseRecord {
  const status = 200;
  const proof = macHex(key, PROBE_RESPONSE_DOMAIN, [
    Buffer.from(request.id, "utf8"),
    Buffer.from(request.challenge, "utf8"),
    NONCE,
    u16(status),
  ]);
  return {
    v: IPC_PROTOCOL_VERSION,
    id: request.id,
    auth: `probe:${NONCE.toString("hex")}:${proof}`,
    status,
    body: { ok: true },
  };
}

function signedResponse(
  request: RequestRecord,
  key: string,
  status: number,
  body: unknown,
): ResponseRecord {
  const proof = macHex(key, RESPONSE_DOMAIN, [
    NONCE,
    Buffer.from(request.auth, "utf8"),
    Buffer.from(request.id, "utf8"),
    u16(status),
  ]);
  return {
    v: IPC_PROTOCOL_VERSION,
    id: request.id,
    auth: `response:${proof}`,
    status,
    body,
  };
}

function requestProofIsValid(request: RequestRecord, key: string): boolean {
  const [kind, nonceHex, proof, ...trailing] = request.auth.split(":");
  if (kind !== "request" || !nonceHex || !proof || trailing.length > 0) return false;
  const version = u16(request.v);
  const expected = macHex(key, REQUEST_DOMAIN, [
    Buffer.from(nonceHex, "hex"),
    version,
    Buffer.from(request.id, "utf8"),
    Buffer.from(request.method, "utf8"),
    Buffer.from(request.path, "utf8"),
    Buffer.from(JSON.stringify(request.body), "utf8"),
  ]);
  return nonceHex === NONCE.toString("hex") && proof === expected;
}

function macHex(key: string, domain: Buffer, fields: Buffer[]): string {
  const mac = createHmac("sha256", key);
  for (const field of [domain, ...fields]) {
    const length = Buffer.alloc(8);
    length.writeBigUInt64BE(BigInt(field.byteLength));
    mac.update(length);
    mac.update(field);
  }
  return mac.digest("hex");
}

function u16(value: number): Buffer {
  const encoded = Buffer.alloc(2);
  encoded.writeUInt16BE(value);
  return encoded;
}

function encodeResponse(response: ResponseRecord): string {
  return `${JSON.stringify(response)}\n`;
}

function makeClient(endpoint: string, key: string): ShodhIpcClient {
  return new ShodhIpcClient(endpoint, key, windowsTestHelper());
}

function windowsTestHelper(): WindowsIpcHelper | undefined {
  return process.platform === "win32"
    ? { command: process.execPath, args: ["-e", WINDOWS_TEST_HELPER, "--"] }
    : undefined;
}

function testEndpoint(): string {
  const suffix = `${process.pid}-${randomUUID()}`;
  return process.platform === "win32"
    ? `\\\\.\\pipe\\shodh-ipc-test-${suffix}`
    : join(tmpdir(), `shodh-ipc-test-${suffix}.sock`);
}

function closeServer(server: Server): Promise<void> {
  return new Promise((resolve) => {
    server.close(() => resolve());
  });
}
