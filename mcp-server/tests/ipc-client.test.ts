import { randomUUID } from "node:crypto";
import { existsSync, rmSync } from "node:fs";
import { createServer, type Server, type Socket } from "node:net";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import {
  IPC_MAX_FRAME_BYTES,
  IpcApiError,
  ShodhIpcClient,
} from "../ipc-client";

const servers: Server[] = [];
const unixEndpoints: string[] = [];

afterEach(async () => {
  await Promise.all(servers.splice(0).map((server) => closeServer(server)));
  for (const endpoint of unixEndpoints.splice(0)) {
    if (existsSync(endpoint)) rmSync(endpoint, { force: true });
  }
});

describe("ShodhIpcClient", () => {
  it("sends the frozen request envelope and accepts a matching response", async () => {
    let captured: Record<string, unknown> | undefined;
    const endpoint = await listen((socket, line) => {
      captured = JSON.parse(line) as Record<string, unknown>;
      socket.end(JSON.stringify({
        v: 1,
        id: captured.id,
        status: 200,
        body: { memories: [] },
      }) + "\n");
    });

    const client = new ShodhIpcClient(endpoint, "secret-key");
    await expect(client.request("/api/recall", "POST", { query: "hello" }, 1_000))
      .resolves.toEqual({ memories: [] });
    expect(captured).toMatchObject({
      v: 1,
      auth: "secret-key",
      method: "POST",
      path: "/api/recall",
      body: { query: "hello" },
    });
    expect(captured?.id).toMatch(/^[0-9a-f-]{36}$/);
  });

  it("maps non-success statuses to API-compatible errors", async () => {
    const endpoint = await listen((socket, line) => {
      const request = JSON.parse(line) as { id: string };
      socket.end(JSON.stringify({
        v: 1,
        id: request.id,
        status: 401,
        body: { code: "INVALID_API_KEY", message: "Invalid API key" },
      }) + "\n");
    });

    const client = new ShodhIpcClient(endpoint, "wrong-key");
    const error = await client.request("/api/users", "GET", null, 1_000).catch((cause) => cause);
    expect(error).toBeInstanceOf(IpcApiError);
    expect(error).toMatchObject({ status: 401 });
    expect((error as Error).message).toContain("INVALID_API_KEY");
  });

  it("rejects a response with the wrong protocol version", async () => {
    const endpoint = await listen((socket, line) => {
      const request = JSON.parse(line) as { id: string };
      socket.end(JSON.stringify({ v: 2, id: request.id, status: 200, body: {} }) + "\n");
    });

    const client = new ShodhIpcClient(endpoint, "key");
    await expect(client.request("/health", "GET", null, 1_000))
      .rejects.toThrow("Unsupported Shodh IPC response version");
  });

  it("rejects a response whose request ID does not match", async () => {
    const endpoint = await listen((socket) => {
      socket.end(JSON.stringify({ v: 1, id: randomUUID(), status: 200, body: {} }) + "\n");
    });

    const client = new ShodhIpcClient(endpoint, "key");
    await expect(client.request("/health", "GET", null, 1_000))
      .rejects.toThrow("request ID did not match");
  });

  it("rejects fields outside the strict version one response envelope", async () => {
    const endpoint = await listen((socket, line) => {
      const request = JSON.parse(line) as { id: string };
      socket.end(JSON.stringify({
        v: 1,
        id: request.id,
        status: 200,
        body: {},
        extra: true,
      }) + "\n");
    });

    const client = new ShodhIpcClient(endpoint, "key");
    await expect(client.request("/health", "GET", null, 1_000))
      .rejects.toThrow("fields did not match");
  });

  it("waits for close and rejects a second response frame", async () => {
    const endpoint = await listen((socket, line) => {
      const request = JSON.parse(line) as { id: string };
      socket.write(JSON.stringify({ v: 1, id: request.id, status: 200, body: {} }) + "\n");
      setTimeout(() => socket.end("{}\n"), 10);
    });

    const client = new ShodhIpcClient(endpoint, "key");
    await expect(client.request("/health", "GET", null, 1_000))
      .rejects.toThrow("data after");
  });

  it("rejects invalid UTF-8 responses", async () => {
    const endpoint = await listen((socket) => {
      socket.end(Buffer.from([0xc3, 0x28, 0x0a]));
    });

    const client = new ShodhIpcClient(endpoint, "key");
    await expect(client.request("/health", "GET", null, 1_000))
      .rejects.toThrow("not valid UTF-8");
  });

  it("rejects response frames over eight MiB", async () => {
    const endpoint = await listen((socket) => {
      socket.end(Buffer.alloc(IPC_MAX_FRAME_BYTES + 1, 0x20));
    });

    const client = new ShodhIpcClient(endpoint, "key");
    await expect(client.request("/health", "GET", null, 2_000))
      .rejects.toThrow("response exceeds");
  });

  it("enforces one total connect/read deadline", async () => {
    const endpoint = await listen(() => {
      // Keep the accepted connection open without replying.
    });

    const client = new ShodhIpcClient(endpoint, "key");
    await expect(client.request("/health", "GET", null, 30))
      .rejects.toThrow("timed out after 30 ms");
  });

  it("rejects oversized requests before opening a connection", async () => {
    const client = new ShodhIpcClient(testEndpoint(), "key");
    const oversized = "x".repeat(IPC_MAX_FRAME_BYTES);
    await expect(client.request("/api/remember", "POST", { content: oversized }, 1_000))
      .rejects.toThrow("request exceeds");
  });
});

async function listen(onLine: (socket: Socket, line: string) => void): Promise<string> {
  const endpoint = testEndpoint();
  const server = createServer((socket) => {
    let request = Buffer.alloc(0);
    socket.on("data", (chunk: Buffer) => {
      request = Buffer.concat([request, chunk]);
      const newline = request.indexOf(0x0a);
      if (newline !== -1) onLine(socket, request.subarray(0, newline).toString("utf8"));
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
