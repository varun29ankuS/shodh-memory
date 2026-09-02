import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

const HERE = path.dirname(fileURLToPath(import.meta.url));

/**
 * Driven over real stdio rather than by importing the handlers.
 *
 * The confinement check is the reason this connector is safe to point at a
 * shared drive, and it is only true end to end: a unit test of `within()` would
 * pass while an argument reached the filesystem by another path. So the tests
 * speak JSON-RPC to a spawned process, the way the seat does.
 */
async function call(root, requests) {
  const child = spawn(process.execPath, [path.join(HERE, "index.js"), root], {
    stdio: ["pipe", "pipe", "ignore"],
  });
  const messages = [
    {
      jsonrpc: "2.0",
      id: 1,
      method: "initialize",
      params: {
        protocolVersion: "2024-11-05",
        capabilities: {},
        clientInfo: { name: "test", version: "1" },
      },
    },
    { jsonrpc: "2.0", method: "notifications/initialized" },
    ...requests,
  ];
  child.stdin.end(messages.map((m) => JSON.stringify(m)).join("\n") + "\n");

  let out = "";
  for await (const chunk of child.stdout) out += chunk;

  const byId = new Map();
  for (const line of out.split("\n")) {
    if (!line.trim()) continue;
    try {
      const message = JSON.parse(line);
      if (message.id !== undefined) byId.set(message.id, message);
    } catch {
      // Partial line at the end of the stream; the ids we need have arrived.
    }
  }
  return byId;
}

const textOf = (message) => message.result.content[0].text;

async function fixture() {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), "corpus-test-"));
  await fs.writeFile(path.join(root, "brief.md"), "Seagirt terminal closed.\nDali struck the bridge.\n");
  await fs.mkdir(path.join(root, "nested"));
  await fs.writeFile(path.join(root, "nested", "note.txt"), "coal exports halted\n");
  await fs.writeFile(path.join(root, "image.bin"), Buffer.from([0, 1, 2, 3]));
  // Outside the root, and the thing an escape would be after.
  await fs.writeFile(path.join(root, "..", path.basename(root) + "-secret.txt"), "API_KEY=hunter2\n");
  return root;
}

test("lists documents recursively, hiding nothing it can read", async () => {
  const root = await fixture();
  const res = await call(root, [
    { jsonrpc: "2.0", id: 2, method: "tools/call", params: { name: "list_documents", arguments: {} } },
  ]);
  const text = textOf(res.get(2));
  assert.match(text, /brief\.md/);
  assert.match(text, /note\.txt/);
});

test("refuses a path that escapes the root", async () => {
  const root = await fixture();
  const res = await call(root, [
    {
      jsonrpc: "2.0",
      id: 2,
      method: "tools/call",
      params: { name: "read_document", arguments: { path: `../${path.basename(root)}-secret.txt` } },
    },
  ]);
  const message = res.get(2);
  assert.equal(message.result.isError, true, "an escape must not succeed");
  assert.match(textOf(message), /outside the corpus root/);
  assert.doesNotMatch(textOf(message), /hunter2/, "and must not leak the content either");
});

test("refuses an absolute path outside the root", async () => {
  const root = await fixture();
  const res = await call(root, [
    {
      jsonrpc: "2.0",
      id: 2,
      method: "tools/call",
      params: { name: "read_document", arguments: { path: path.join(os.tmpdir(), "..") } },
    },
  ]);
  assert.equal(res.get(2).result.isError, true);
});

test("reads a text document", async () => {
  const root = await fixture();
  const res = await call(root, [
    { jsonrpc: "2.0", id: 2, method: "tools/call", params: { name: "read_document", arguments: { path: "brief.md" } } },
  ]);
  assert.match(textOf(res.get(2)), /Dali struck the bridge/);
});

test("declines a binary file instead of returning mojibake", async () => {
  const root = await fixture();
  const res = await call(root, [
    { jsonrpc: "2.0", id: 2, method: "tools/call", params: { name: "read_document", arguments: { path: "image.bin" } } },
  ]);
  assert.match(textOf(res.get(2)), /Not a text document/);
});

test("search reports file and line, so a hit is actionable", async () => {
  const root = await fixture();
  const res = await call(root, [
    { jsonrpc: "2.0", id: 2, method: "tools/call", params: { name: "search_documents", arguments: { query: "coal" } } },
  ]);
  assert.match(textOf(res.get(2)), /note\.txt:1: coal exports halted/);
});

test("exposes no write, delete or move tool", async () => {
  const root = await fixture();
  const res = await call(root, [{ jsonrpc: "2.0", id: 2, method: "tools/list" }]);
  const names = res.get(2).result.tools.map((t) => t.name);
  assert.deepEqual(names.sort(), ["list_documents", "read_document", "search_documents"]);
  // Read-only by construction, not by policy: policy is the second line of
  // defence, and this asserts the first one still holds.
  for (const forbidden of ["write", "delete", "move", "create", "append"]) {
    assert.equal(
      names.some((n) => n.includes(forbidden)),
      false,
      `connector must not expose a ${forbidden} tool`,
    );
  }
});
