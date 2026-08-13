/**
 * End-to-end for GET /healthz over real HTTP.
 *
 * The unit tests cover classifyTransportError in isolation, but the property
 * that matters is about the bytes that leave the process on an unauthenticated
 * route — and no unit test can see those. Everything here is real: a real seat
 * server on a real socket, a real ShodhBackend, and a real backend process that
 * we make fail in each way that matters. Nothing is stubbed.
 *
 * Run: npm run build && npm test
 */
import { test, before, after } from "node:test";
import assert from "node:assert/strict";
import http from "node:http";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";

import { SeatServer } from "../dist/server.js";
import { ShodhBackend } from "../dist/backend.js";
import { ModelRegistry } from "../dist/models-registry.js";
import { FileCredentialStore } from "../dist/credentials.js";
import { LearningLedger } from "../dist/ledger.js";
import { McpHost } from "../dist/mcp.js";
import { SeatStore } from "../dist/store.js";

const FAILURE_VOCABULARY = new Set([
	"timeout", "refused", "dns", "tls", "reset", "unreachable", "protocol", "other",
	"http-client-error",
]);

let dataDir;
const started = [];

/** A real backend on a real port, answering however the test needs it to. */
async function backendServing(handler) {
	const server = http.createServer(handler);
	await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
	started.push(server);
	return `http://127.0.0.1:${server.address().port}`;
}

/** A real seat server pointed at `apiUrl`. Returns its base URL. */
async function seatPointedAt(apiUrl, timeoutMs = 1500) {
	const config = {
		apiUrl,
		apiKey: "test-key",
		host: "127.0.0.1",
		port: 0,
		dataDir,
		ollamaBaseUrl: "http://127.0.0.1:45997/v1",
		lmStudioBaseUrl: "http://127.0.0.1:45996/v1",
		vllmBaseUrl: "http://127.0.0.1:45995/v1",
		localContextWindow: 8192,
		localMaxTokens: 1024,
		backendTimeoutMs: timeoutMs,
		mcpConnectTimeoutMs: 500,
		mcpServers: [],
	};
	const credentials = new FileCredentialStore(dataDir);
	const seat = new SeatServer({
		config,
		backend: new ShodhBackend(apiUrl, config.apiKey, timeoutMs),
		registry: new ModelRegistry(config, credentials),
		ledger: new LearningLedger(dataDir),
		mcpHost: new McpHost({ connectTimeoutMs: 500, log: () => {} }),
		store: new SeatStore(dataDir),
	});
	// port 0 asks the OS for a free port; read back what it gave us.
	await seat.listen();
	started.push({ close: () => seat.close() });
	return `http://127.0.0.1:${seat.server.address().port}`;
}

const getHealth = async (base) => {
	const res = await fetch(`${base}/healthz`);
	return { status: res.status, body: await res.json() };
};

before(() => {
	dataDir = fs.mkdtempSync(path.join(os.tmpdir(), "seat-healthz-"));
});

after(async () => {
	for (const s of started.reverse()) await new Promise((r) => (s.close(r), setTimeout(r, 0)));
	fs.rmSync(dataDir, { recursive: true, force: true });
});

// ── Positive: the backend is healthy ────────────────────────────────────────

test("healthy backend answers 200 with a normalised detail", async () => {
	const api = await backendServing((_, res) => {
		res.writeHead(200, { "Content-Type": "application/json" });
		res.end(JSON.stringify({ status: "healthy", version: "0.1.0" }));
	});
	const { status, body } = await getHealth(await seatPointedAt(api));
	assert.equal(status, 200);
	assert.equal(body.seat, "ok");
	assert.equal(body.backend.ok, true);
	assert.equal(body.backend.detail, "healthy");
});

test("an unrecognised status is normalised, not echoed", async () => {
	const api = await backendServing((_, res) => {
		res.writeHead(200, { "Content-Type": "application/json" });
		res.end(JSON.stringify({ status: "surprise-value-from-upstream", version: "0.1.0" }));
	});
	const { status, body } = await getHealth(await seatPointedAt(api));
	assert.equal(status, 503);
	assert.equal(body.backend.ok, false);
	assert.equal(body.backend.detail, "unexpected-status");
	assert.ok(!JSON.stringify(body).includes("surprise-value-from-upstream"));
});

// ── Negative: each failure mode, end to end ─────────────────────────────────

test("connection refused is reported as refused, not unreachable", async () => {
	const { status, body } = await getHealth(await seatPointedAt("http://127.0.0.1:45999"));
	assert.equal(status, 503);
	assert.equal(body.backend.detail, "refused");
});

test("dns failure is distinguishable from refused", async () => {
	const { body } = await getHealth(await seatPointedAt("http://no-such-host-xyzzy.invalid"));
	assert.equal(body.backend.detail, "dns");
});

test("a hung backend is reported as timeout", async () => {
	const api = await backendServing(() => { /* never responds */ });
	const { body } = await getHealth(await seatPointedAt(api, 300));
	assert.equal(body.backend.detail, "timeout");
});

// This is the branch the PR collapsed: a backend answering 500 is reachable.
test("an erroring backend reports its status, not unreachable", async () => {
	const api = await backendServing((_, res) => {
		res.writeHead(500, { "Content-Type": "application/json" });
		res.end(JSON.stringify({ error: "boom" }));
	});
	const { status, body } = await getHealth(await seatPointedAt(api));
	assert.equal(status, 503);
	assert.equal(body.backend.detail, "http-500");
	assert.notEqual(body.backend.detail, "unreachable");
});

// 4xx is collapsed on purpose: a bare http-401 tells an anonymous caller that
// the seat's own backend credential is rejected — our configuration, not the
// backend's liveness.
test("a rejected backend credential does not name its status", async () => {
	const api = await backendServing((_, res) => {
		res.writeHead(401, { "Content-Type": "application/json" });
		res.end(JSON.stringify({ error: "bad key" }));
	});
	const { body } = await getHealth(await seatPointedAt(api));
	assert.equal(body.backend.detail, "http-client-error");
	assert.ok(!JSON.stringify(body).includes("401"), "4xx status leaked to an anonymous caller");
});

test("a proxy error page is protocol, not unreachable", async () => {
	const api = await backendServing((_, res) => {
		res.writeHead(200, { "Content-Type": "text/html" });
		res.end("<html><body>502 Bad Gateway</body></html>");
	});
	const { body } = await getHealth(await seatPointedAt(api));
	assert.equal(body.backend.detail, "protocol");
});

// ── The security property, measured where it actually applies ───────────────

test("no failure response leaks the backend host, port or raw message", async () => {
	const secretHost = "internal-backend.corp.invalid";
	const cases = [
		`http://127.0.0.1:45999`,
		`http://${secretHost}`,
	];
	for (const api of cases) {
		const { body } = await getHealth(await seatPointedAt(api));
		const wire = JSON.stringify(body);
		assert.ok(!wire.includes(secretHost), `response leaked the host: ${wire}`);
		assert.ok(!wire.includes("45999"), `response leaked the port: ${wire}`);
		assert.ok(!wire.includes("fetch failed"), `response leaked the raw message: ${wire}`);
		assert.ok(!wire.includes("ECONNREFUSED"), `response leaked the errno: ${wire}`);
		assert.ok(
			FAILURE_VOCABULARY.has(body.backend.detail) || /^http-\d+$/.test(body.backend.detail),
			`detail "${body.backend.detail}" is outside the published vocabulary`,
		);
	}
});

// The route answers before authorize(), so this must hold with a token set.
test("/healthz stays reachable unauthenticated — the reason detail must be safe", async () => {
	const api = await backendServing((_, res) => {
		res.writeHead(200, { "Content-Type": "application/json" });
		res.end(JSON.stringify({ status: "healthy", version: "0.1.0" }));
	});
	const base = await seatPointedAt(api);
	const res = await fetch(`${base}/healthz`); // no Authorization header
	assert.equal(res.status, 200);
});
