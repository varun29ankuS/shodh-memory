/**
 * End-to-end for POST /v1/conversations/{id}/view-report over real HTTP.
 *
 * The unit tests cover the wire contract and the rendezvous in isolation. What
 * no unit test can see is the SEAM: whether the route is reachable at all,
 * whether a rejected body comes back as a 400 a client can act on, whether an
 * outcome for an ask nobody issued is refused, and whether an accepted one
 * actually lands in the event store the audit export reads from. Every one of
 * those is a way the loop could be closed on paper and open in practice.
 *
 * Everything here is real: a real seat server on a real socket, a real SQLite
 * store on disk, a real ViewLink. Nothing is stubbed.
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
import { ViewLink } from "../dist/view-link.js";
import { buildAuditRows } from "../dist/audit.js";

let dataDir;
let base;
let store;
let viewLink;
const started = [];

const CONVERSATION = "conv-e2e";

function view(over = {}) {
	return {
		destination: "/graph",
		profile: "demo",
		cue: null,
		focus: null,
		claimed: [],
		offers: [],
		...over,
	};
}

async function post(conversationId, body) {
	const res = await fetch(`${base}/v1/conversations/${conversationId}/view-report`, {
		method: "POST",
		headers: { "Content-Type": "application/json" },
		body: JSON.stringify(body),
	});
	return { status: res.status, body: await res.json() };
}

before(async () => {
	dataDir = fs.mkdtempSync(path.join(os.tmpdir(), "seat-viewreport-"));
	const config = {
		apiUrl: "http://127.0.0.1:45999",
		apiKey: "test-key",
		host: "127.0.0.1",
		port: 0,
		dataDir,
		ollamaBaseUrl: "http://127.0.0.1:45997/v1",
		lmStudioBaseUrl: "http://127.0.0.1:45996/v1",
		vllmBaseUrl: "http://127.0.0.1:45995/v1",
		localContextWindow: 8192,
		localMaxTokens: 1024,
		backendTimeoutMs: 500,
		mcpConnectTimeoutMs: 500,
		mcpServers: [],
	};
	const credentials = new FileCredentialStore(dataDir);
	store = new SeatStore(dataDir);
	viewLink = new ViewLink(200);
	const seat = new SeatServer({
		config,
		backend: new ShodhBackend(config.apiUrl, config.apiKey, 500),
		registry: new ModelRegistry(config, credentials),
		ledger: new LearningLedger(dataDir),
		mcpHost: new McpHost({ connectTimeoutMs: 500, log: () => {} }),
		store,
		viewLink,
	});
	await seat.listen();
	started.push({ close: () => seat.close() });
	base = `http://127.0.0.1:${seat.server.address().port}`;

	store.createConversation({
		conversationId: CONVERSATION,
		userId: "demo",
		provider: "anthropic",
		modelId: "m",
		modelName: "M",
		createdAt: new Date(),
	});
});

after(async () => {
	for (const s of started.reverse()) await s.close();
	fs.rmSync(dataDir, { recursive: true, force: true });
});

// ── The route exists and answers ────────────────────────────────────────────

test("a report for an unknown conversation is a 404, not a silent accept", async () => {
	const { status } = await post("conv-does-not-exist", { outcomes: [], view: view() });
	assert.equal(status, 404);
});

test("a state this seat does not recognise is a 400 naming the ones it does", async () => {
	// The browser and the seat keep two copies of the vocabulary. A coerced
	// value here would write a false statement about a person's decision into a
	// durable row, so drift has to be a loud failure at the boundary.
	viewLink.open(CONVERSATION, "call-bad");
	const { status, body } = await post(CONVERSATION, {
		outcomes: [{ tool_call_id: "call-bad", dimension: "destination", state: "dismissed" }],
		view: view(),
	});
	assert.equal(status, 400);
	assert.match(body.error, /declined/);
	assert.match(body.error, /expired/);
});

test("an outcome for an ask nobody issued is refused", async () => {
	// Otherwise anything holding the bearer token could write verdicts for
	// commands that appear nowhere in the trail.
	const { status, body } = await post(CONVERSATION, {
		outcomes: [{ tool_call_id: "never-asked", dimension: "cue", state: "applied" }],
		view: view(),
	});
	assert.equal(status, 404);
	assert.match(body.error, /never-asked/);
});

// ── The verdict reaches the waiting tool ────────────────────────────────────

test("a report resolves the tool call that is waiting on it", async () => {
	viewLink.open(CONVERSATION, "call-1");
	const pending = viewLink.await("call-1");
	const { status, body } = await post(CONVERSATION, {
		outcomes: [
			{ tool_call_id: "call-1", dimension: "cue", state: "applied" },
			{ tool_call_id: "call-1", dimension: "destination", state: "offered" },
		],
		view: view({ claimed: ["destination"] }),
	});
	assert.equal(status, 200);
	assert.equal(body.recorded, 2);
	assert.deepEqual(await pending, [
		{ tool_call_id: "call-1", dimension: "cue", state: "applied" },
		{ tool_call_id: "call-1", dimension: "destination", state: "offered" },
	]);
});

test("nothing reported leaves the tool with null — the honest unknown", async () => {
	viewLink.open(CONVERSATION, "call-silent");
	assert.equal(await viewLink.await("call-silent"), null);
});

// ── The verdict reaches the audit trail ─────────────────────────────────────

test("an accepted outcome becomes a durable row the export can read", async () => {
	viewLink.open(CONVERSATION, "call-2");
	await post(CONVERSATION, {
		outcomes: [{ tool_call_id: "call-2", dimension: "destination", state: "declined" }],
		view: view({ destination: "/chat" }),
	});
	const events = store.queryEvents({ conversationId: CONVERSATION, types: ["view_outcome"] });
	const mine = events.filter((row) => row.event.tool_call_id === "call-2");
	assert.equal(mine.length, 1);

	const [row] = buildAuditRows({ entries: [], events: mine });
	assert.equal(row.source, "view");
	assert.equal(row.kind, "view_outcome");
	// The person's decision, filed under the person.
	assert.equal(row.actor, "user");
	assert.equal(row.ref, "call-2");
	assert.deepEqual(JSON.parse(row.detail), {
		dimension: "destination",
		state: "declined",
		at: "/chat",
	});
});

test("an offer accepted AFTER a restart is still recorded, from the durable ask", async () => {
	// The registry's memory dies with the process; a Follow chip on screen does
	// not. Refusing that report would drop the one record of a person accepting
	// an offer — so the store is consulted when the registry does not know.
	store.appendEvent(CONVERSATION, {
		turn: 1,
		ts: new Date().toISOString(),
		event: {
			type: "view_command",
			tool_call_id: "call-before-restart",
			reason: "these cluster on the coast",
			destination: "/geo",
			entities: [],
			unresolved: [],
			focus: null,
		},
	});
	assert.equal(viewLink.knows("call-before-restart"), false, "the registry must not know it");

	const { status, body } = await post(CONVERSATION, {
		outcomes: [{ tool_call_id: "call-before-restart", dimension: "destination", state: "followed" }],
		view: view(),
	});
	assert.equal(status, 200);
	assert.equal(body.recorded, 1);
});

// ── Perception ──────────────────────────────────────────────────────────────

test("a probe is answered over the wire and the state is cached with its age", async () => {
	const probeId = viewLink.openProbe(CONVERSATION);
	const pending = viewLink.awaitProbe(probeId);
	const { status } = await post(CONVERSATION, {
		probe_id: probeId,
		outcomes: [],
		view: view({ destination: "/geo", claimed: ["cue"], focus: { id: "u-1", name: "Dali" } }),
	});
	assert.equal(status, 200);

	const answered = await pending;
	assert.equal(answered.destination, "/geo");
	assert.deepEqual(answered.claimed, ["cue"]);
	assert.deepEqual(answered.focus, { id: "u-1", name: "Dali" });

	const cached = viewLink.lastSnapshot(CONVERSATION);
	assert.equal(cached.view.destination, "/geo");
	assert.ok(cached.ageMs >= 0);
});

test("a report with neither an outcome nor a probe is accepted as a state update", async () => {
	// The browser volunteers its state alongside every verdict; a report that
	// carries only state is the same thing with nothing to answer, and rejecting
	// it would make the cache staler than it needs to be.
	const { status, body } = await post(CONVERSATION, { outcomes: [], view: view() });
	assert.equal(status, 200);
	assert.equal(body.recorded, 0);
	assert.deepEqual(body.unrecognised, []);
});

test("the route is reachable at all — a wrong method still 404s rather than hanging", async () => {
	const res = await fetch(`${base}/v1/conversations/${CONVERSATION}/view-report`, { method: "GET" });
	assert.equal(res.status, 404);
	assert.ok(http.STATUS_CODES[res.status]);
});
