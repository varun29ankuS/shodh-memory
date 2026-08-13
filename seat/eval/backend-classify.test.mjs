/**
 * Transport-failure classification, checked two ways.
 *
 * The synthetic half pins the error shapes the classifier decodes. The live
 * half proves those shapes are the ones Node actually produces — the original
 * bug here was a classifier written against assumed shapes that returned
 * "other" for every real failure, and only a real socket catches that.
 *
 * Run: npm run build && npm test
 */
import { test } from "node:test";
import assert from "node:assert/strict";

import { ShodhBackend, ShodhBackendError, classifyTransportError, healthDetailForHttp } from "../dist/backend.js";

const VOCABULARY = new Set([
	"timeout", "refused", "dns", "tls", "reset", "unreachable", "protocol", "http", "other",
]);

// ── Synthetic shapes ────────────────────────────────────────────────────────

test("bare DOMException timeout (AbortSignal.timeout rejects with this directly)", () => {
	assert.equal(classifyTransportError(new DOMException("The operation was aborted due to timeout", "TimeoutError")), "timeout");
});

test("DOMException numeric code does not leak into the errno match", () => {
	const e = new DOMException("aborted", "TimeoutError");
	assert.equal(typeof e.code, "number", "guarding this is the point of the string check");
	assert.equal(classifyTransportError(e), "timeout");
});

test("fetch wraps the real cause one level down", () => {
	for (const [code, expected] of [
		["ECONNREFUSED", "refused"],
		["ENOTFOUND", "dns"],
		["EAI_AGAIN", "dns"],
		["ECONNRESET", "reset"],
		["EHOSTUNREACH", "unreachable"],
		["UND_ERR_HEADERS_TIMEOUT", "timeout"],
	]) {
		const cause = Object.assign(new Error(`connect ${code}`), { code });
		const wrapped = new TypeError("fetch failed", { cause });
		assert.equal(classifyTransportError(wrapped), expected, `${code} should classify as ${expected}`);
	}
});

test("dual-stack host nests an AggregateError inside the cause", () => {
	const inner = ["::1", "127.0.0.1"].map((addr) =>
		Object.assign(new Error(`connect ECONNREFUSED ${addr}:45999`), { code: "ECONNREFUSED" }),
	);
	const wrapped = new TypeError("fetch failed", { cause: new AggregateError(inner, "") });
	assert.equal(classifyTransportError(wrapped), "refused");
});

test("TLS cert codes carry no ERR_TLS_ prefix", () => {
	const cause = Object.assign(new Error("self-signed certificate"), { code: "DEPTH_ZERO_SELF_SIGNED_CERT" });
	assert.equal(classifyTransportError(new TypeError("fetch failed", { cause })), "tls");
});

test("unknown failures fall back to other, and a cyclic cause terminates", () => {
	assert.equal(classifyTransportError(new Error("something new")), "other");
	const cyclic = new Error("loop");
	cyclic.cause = cyclic;
	assert.equal(classifyTransportError(cyclic), "other");
});

// ── Live sockets ────────────────────────────────────────────────────────────

const probe = (url, timeoutMs = 800) => new ShodhBackend(url, "test-key", timeoutMs).health();

test("live: connection refused", async () => {
	const error = await probe("http://127.0.0.1:45999").then(() => null, (e) => e);
	assert.ok(error instanceof ShodhBackendError);
	assert.equal(error.kind, "refused");
});

test("live: dns failure", async () => {
	const error = await probe("http://no-such-host-xyzzy.invalid").then(() => null, (e) => e);
	assert.ok(error instanceof ShodhBackendError);
	assert.equal(error.kind, "dns");
});

test("live: timeout is distinct from refused", async () => {
	const error = await probe("http://10.255.255.1:3030", 300).then(() => null, (e) => e);
	assert.ok(error instanceof ShodhBackendError);
	assert.equal(error.kind, "timeout");
});

// ── The security property ───────────────────────────────────────────────────

test("no classification carries the host, port, or raw message", async () => {
	for (const url of ["http://127.0.0.1:45999", "http://no-such-host-xyzzy.invalid"]) {
		const error = await probe(url).then(() => null, (e) => e);
		assert.ok(VOCABULARY.has(error.kind), `${error.kind} is outside the closed vocabulary`);
		assert.doesNotMatch(error.kind, /[.:/]/, "a classification must not look like a URL");
		// The detail is safe precisely because the message it came from is not.
		assert.match(error.message, /Backend unreachable/);
	}
});

// ── The kind cannot be forged ───────────────────────────────────────────────

test("a transport kind can only come from classifying a real cause", () => {
	const cause = Object.assign(new Error("connect ECONNREFUSED"), { code: "ECONNREFUSED" });
	const err = ShodhBackendError.transport("Backend unreachable", new TypeError("fetch failed", { cause }));
	assert.equal(err.kind, "refused");
	assert.equal(err.status, 0);
	// The three factories are the whole supported surface. `private constructor`
	// is erased at runtime, so what stops a call site asserting a kind that
	// contradicts its cause is the type checker — see the compile-time check in
	// eval/compile-guards.md, exercised by `npm run typecheck`.
	assert.deepEqual(
		Object.getOwnPropertyNames(ShodhBackendError)
			.filter((k) => typeof ShodhBackendError[k] === "function")
			.sort(),
		["http", "protocol", "transport"],
	);
});

test("http and protocol factories fix their own kind", () => {
	assert.equal(ShodhBackendError.http("e", 500, "").kind, "http");
	assert.equal(ShodhBackendError.protocol("e", 200, "<html>").kind, "protocol");
});

test("4xx is collapsed for the unauthenticated route, 5xx keeps its status", () => {
	assert.equal(healthDetailForHttp(500), "http-500");
	assert.equal(healthDetailForHttp(503), "http-503");
	assert.equal(healthDetailForHttp(401), "http-client-error");
	assert.equal(healthDetailForHttp(404), "http-client-error");
});

test("every DNS errno a runner might report maps to dns", () => {
	for (const code of ["ENOTFOUND", "EAI_AGAIN", "EAI_NONAME", "EAI_FAIL", "EAI_NODATA", "ENODATA"]) {
		const cause = Object.assign(new Error(code), { code });
		assert.equal(classifyTransportError(new TypeError("fetch failed", { cause })), "dns", code);
	}
});
