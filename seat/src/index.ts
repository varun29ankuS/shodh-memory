/**
 * Seat harness entry point.
 */

import * as fsp from "node:fs/promises";
import { ShodhBackend } from "./backend.js";
import { loadConfig, type McpServerConfig, parseMcpServers } from "./config.js";
import { FileCredentialStore } from "./credentials.js";
import { LearningLedger } from "./ledger.js";
import { McpHost } from "./mcp.js";
import { ModelRegistry } from "./models-registry.js";
import { SeatServer } from "./server.js";
import { SeatStore } from "./store.js";
import { ViewLink } from "./view-link.js";

async function loadMcpServers(configPath: string | undefined): Promise<McpServerConfig[]> {
	if (!configPath) return [];
	const raw = await fsp.readFile(configPath, "utf8");
	return parseMcpServers(raw, configPath);
}

async function main(): Promise<void> {
	const config = loadConfig();
	const backend = new ShodhBackend(config.apiUrl, config.apiKey, config.backendTimeoutMs);
	const credentials = new FileCredentialStore(config.dataDir);
	const registry = new ModelRegistry(config, credentials);
	const ledger = new LearningLedger(config.dataDir);
	const store = new SeatStore(config.dataDir);
	// One per process: the view tools register an ask on it and the view-report
	// route resolves it. Two instances would be two conversations that never meet.
	const viewLink = new ViewLink();
	const mcpHost = new McpHost({
		connectTimeoutMs: config.mcpConnectTimeoutMs,
		log: (message) => console.warn(message),
	});

	try {
		const health = await backend.health();
		console.log(`[seat] backend ${config.apiUrl} reachable (status: ${health.status})`);
	} catch (error) {
		console.warn(
			`[seat] WARNING: backend ${config.apiUrl} not reachable yet — memory tools will fail until it is up. ` +
				`(${error instanceof Error ? error.message : String(error)})`,
		);
	}

	const localErrors = await registry.refreshLocal();
	for (const [provider, message] of Object.entries(localErrors)) {
		console.log(`[seat] local provider ${provider} offline: ${message}`);
	}

	// A malformed servers file is fatal (the operator asked for these servers
	// and got none of them, which they must be told loudly), but the
	// CONNECTIONS are not: they run behind the listener, so one endpoint that
	// accepts a socket and says nothing cannot decide when the seat starts
	// answering. Conversations pick up whatever is connected at the time of
	// each turn, so a server that finishes connecting a second after boot is
	// still available to the first message.
	const mcpServers = await loadMcpServers(config.mcpConfigPath);
	const mcpConnected = mcpServers.length > 0 ? mcpHost.connect(mcpServers) : Promise.resolve();

	const server = new SeatServer({ config, backend, registry, ledger, mcpHost, store, viewLink });
	await server.listen();
	console.log(`[seat] listening on http://${config.host}:${config.port}`);

	void mcpConnected.then(() => {
		for (const mcp of mcpHost.listServers()) {
			const where = mcp.endpoint ?? mcp.command ?? "";
			if (mcp.status === "ready") {
				console.log(`[seat] MCP "${mcp.name}" ready over ${mcp.transport}: ${mcp.tool_count} tools bridged`);
			} else {
				console.warn(`[seat] MCP "${mcp.name}" ${mcp.status} (${mcp.transport}${where ? ` → ${where}` : ""}): ${mcp.error ?? "no reason reported"}`);
			}
		}
	});
	console.log(`[seat] learning ledger: ${ledger.file}`);
	console.log(`[seat] conversation store + provider credentials: ${config.dataDir}`);

	const shutdown = async (signal: string): Promise<void> => {
		console.log(`[seat] ${signal} received, shutting down`);
		await server.close();
		await mcpHost.close();
		process.exit(0);
	};
	process.on("SIGINT", () => void shutdown("SIGINT"));
	process.on("SIGTERM", () => void shutdown("SIGTERM"));
}

main().catch((error) => {
	console.error(`[seat] fatal: ${error instanceof Error ? error.message : String(error)}`);
	process.exit(1);
});
