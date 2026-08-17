/**
 * Conversation persistence: SQLite in the seat data directory.
 *
 * Why not pi's `@earendil-works/pi-session-backend-sqlite-node`: that package
 * implements pi-agent-core's `SessionRepository` — session trees keyed by a
 * working directory, with entry lanes, branch caches, leases and FTS. The seat
 * does not use pi's session layer at all (`Conversation` wraps a raw `Agent`),
 * and the seat's product surface is its OWN event stream — `memory_recall`
 * with full ScoreAttribution, `proactive_context`, reinforcements, usage —
 * which pi's repository has no representation for. Adopting it would still
 * leave every seat event needing a second store, so the seat gets one store
 * shaped like the seat instead: three tables, `node:sqlite`, no dependency.
 *
 * What is durable, and why:
 * - `conversations` — listing metadata plus accumulated token/cost totals, so
 *   the session list never has to replay transcripts to show real numbers.
 * - `transcripts` — the pi `AgentMessage[]` snapshot after each turn. This is
 *   what re-seeds `Agent.state.messages` when a conversation is reopened
 *   after a restart, and the authority for rendered text.
 * - `events` — every SeatEvent except the two delta streams (`text_delta`,
 *   `thinking_delta`, whose final form lives in the transcript). This is what
 *   lets the UI rebuild the evidence surface — recalls, attributions,
 *   reinforcements, ledger references — for a reopened conversation.
 *
 * The database lives in SEAT_DATA_DIR next to the learning ledger,
 * deliberately outside watched/synced folders (see config.ts).
 */

import * as fs from "node:fs";
import * as path from "node:path";
import { DatabaseSync } from "node:sqlite";
import type { SeatEvent } from "./events.js";

/** Usage totals accumulated from `usage` events, in the event's own units. */
export interface UsageTotals {
	input: number;
	output: number;
	cache_read: number;
	cache_write: number;
	reasoning: number;
	total_tokens: number;
	cost_total: number;
}

export interface StoredConversation {
	conversation_id: string;
	user_id: string;
	/** Null until the first user message names it. */
	title: string | null;
	provider: string;
	model_id: string;
	model_name: string;
	system_prompt: string | null;
	created_at: string;
	updated_at: string;
	turns: number;
	usage: UsageTotals;
}

/** A durable SeatEvent with its position in the conversation. */
export interface StoredEvent {
	turn: number;
	ts: string;
	event: SeatEvent;
}

/**
 * A durable event carrying the conversation and user it belongs to, for reads
 * that span conversations. `listEvents` can omit both because the caller
 * already named the conversation; an audit read cannot.
 */
export interface StoredEventRow extends StoredEvent {
	conversation_id: string;
	user_id: string;
}

/**
 * Filter for {@link SeatStore.queryEvents}. Every field narrows; omitting all
 * of them reads the whole store, which is why `limit` has a default.
 */
export interface EventQuery {
	/** Backend user namespace, as recorded on the conversation. */
	userId?: string;
	conversationId?: string;
	/** SeatEvent `type` values to include. Empty or omitted means every type. */
	types?: readonly SeatEvent["type"][];
	/** ISO-8601 UTC, inclusive lower bound on the event timestamp. */
	since?: string;
	/** ISO-8601 UTC, exclusive upper bound on the event timestamp. */
	until?: string;
	limit?: number;
}

/** Default read ceiling for {@link SeatStore.queryEvents}. */
export const DEFAULT_EVENT_QUERY_LIMIT = 5000;

/**
 * Event types that are NOT persisted.
 *
 * The deltas are transient because the transcript holds their final text.
 * `view_probe` is transient for a different reason: it is a QUESTION the seat
 * asked the browser, and the `inspect_view` tool call that produced it is
 * already a durable row in the audit trail. Storing the probe as well would put
 * two lines in the artefact for one act, in a file whose whole value is that
 * each line is a distinct thing that happened.
 */
const TRANSIENT_EVENT_TYPES = new Set<SeatEvent["type"]>(["text_delta", "thinking_delta", "view_probe"]);

export function isDurableEvent(event: SeatEvent): boolean {
	return !TRANSIENT_EVENT_TYPES.has(event.type);
}

export const EMPTY_USAGE_TOTALS: UsageTotals = {
	input: 0,
	output: 0,
	cache_read: 0,
	cache_write: 0,
	reasoning: 0,
	total_tokens: 0,
	cost_total: 0,
};

interface ConversationRow {
	conversation_id: string;
	user_id: string;
	title: string | null;
	provider: string;
	model_id: string;
	model_name: string;
	system_prompt: string | null;
	created_at: string;
	updated_at: string;
	turns: number;
	usage_input: number;
	usage_output: number;
	usage_cache_read: number;
	usage_cache_write: number;
	usage_reasoning: number;
	usage_total_tokens: number;
	usage_cost_total: number;
}

function rowToConversation(row: ConversationRow): StoredConversation {
	return {
		conversation_id: row.conversation_id,
		user_id: row.user_id,
		title: row.title,
		provider: row.provider,
		model_id: row.model_id,
		model_name: row.model_name,
		system_prompt: row.system_prompt,
		created_at: row.created_at,
		updated_at: row.updated_at,
		turns: row.turns,
		usage: {
			input: row.usage_input,
			output: row.usage_output,
			cache_read: row.usage_cache_read,
			cache_write: row.usage_cache_write,
			reasoning: row.usage_reasoning,
			total_tokens: row.usage_total_tokens,
			cost_total: row.usage_cost_total,
		},
	};
}

export interface CreateConversationInput {
	conversationId: string;
	userId: string;
	provider: string;
	modelId: string;
	modelName: string;
	systemPrompt?: string;
	createdAt: Date;
}

export interface TurnPersistInput {
	conversationId: string;
	/** Full `AgentMessage[]` snapshot — replaces the previous transcript. */
	messages: unknown[];
	turns: number;
	usageDelta: UsageTotals;
	/** Durable events raised during this turn, in emission order. */
	events: StoredEvent[];
	/** Set as the title if the conversation has none yet. */
	titleCandidate?: string;
}

export class SeatStore {
	private readonly db: DatabaseSync;

	constructor(dataDir: string) {
		fs.mkdirSync(dataDir, { recursive: true });
		this.db = new DatabaseSync(path.join(dataDir, "seat.db"));
		this.db.exec("PRAGMA journal_mode = WAL");
		this.db.exec("PRAGMA foreign_keys = ON");
		this.db.exec(`
			CREATE TABLE IF NOT EXISTS conversations (
				conversation_id   TEXT PRIMARY KEY,
				user_id           TEXT NOT NULL,
				title             TEXT,
				provider          TEXT NOT NULL,
				model_id          TEXT NOT NULL,
				model_name        TEXT NOT NULL,
				system_prompt     TEXT,
				created_at        TEXT NOT NULL,
				updated_at        TEXT NOT NULL,
				turns             INTEGER NOT NULL DEFAULT 0,
				usage_input       REAL NOT NULL DEFAULT 0,
				usage_output      REAL NOT NULL DEFAULT 0,
				usage_cache_read  REAL NOT NULL DEFAULT 0,
				usage_cache_write REAL NOT NULL DEFAULT 0,
				usage_reasoning   REAL NOT NULL DEFAULT 0,
				usage_total_tokens REAL NOT NULL DEFAULT 0,
				usage_cost_total  REAL NOT NULL DEFAULT 0
			);
			CREATE INDEX IF NOT EXISTS idx_conversations_user
				ON conversations (user_id, updated_at DESC);
			CREATE TABLE IF NOT EXISTS transcripts (
				conversation_id TEXT PRIMARY KEY
					REFERENCES conversations(conversation_id) ON DELETE CASCADE,
				messages        TEXT NOT NULL
			);
			CREATE TABLE IF NOT EXISTS events (
				id              INTEGER PRIMARY KEY AUTOINCREMENT,
				conversation_id TEXT NOT NULL
					REFERENCES conversations(conversation_id) ON DELETE CASCADE,
				turn            INTEGER NOT NULL,
				ts              TEXT NOT NULL,
				type            TEXT NOT NULL,
				payload         TEXT NOT NULL
			);
			CREATE INDEX IF NOT EXISTS idx_events_conversation
				ON events (conversation_id, id);
			-- Audit reads scan by type and by time ACROSS conversations, which the
			-- conversation-keyed index above cannot serve.
			CREATE INDEX IF NOT EXISTS idx_events_type_ts
				ON events (type, ts);
			CREATE INDEX IF NOT EXISTS idx_events_ts
				ON events (ts);
		`);
	}

	close(): void {
		this.db.close();
	}

	createConversation(input: CreateConversationInput): StoredConversation {
		const now = input.createdAt.toISOString();
		this.db
			.prepare(
				`INSERT INTO conversations
					(conversation_id, user_id, title, provider, model_id, model_name, system_prompt, created_at, updated_at)
				 VALUES (?, ?, NULL, ?, ?, ?, ?, ?, ?)`,
			)
			.run(
				input.conversationId,
				input.userId,
				input.provider,
				input.modelId,
				input.modelName,
				input.systemPrompt ?? null,
				now,
				now,
			);
		const created = this.getConversation(input.conversationId);
		if (!created) throw new Error(`Conversation ${input.conversationId} vanished on insert`);
		return created;
	}

	getConversation(conversationId: string): StoredConversation | undefined {
		const row = this.db
			.prepare(`SELECT * FROM conversations WHERE conversation_id = ?`)
			.get(conversationId) as ConversationRow | undefined;
		return row ? rowToConversation(row) : undefined;
	}

	listConversations(userId?: string): StoredConversation[] {
		const rows = (
			userId
				? this.db
						.prepare(`SELECT * FROM conversations WHERE user_id = ? ORDER BY updated_at DESC`)
						.all(userId)
				: this.db.prepare(`SELECT * FROM conversations ORDER BY updated_at DESC`).all()
		) as unknown as ConversationRow[];
		return rows.map(rowToConversation);
	}

	renameConversation(conversationId: string, title: string): void {
		this.db
			.prepare(`UPDATE conversations SET title = ?, updated_at = ? WHERE conversation_id = ?`)
			.run(title, new Date().toISOString(), conversationId);
	}

	deleteConversation(conversationId: string): boolean {
		const result = this.db
			.prepare(`DELETE FROM conversations WHERE conversation_id = ?`)
			.run(conversationId);
		return result.changes > 0;
	}

	setModel(conversationId: string, provider: string, modelId: string, modelName: string): void {
		this.db
			.prepare(
				`UPDATE conversations SET provider = ?, model_id = ?, model_name = ?, updated_at = ?
				 WHERE conversation_id = ?`,
			)
			.run(provider, modelId, modelName, new Date().toISOString(), conversationId);
	}

	loadTranscript(conversationId: string): unknown[] | undefined {
		const row = this.db
			.prepare(`SELECT messages FROM transcripts WHERE conversation_id = ?`)
			.get(conversationId) as { messages: string } | undefined;
		if (!row) return undefined;
		return JSON.parse(row.messages) as unknown[];
	}

	listEvents(conversationId: string): StoredEvent[] {
		const rows = this.db
			.prepare(`SELECT turn, ts, payload FROM events WHERE conversation_id = ? ORDER BY id`)
			.all(conversationId) as unknown as { turn: number; ts: string; payload: string }[];
		return rows.map((row) => ({
			turn: row.turn,
			ts: row.ts,
			event: JSON.parse(row.payload) as SeatEvent,
		}));
	}

	/**
	 * Durable events across conversations, oldest first.
	 *
	 * `listEvents` answers "what happened in this conversation"; this answers
	 * "what happened, full stop" — the question an audit read asks and the one
	 * the conversation-keyed path could not answer at all. The join onto
	 * `conversations` supplies the user namespace, which the events table does
	 * not carry (it is a property of the conversation, not of each event).
	 *
	 * Ordering is `ts, id`: event timestamps are `Date.toISOString()`, so UTC
	 * with a fixed-width millisecond field, and lexicographic order over those
	 * is chronological order. `id` breaks same-millisecond ties by insertion,
	 * making the sequence total and the read repeatable.
	 *
	 * `limit` keeps the MOST RECENT matches (selected descending, returned
	 * ascending). Keeping the oldest instead would truncate a window at its
	 * newest edge, which for tool calls means retaining a `tool_call_start`
	 * while cutting its `tool_call_end` — rendering a call that completed as
	 * one that never returned. A dropped call is a visible gap; a fabricated
	 * hang is a false statement.
	 */
	queryEvents(query: EventQuery = {}): StoredEventRow[] {
		const conditions: string[] = [];
		const params: (string | number)[] = [];
		if (query.userId !== undefined) {
			conditions.push("c.user_id = ?");
			params.push(query.userId);
		}
		if (query.conversationId !== undefined) {
			conditions.push("e.conversation_id = ?");
			params.push(query.conversationId);
		}
		if (query.types !== undefined && query.types.length > 0) {
			conditions.push(`e.type IN (${query.types.map(() => "?").join(", ")})`);
			params.push(...query.types);
		}
		if (query.since !== undefined) {
			conditions.push("e.ts >= ?");
			params.push(query.since);
		}
		if (query.until !== undefined) {
			conditions.push("e.ts < ?");
			params.push(query.until);
		}
		const where = conditions.length > 0 ? `WHERE ${conditions.join(" AND ")}` : "";
		params.push(query.limit ?? DEFAULT_EVENT_QUERY_LIMIT);

		const rows = this.db
			.prepare(
				`SELECT * FROM (
					 SELECT e.id AS event_id, e.conversation_id, c.user_id, e.turn, e.ts, e.payload
					 FROM events e
					 JOIN conversations c ON c.conversation_id = e.conversation_id
					 ${where}
					 ORDER BY e.ts DESC, e.id DESC
					 LIMIT ?
				 )
				 ORDER BY ts, event_id`,
			)
			.all(...params) as unknown as {
			conversation_id: string;
			user_id: string;
			turn: number;
			ts: string;
			payload: string;
		}[];

		return rows.map((row) => ({
			conversation_id: row.conversation_id,
			user_id: row.user_id,
			turn: row.turn,
			ts: row.ts,
			event: JSON.parse(row.payload) as SeatEvent,
		}));
	}

	/**
	 * Persist one event on its own, outside a turn's commit.
	 *
	 * WHY THIS EXISTS SEPARATELY FROM `persistTurn`. Everything a turn produces
	 * is written when the turn ends, because the seat is what produces it. A view
	 * outcome is produced by the BROWSER and can arrive at any time — including
	 * after the turn is closed, when a person finally accepts a Follow offer that
	 * has been sitting on screen. There is no open turn to attach that to, and
	 * holding it until the next one would mean losing it on a restart and
	 * reordering it behind events that happened later.
	 *
	 * `turn` is the turn the caller says this belongs to. For a late outcome that
	 * is the turn in progress when it arrived, which is where a reader looking at
	 * the timeline would expect to find it; the link back to the act it answers
	 * is the `tool_call_id` in its payload, not its position.
	 *
	 * Returns false when the conversation does not exist — the foreign key would
	 * reject the insert, and a caller writing outcomes for a deleted conversation
	 * should be told rather than have the failure surface as a thrown constraint.
	 */
	appendEvent(conversationId: string, stored: StoredEvent): boolean {
		if (!this.getConversation(conversationId)) return false;
		this.db
			.prepare(`INSERT INTO events (conversation_id, turn, ts, type, payload) VALUES (?, ?, ?, ?, ?)`)
			.run(
				conversationId,
				stored.turn,
				stored.ts,
				stored.event.type,
				JSON.stringify(stored.event),
			);
		return true;
	}

	/**
	 * Persist one finished (or aborted) turn atomically: transcript snapshot,
	 * durable events, usage accumulation, turn count, and the title if it is
	 * the first one. A crash between turns therefore never leaves a transcript
	 * ahead of its events or totals ahead of either.
	 */
	persistTurn(input: TurnPersistInput): void {
		const now = new Date().toISOString();
		this.db.exec("BEGIN IMMEDIATE");
		try {
			this.db
				.prepare(
					`INSERT INTO transcripts (conversation_id, messages) VALUES (?, ?)
					 ON CONFLICT(conversation_id) DO UPDATE SET messages = excluded.messages`,
				)
				.run(input.conversationId, JSON.stringify(input.messages));

			const insertEvent = this.db.prepare(
				`INSERT INTO events (conversation_id, turn, ts, type, payload) VALUES (?, ?, ?, ?, ?)`,
			);
			for (const stored of input.events) {
				insertEvent.run(
					input.conversationId,
					stored.turn,
					stored.ts,
					stored.event.type,
					JSON.stringify(stored.event),
				);
			}

			this.db
				.prepare(
					`UPDATE conversations SET
						turns = ?,
						updated_at = ?,
						title = COALESCE(title, ?),
						usage_input = usage_input + ?,
						usage_output = usage_output + ?,
						usage_cache_read = usage_cache_read + ?,
						usage_cache_write = usage_cache_write + ?,
						usage_reasoning = usage_reasoning + ?,
						usage_total_tokens = usage_total_tokens + ?,
						usage_cost_total = usage_cost_total + ?
					 WHERE conversation_id = ?`,
				)
				.run(
					input.turns,
					now,
					input.titleCandidate ?? null,
					input.usageDelta.input,
					input.usageDelta.output,
					input.usageDelta.cache_read,
					input.usageDelta.cache_write,
					input.usageDelta.reasoning,
					input.usageDelta.total_tokens,
					input.usageDelta.cost_total,
					input.conversationId,
				);
			this.db.exec("COMMIT");
		} catch (error) {
			try {
				this.db.exec("ROLLBACK");
			} catch {
				// Rollback failure must not mask the original error.
			}
			throw error;
		}
	}
}

/** Title derivation: the first user message, whitespace-collapsed and cut at a
 *  word boundary. Real content, never invented — a conversation with no user
 *  text yet keeps a null title and the UI shows its created time instead. */
export function deriveTitle(firstUserText: string): string | undefined {
	const collapsed = firstUserText.replace(/\s+/g, " ").trim();
	if (!collapsed) return undefined;
	if (collapsed.length <= 80) return collapsed;
	const cut = collapsed.slice(0, 80);
	const lastSpace = cut.lastIndexOf(" ");
	return `${cut.slice(0, lastSpace > 40 ? lastSpace : 80)}…`;
}
