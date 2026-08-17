/**
 * Recorded work, as something an agent can pick up and be held to.
 *
 * THE SECOND CONSUMER of the same tool layer as view-tools.ts, and it is here
 * for the same reason: the model could already read todos (they ride along on a
 * recall) but it could not touch one. Listing, claiming, commenting and updating
 * are the four verbs that turn "the model knows about this task" into "the model
 * did something about this task and the record says so".
 *
 * ACCOUNTABILITY IS A COMMENT, BECAUSE THE MODEL HAS NOWHERE ELSE TO SIGN.
 * `Todo` (src/memory/types.rs) has no assignee, no executor and no actor field
 * of any kind; the only per-todo place a name can be written is
 * `TodoComment.author`, which is free text and defaults to the user's own id.
 * So every mutation below also writes a comment authored by the model, in the
 * form `agent:<provider>/<model-id>`. Without it a todo moved by the model is
 * indistinguishable from one the person moved by hand, and "which tool was used
 * when" — the claim this product makes — would stop at the workbench door.
 *
 * WHAT IS NOT DONE HERE, deliberately: no new status, no executor field, no
 * lease. Those are schema changes to the Rust model and belong in a Rust change
 * with a migration, not smuggled in as a convention a TypeScript file invented.
 * The report at the end of this work names precisely what is missing.
 *
 * Every rule that can be a pure function is one, above the tool definitions.
 */

import type { AgentTool, AgentToolResult } from "@earendil-works/pi-agent-core";
import { Type } from "@earendil-works/pi-ai";
import type { BackendTodo, ShodhBackend, TodoStatus } from "./backend.js";
import type { ModelRef } from "./events.js";
import { composeToolDescription } from "./tool-descriptions.js";

/** src/memory/types.rs `TodoStatus`, in workflow order. */
export const TODO_STATUSES = ["backlog", "todo", "in_progress", "blocked", "done", "cancelled"] as const;

/** Statuses from which no work can be claimed: the todo is already settled. */
const TERMINAL_STATUSES: ReadonlySet<string> = new Set<TodoStatus>(["done", "cancelled"]);

/**
 * The canonical spellings only.
 *
 * `TodoStatus::from_str_loose` accepts a wider set of synonyms ("doing",
 * "waiting", "wont_do"), and this deliberately does not: a tool that silently
 * accepts six spellings of one status teaches the model six spellings, and the
 * one it picks next will be the one the next endpoint rejects. Unknown input is
 * an error naming the six, which is a thing the model can act on.
 */
export function parseTodoStatus(raw: string): TodoStatus | null {
	const value = raw.trim().toLowerCase();
	return (TODO_STATUSES as readonly string[]).includes(value) ? (value as TodoStatus) : null;
}

/**
 * How the model signs its work.
 *
 * The model reference and not the string "agent": two different models moving
 * the same todo on two different days is exactly the distinction an audit is
 * for, and the seat already knows which one is running.
 */
export function agentAuthor(model: ModelRef): string {
	return `agent:${model.provider}/${model.id}`;
}

/**
 * The id a person and a model both use — `Todo::short_id()`, reproduced.
 *
 * NOT SERVED BY THE API. `short_id` is a Rust method, not a field, so
 * `/api/todos` returns `seq_num` and `project_prefix` and leaves the assembly to
 * the caller (only the recall payload's `RecallTodo` carries a computed one).
 * Getting it wrong is not cosmetic: it is the handle the model quotes back to
 * the user and passes to the next tool call, and `find_todo_by_prefix` resolves
 * "BOLT-7" by splitting exactly this shape.
 *
 * The legacy branch is the Rust fallback verbatim — `TodoId::short()` is
 * "SHO-" followed by the first four characters of the uuid — and it exists
 * because todos written before sequential numbering still have `seq_num` 0.
 */
export function shortIdOf(todo: Pick<BackendTodo, "id" | "seq_num" | "project_prefix">): string {
	if (todo.seq_num > 0) return `${todo.project_prefix ?? "SHO"}-${todo.seq_num}`;
	return `SHO-${todo.id.slice(0, 4)}`;
}

/**
 * One todo, as one line the model can read and cite.
 *
 * PROJECTED, NEVER FORWARDED WHOLE. The backend serializes the full `Todo`,
 * including a 384-float `embedding` per row on this branch (the strip landed on
 * `fix/todos-api-defects`, which is not merged here) — about 287KB for fifty
 * todos, none of it readable by a model. Everything a tool returns becomes
 * context, so the projection is the point, not a nicety.
 *
 * The blocked half is always both halves: `blocked_on` is free text about
 * who or what ("waiting on the vendor") and `blocked_by` is real todo
 * references. A line showing one and hiding the other makes a blocked task look
 * either unexplained or unblocked.
 */
export function formatTodoLine(todo: BackendTodo): string {
	const parts = [`[${shortIdOf(todo)}] (${todo.status}, ${todo.priority}) ${todo.content}`];
	const project = todo.project_prefix ?? todo.project;
	if (project) parts.push(`project ${project}`);
	if (todo.contexts.length > 0) parts.push(todo.contexts.join(" "));
	if (todo.due_date) parts.push(`due ${todo.due_date}`);
	if (todo.parent_id) parts.push("subtask");
	if (todo.blocked_on) parts.push(`blocked on: ${todo.blocked_on}`);
	if (todo.blocked_by.length > 0) {
		parts.push(`blocked by ${todo.blocked_by.length} todo(s)`);
	}
	return parts.join(" · ");
}

/** The comments a model wrote, so a second agent can see the first one's claim. */
export function agentComments(todo: BackendTodo): { author: string; content: string; created_at: string }[] {
	return todo.comments
		.filter((comment) => comment.author.startsWith("agent:"))
		.map((comment) => ({ author: comment.author, content: comment.content, created_at: comment.created_at }));
}

/**
 * Why a claim cannot proceed, or null when it can.
 *
 * Two refusals, and the second is the one that matters. A settled todo is not
 * work; and a todo another agent has already claimed is work someone else may
 * be doing RIGHT NOW. The model cannot know that from `status` alone — there is
 * no assignee — so the prior claim comment is the only evidence, and it is
 * surfaced rather than overridden.
 */
export function claimRefusal(todo: BackendTodo, author: string): string | null {
	if (TERMINAL_STATUSES.has(todo.status)) {
		return `[${shortIdOf(todo)}] is already ${todo.status}. There is no work to claim.`;
	}
	const others = agentComments(todo).filter((comment) => comment.author !== author);
	const last = others[others.length - 1];
	if (todo.status === "in_progress" && last !== undefined) {
		return (
			`[${shortIdOf(todo)}] is already in progress and was claimed by ${last.author} at ${last.created_at}: ` +
			`"${last.content}". This model has no way to take it over safely — the todo has no assignee field, so a ` +
			"second claim would overwrite nothing and leave two agents believing they own it. Ask the user."
		);
	}
	return null;
}

/**
 * What the model is told a claim actually achieved.
 *
 * A CLAIM IS TWO CALLS AND THEY CAN DISAGREE. The status change is one request
 * and the signing comment is another; the backend offers no way to do both at
 * once. So the report states each outcome separately: a claim whose comment
 * failed has moved the todo to in_progress with NOBODY recorded as having moved
 * it, and a model told "claimed" would go on to work under an attribution that
 * does not exist.
 */
export function composeClaimReport(input: {
	shortId: string;
	previousStatus: string;
	statusChanged: boolean;
	commentError: string | null;
	author: string;
}): string {
	const lines: string[] = [];
	lines.push(
		input.statusChanged
			? `[${input.shortId}] moved ${input.previousStatus} → in_progress.`
			: `[${input.shortId}] was already in_progress; status unchanged.`,
	);
	if (input.commentError === null) {
		lines.push(`Claim recorded on the todo as ${input.author}.`);
	} else {
		lines.push(
			`WARNING: the status changed but the claim comment FAILED (${input.commentError}). ` +
				`Nothing on [${input.shortId}] records that you took it — the todo has no assignee field, so the ` +
				"comment was the only attribution. Retry comment_on_todo before doing the work, or tell the user.",
		);
	}
	return lines.join("\n");
}

/**
 * The filters that were actually in force, in the model's own vocabulary.
 *
 * Echoed back rather than assumed, because the model does not reliably remember
 * what it sent by the time it reads the answer, and because one of these was
 * never sent at all: `list_todos` defaults to excluding settled work, so a board
 * whose every todo is `done` comes back empty from a call that named no status.
 * That default is the single most likely cause of a surprising empty listing and
 * the one the model has no way to see.
 */
export function describeTodoFilters(params: {
	status?: readonly string[];
	project?: string;
	context?: string;
	priority?: string;
	query?: string;
}): string[] {
	const applied: string[] = [];
	if (params.status && params.status.length > 0) applied.push(`status ${params.status.join(" or ")}`);
	else applied.push("status: anything not settled (done and cancelled were excluded — you did not ask for them)");
	if (params.project) applied.push(`project "${params.project}"`);
	if (params.context) applied.push(`context "${params.context}"`);
	if (params.priority) applied.push(`priority ${params.priority}`);
	if (params.query) applied.push(`text matching "${params.query}"`);
	return applied;
}

/**
 * What an empty listing says, and what to do about it.
 *
 * The recovery step is chosen by whether anything narrowed the query beyond the
 * settled-work default: with extra filters the useful next move is to drop them,
 * and with none there is genuinely nothing to list and retrying is a waste of a
 * turn. Telling the model to "try broader filters" when it used none is how a
 * tool teaches a loop.
 */
export function composeEmptyListingReport(applied: readonly string[]): string {
	const narrowed = applied.length > 1;
	return (
		`No todos matched. Filters in force: ${applied.join("; ")}.\n` +
		(narrowed
			? "Drop the narrowing filters and list again before concluding there is no such work."
			: "Nothing is filtered but settled work, so this profile has no open todos — say so rather than searching again.")
	);
}

/**
 * A `[mem:xxxxxxxx]` citation, reduced to the eight characters that identify it.
 *
 * THE MODEL ONLY EVER HAS SHORT IDS. Every surface that shows it a memory —
 * recall results, the auto-surfaced block, its own writes — prints the first
 * eight hex characters of the uuid, because that is the citation contract. The
 * backend's todo-to-memory link, by contrast, verifies FULL uuids and rejects
 * anything else. So a `create_todo` that took memory ids as typed would reject
 * every id the model is capable of producing.
 *
 * Both spellings are accepted because both are things the model has seen: the
 * bracketed citation is what it writes into prose, the bare eight characters are
 * what it reads out of a listing.
 */
export function memoryCitationKey(raw: string): string | null {
	const trimmed = raw.trim();
	const bracketed = /^\[mem:([0-9a-fA-F]{8})\]$/.exec(trimmed);
	if (bracketed) return bracketed[1]!.toLowerCase();
	return /^[0-9a-fA-F]{8}$/.test(trimmed) ? trimmed.toLowerCase() : null;
}

/** One requested memory link, resolved or refused. */
export interface MemoryLinkOutcome {
	/** Full uuids, ready for the backend's verification. */
	ids: string[];
	/** What the model typed that this seat could not turn into a memory. */
	unknown: string[];
}

/**
 * Turn the model's citations into memory uuids, refusing anything it has not
 * been shown.
 *
 * THE RESTRICTION IS THE FEATURE. `known` holds only the memories surfaced in
 * THIS run — recalled, auto-surfaced, or written by the model itself — which is
 * exactly the set the citation contract already polices in an answer. A todo
 * linked to a memory the model never read is a "why does this task exist" chain
 * whose first link is a guess, and the backend cannot catch it: it verifies that
 * the uuid exists, not that anybody looked at it.
 */
export function resolveMemoryLinks(
	raw: readonly string[],
	known: (shortId: string) => string | null,
): MemoryLinkOutcome {
	const ids: string[] = [];
	const unknown: string[] = [];
	const seen = new Set<string>();
	for (const value of raw) {
		const key = memoryCitationKey(value);
		const full = key === null ? null : known(key);
		if (full === null) {
			unknown.push(value.trim());
			continue;
		}
		if (seen.has(full)) continue;
		seen.add(full);
		ids.push(full);
	}
	return { ids, unknown };
}

/**
 * What the model is told a creation achieved.
 *
 * THE SHORT ID IS THE POINT. It is the handle every other todo tool takes, and a
 * creation that did not return one would force a `list_todos` round trip before
 * the model could do anything with the thing it just made. The signature is
 * reported separately for the same reason a claim's is: creation and signing are
 * two calls, and a todo created with nobody recorded as having created it is
 * indistinguishable from one the person typed.
 */
export function composeCreateReport(input: {
	shortId: string;
	linked: number;
	unknownLinks: readonly string[];
	commentError: string | null;
	author: string;
}): string {
	const lines = [`Created [${input.shortId}].`];
	if (input.linked > 0) {
		lines.push(
			`Linked to ${input.linked} ${input.linked === 1 ? "memory" : "memories"}, so the todo records why it exists.`,
		);
	}
	if (input.unknownLinks.length > 0) {
		lines.push(
			`NOT linked: ${input.unknownLinks.join(", ")} — these are not memories you have been shown in this ` +
				"conversation. Link only ids from a recall result, the surfaced-memory block, or your own writes; " +
				"recall the memory first if you want the link.",
		);
	}
	lines.push(
		input.commentError === null
			? `Recorded on the todo as ${input.author}.`
			: `WARNING: the todo exists but the note saying you created it FAILED (${input.commentError}). ` +
					`Nothing on [${input.shortId}] distinguishes it from a todo the user typed. Retry ` +
					"comment_on_todo, or tell the user it is unattributed.",
	);
	return lines.join("\n");
}

export interface TodoToolContext {
	backend: ShodhBackend;
	userId: string;
	/** Read at call time: the conversation's model can change mid-session. */
	getModel(): ModelRef;
	/**
	 * The full uuid for an 8-character citation the model has actually been
	 * shown this run, or null.
	 *
	 * Supplied by the conversation rather than looked up, because "has the model
	 * seen this" is not a question the backend can answer — it knows which
	 * memories exist, not which ones were put in front of anybody.
	 */
	resolveMemoryCitation(shortId: string): string | null;
}

const TODO_ID_DESCRIPTION =
	'Short id as shown in listings ("BOLT-7", "SHO-3"), a bare sequence number, or a uuid prefix.';

const listParameters = Type.Object({
	status: Type.Optional(
		Type.Array(
			Type.Union(TODO_STATUSES.map((status) => Type.Literal(status))),
			{ maxItems: TODO_STATUSES.length, description: "Statuses to include. Default: everything not settled." },
		),
	),
	project: Type.Optional(Type.String({ minLength: 1, maxLength: 100, description: "Project name or prefix." })),
	context: Type.Optional(Type.String({ minLength: 1, maxLength: 100, description: "GTD context, e.g. @computer." })),
	priority: Type.Optional(
		Type.Union(
			(["urgent", "high", "medium", "low", "none"] as const).map((priority) => Type.Literal(priority)),
		),
	),
	query: Type.Optional(Type.String({ minLength: 1, maxLength: 500, description: "Free-text filter over content." })),
	limit: Type.Optional(Type.Integer({ minimum: 1, maximum: 50, description: "Maximum todos (default 20)." })),
});

const TODO_PRIORITIES = ["urgent", "high", "medium", "low", "none"] as const;

const createParameters = Type.Object({
	content: Type.String({
		minLength: 3,
		maxLength: 500,
		description:
			"The work, as one actionable line. A GTD context written inline (\"@computer\", \"@phone\") is extracted " +
			"automatically, so write it the way the user would say it.",
	}),
	why: Type.String({
		minLength: 10,
		maxLength: 1000,
		description:
			"Why this task exists, in your own words. Recorded on the todo under your model identity — it is the only " +
			"thing that distinguishes a todo you created from one the user typed.",
	}),
	priority: Type.Optional(
		Type.Union(TODO_PRIORITIES.map((priority) => Type.Literal(priority)), {
			description: "Default medium. Judge it from what the user said, not from your own sense of urgency.",
		}),
	),
	project: Type.Optional(
		Type.String({
			minLength: 1,
			maxLength: 100,
			description: "Project name. An unknown name CREATES that project, so take it from a list_todos result.",
		}),
	),
	due_date: Type.Optional(
		Type.String({
			minLength: 3,
			maxLength: 60,
			description: 'When it is due — "2026-09-01", "tomorrow", "next friday". Only when the user gave one.',
		}),
	),
	because_of: Type.Optional(
		Type.Array(Type.String({ minLength: 8, maxLength: 20 }), {
			maxItems: 8,
			description:
				"Memory ids that motivated this task, as [mem:<id>] or the bare 8 characters. Only ids surfaced to you " +
				"in this conversation are accepted; anything else is reported back unlinked rather than guessed at.",
		}),
	),
});

const claimParameters = Type.Object({
	todo_id: Type.String({ minLength: 1, maxLength: 100, description: TODO_ID_DESCRIPTION }),
	plan: Type.String({
		minLength: 10,
		maxLength: 1000,
		description:
			"What you intend to do about this todo. Recorded on it under your model identity — it is the only record " +
			"that you took the work, and the next agent to look will read it.",
	}),
});

const updateParameters = Type.Object({
	todo_id: Type.String({ minLength: 1, maxLength: 100, description: TODO_ID_DESCRIPTION }),
	status: Type.Optional(
		Type.Union(TODO_STATUSES.map((status) => Type.Literal(status)), {
			description: "New status. `blocked` requires blocked_on.",
		}),
	),
	priority: Type.Optional(
		Type.Union((["urgent", "high", "medium", "low", "none"] as const).map((p) => Type.Literal(p))),
	),
	blocked_on: Type.Optional(
		Type.String({
			minLength: 3,
			maxLength: 500,
			description: "Who or what the work is waiting on, in words. Required when status is `blocked`.",
		}),
	),
	note: Type.String({
		minLength: 10,
		maxLength: 1000,
		description: "Why you are making this change. Recorded on the todo under your model identity.",
	}),
});

const commentParameters = Type.Object({
	todo_id: Type.String({ minLength: 1, maxLength: 100, description: TODO_ID_DESCRIPTION }),
	content: Type.String({ minLength: 3, maxLength: 4000, description: "The comment, markdown allowed." }),
	kind: Type.Optional(
		Type.Union(
			(["comment", "progress", "resolution", "activity"] as const).map((kind) => Type.Literal(kind)),
			{ description: "Comment type (default progress for agent notes)." },
		),
	),
});

function textResult<T>(text: string, details: T): AgentToolResult<T> {
	return { content: [{ type: "text", text }], details };
}

/** Fetch a todo or throw a message the model can act on. */
async function requireTodo(context: TodoToolContext, todoId: string): Promise<BackendTodo> {
	const response = await context.backend.getTodo(context.userId, todoId);
	if (!response.todo) {
		throw new Error(
			`No todo matches "${todoId}". Use list_todos to see the ids that exist; ids look like "BOLT-7" or "SHO-3".`,
		);
	}
	return response.todo;
}

export function createTodoTools(context: TodoToolContext): AgentTool<any>[] {
	const listTool: AgentTool<typeof listParameters> = {
		name: "list_todos",
		label: "List todos",
		description: composeToolDescription("list_todos", {
			does:
				"Lists the work recorded against this profile, one line each, with status, priority, project, contexts, " +
				"due date and blockers.",
			useWhen:
				"Read it before claiming, updating or commenting on anything: the short ids it returns are the handles " +
				"every other todo tool takes, and it is the only way to learn what those ids are.",
			notFor:
				"It is not a search over memory — todos are recorded work, not what was remembered, and recall_memory " +
				"covers the latter. Do not call it repeatedly within one turn to watch for change; nothing moves the " +
				"board but you and the user.",
			returns:
				"Settled work is excluded unless you name `done` or `cancelled` in `status` explicitly. It does NOT " +
				"return a todo's comments, so it cannot tell you whether another model has already claimed one — " +
				"claim_todo performs that check and refuses when it finds a prior claim.",
		}),
		parameters: listParameters,
		execute: async (_toolCallId, params) => {
			const response = await context.backend.listTodos({
				userId: context.userId,
				status: params.status,
				project: params.project,
				context: params.context,
				priority: params.priority,
				query: params.query,
				limit: params.limit ?? 20,
				// Settled work is included only when explicitly asked for, so the
				// default listing is the work that is still work.
				includeCompleted: params.status?.some((status) => TERMINAL_STATUSES.has(status)) ?? false,
			});
			if (response.todos.length === 0) {
				// AN EMPTY LISTING IS A DEAD END UNLESS IT SAYS WHAT IT ASKED FOR.
				// "No todos match those filters" leaves the model with no way to
				// tell an empty board from an over-narrow query, and the two call
				// for opposite next moves — one is an answer, the other is a retry.
				// So the filters that were actually in force are echoed back, and
				// the default that is easiest to forget is named explicitly.
				return textResult(composeEmptyListingReport(describeTodoFilters(params)), { count: 0 });
			}
			const lines = [`${response.todos.length} todo(s):`];
			for (const todo of response.todos) lines.push(`- ${formatTodoLine(todo)}`);
			return textResult(lines.join("\n"), { count: response.todos.length });
		},
	};

	/**
	 * The verb the surface was missing.
	 *
	 * WHY ITS ABSENCE WAS NOT NEUTRAL. The model could already create todos — the
	 * bridged MCP `add_todo` was left unfiltered precisely because there was "no
	 * native equivalent" (conversation.ts) — and that path writes no attribution
	 * at all. So creation was the one mutation on this surface that arrived
	 * anonymous, on a surface whose entire argument is that every mutation carries
	 * the model's identity. A native verb that signs its work closes that, and
	 * `add_todo` joins the filtered list.
	 */
	const createTool: AgentTool<typeof createParameters> = {
		name: "create_todo",
		label: "Create a todo",
		description: composeToolDescription("create_todo", {
			does:
				"Records a new piece of work against this profile and signs it with your model identity, optionally " +
				"linking the memories that motivated it.",
			useWhen:
				"Use it when the conversation surfaces work that should outlive it — something the user said they need " +
				"to do, or a follow-up your own findings imply and they agreed to. Say why in `why`: it is the only " +
				"record that this todo came from you rather than from their own hand.",
			notFor:
				"Do not create work the user did not ask for and has not agreed to; a board filling with an " +
				"assistant's suggestions is worse than an empty one. Check list_todos first — nothing here detects a " +
				"duplicate, so calling twice makes two todos. And do not use it to store a FACT or a decision, which " +
				"is remember_memory's job; a todo is something to be done.",
			returns:
				"The new short id, which the other todo tools take immediately. `because_of` accepts only memory ids " +
				"you have actually been shown in this conversation — anything else comes back named and unlinked — and " +
				"the backend rejects the whole call if a linked memory or an unknown project reference does not " +
				"resolve. The todo is created open and unclaimed: claim_todo is a separate act.",
		}),
		parameters: createParameters,
		execute: async (_toolCallId, params) => {
			const author = agentAuthor(context.getModel());
			const { ids, unknown } = resolveMemoryLinks(params.because_of ?? [], context.resolveMemoryCitation);

			const response = await context.backend.createTodo({
				userId: context.userId,
				content: params.content,
				priority: params.priority,
				project: params.project,
				dueDate: params.due_date,
				// Omitted entirely when empty rather than sent as [], so the
				// handler's own default path runs.
				relatedMemoryIds: ids.length > 0 ? ids : undefined,
			});
			if (!response.todo) {
				throw new Error(
					"The backend accepted the request and returned no todo, so nothing was created and there is no id " +
						"to work with. Retry once; if it fails again, tell the user the task was not recorded rather " +
						"than assuming it was.",
				);
			}
			const shortId = shortIdOf(response.todo);

			// The todo exists. A signing failure must not throw — that would tell
			// the model nothing was created while the board says otherwise, which
			// is the same defect claim_todo avoids for the same reason.
			let commentError: string | null = null;
			try {
				await context.backend.addTodoComment({
					userId: context.userId,
					todoId: response.todo.id,
					content: `Created by ${author}. Why: ${params.why}`,
					author,
					commentType: "activity",
				});
			} catch (error) {
				commentError = error instanceof Error ? error.message : String(error);
			}

			return textResult(
				composeCreateReport({ shortId, linked: ids.length, unknownLinks: unknown, commentError, author }),
				{ todo_id: shortId, linked_memory_ids: ids, unlinked: unknown, signed: commentError === null },
			);
		},
	};

	const claimTool: AgentTool<typeof claimParameters> = {
		name: "claim_todo",
		label: "Claim a todo",
		description: composeToolDescription("claim_todo", {
			does:
				"Takes a todo: moves it to in_progress and records on the todo itself, under your model identity, that " +
				"you took it and what you intend to do about it.",
			useWhen:
				"Claim a todo before doing any of the work it describes, so that a second agent looking at the same " +
				"board can see the work is spoken for and what the plan is.",
			notFor:
				"Do not claim work you are not about to start in this turn, and do not claim a todo merely so you can " +
				"comment on it — comment_on_todo needs no claim. There is no way to release a claim from here, so one " +
				"you abandon leaves the todo sitting in_progress with your name on it.",
			returns:
				"Confirmation that the status moved and that the claim was recorded — separately, because they are two " +
				"backend calls that can disagree, and a claim whose comment failed has moved the work with nobody " +
				"recorded as having moved it. It refuses outright when the todo is already settled, or when another " +
				"model's claim is already on it: this system has no assignee field, so a second claim would overwrite " +
				"nothing and leave two agents each believing they own the work.",
		}),
		parameters: claimParameters,
		execute: async (_toolCallId, params) => {
			const author = agentAuthor(context.getModel());
			const todo = await requireTodo(context, params.todo_id);

			const refusal = claimRefusal(todo, author);
			if (refusal) throw new Error(refusal);

			const shortId = shortIdOf(todo);
			const previousStatus = todo.status;
			const statusChanged = todo.status !== "in_progress";
			if (statusChanged) {
				await context.backend.updateTodo(context.userId, params.todo_id, { status: "in_progress" });
			}

			// The status write has already happened. A failure here must not
			// undo it by throwing — the todo really is in progress, and a thrown
			// tool call would tell the model nothing happened while the board
			// says otherwise. It is reported instead.
			let commentError: string | null = null;
			try {
				await context.backend.addTodoComment({
					userId: context.userId,
					todoId: params.todo_id,
					content: `Claimed by ${author}. Plan: ${params.plan}`,
					author,
					commentType: "activity",
				});
			} catch (error) {
				commentError = error instanceof Error ? error.message : String(error);
			}

			return textResult(
				composeClaimReport({ shortId, previousStatus, statusChanged, commentError, author }),
				{ todo_id: shortId, status: "in_progress", claim_recorded: commentError === null },
			);
		},
	};

	const updateTool: AgentTool<typeof updateParameters> = {
		name: "update_todo",
		label: "Update a todo",
		description: composeToolDescription("update_todo", {
			does:
				"Changes a todo's status, priority or blocker and records why, as a note on the todo under your model " +
				"identity, so that the change is attributable to you rather than to nobody.",
			useWhen:
				"Use it when the state of the work has actually changed: you finished it, you started it, or you found " +
				"out it cannot proceed. Marking it `blocked` requires naming what it is waiting on, because a task that " +
				"stopped for no recorded reason is worse than one nobody touched.",
			notFor:
				"Do not use it to leave a note without changing anything — that is comment_on_todo, and this tool " +
				"refuses a note-only call. Do not set `done` speculatively: completion is not reversible from here, and " +
				"it stamps the completion time, rolls a recurring todo over to its next occurrence, and unblocks " +
				"everything that was waiting on it.",
			returns:
				"What actually changed, field by field, plus the consequences of a completion — the next occurrence it " +
				"created and the todos it unblocked, each by short id. It reports separately whether the explanatory " +
				"note was recorded, because a change nobody can attribute is a hole in the same audit trail this tool " +
				"exists to keep whole.",
		}),
		parameters: updateParameters,
		execute: async (_toolCallId, params) => {
			const author = agentAuthor(context.getModel());
			const todo = await requireTodo(context, params.todo_id);
			const shortId = shortIdOf(todo);

			let status: TodoStatus | undefined;
			if (params.status !== undefined) {
				const parsed = parseTodoStatus(params.status);
				if (!parsed) {
					throw new Error(
						`"${params.status}" is not a status. Valid values: ${TODO_STATUSES.join(", ")}.`,
					);
				}
				status = parsed;
			}

			// A blocker with no reason is a task that has stopped for no recorded
			// cause, and `blocked_on` is the only field that can hold one in
			// words. The backend accepts the status without it; this does not.
			if (status === "blocked" && !params.blocked_on && !todo.blocked_on) {
				throw new Error(
					`Refusing to block [${shortId}] without saying what it is waiting on. Pass blocked_on.`,
				);
			}

			const changes: string[] = [];
			if (status === "done") {
				// ROUTED THROUGH complete, NOT through update. `update_todo` in
				// src/handlers/todos.rs assigns the status and stops: it does not
				// stamp `completed_at`, does not roll a recurring todo over to its
				// next occurrence, and does not compute which dependents this
				// unblocks. A todo "finished" through the update path is a todo
				// whose recurrence silently stopped.
				const completion = await context.backend.completeTodo(context.userId, params.todo_id);
				changes.push("completed");
				if (completion.next_recurrence) {
					changes.push(`next occurrence created as [${shortIdOf(completion.next_recurrence)}]`);
				}
				if (completion.unblocked.length > 0) {
					changes.push(
						`unblocked ${completion.unblocked.map((other) => `[${shortIdOf(other)}]`).join(", ")}`,
					);
				}
			} else if (status !== undefined || params.priority !== undefined || params.blocked_on !== undefined) {
				await context.backend.updateTodo(context.userId, params.todo_id, {
					status,
					priority: params.priority,
					blockedOn: params.blocked_on,
				});
				if (status !== undefined) changes.push(`${todo.status} → ${status}`);
				if (params.priority !== undefined) changes.push(`priority ${todo.priority} → ${params.priority}`);
				if (params.blocked_on !== undefined) changes.push(`blocked on: ${params.blocked_on}`);
			} else {
				throw new Error(
					`Nothing to change on [${shortId}]: give a status, a priority, or blocked_on. ` +
						"A note by itself belongs on comment_on_todo.",
				);
			}

			let commentError: string | null = null;
			try {
				await context.backend.addTodoComment({
					userId: context.userId,
					todoId: params.todo_id,
					content: `${changes.join("; ")} — ${params.note}`,
					author,
					commentType: status === "done" ? "resolution" : "progress",
				});
			} catch (error) {
				commentError = error instanceof Error ? error.message : String(error);
			}

			const lines = [`[${shortId}] ${changes.join("; ")}.`];
			lines.push(
				commentError === null
					? `Recorded on the todo as ${author}.`
					: `WARNING: the change was applied but the note FAILED to record (${commentError}). ` +
							"Nothing attributes this change to you.",
			);
			return textResult(lines.join("\n"), { todo_id: shortId, changes, note_recorded: commentError === null });
		},
	};

	const commentTool: AgentTool<typeof commentParameters> = {
		name: "comment_on_todo",
		label: "Comment on a todo",
		description: composeToolDescription("comment_on_todo", {
			does: "Adds a note to a todo — progress, a finding, a reason — recorded under your model identity.",
			useWhen:
				"Use it to leave behind what the next agent or the user would need to know and could not reconstruct: " +
				"what you tried, what you found, why the obvious approach does not work here.",
			notFor:
				"Do not use it to change status, priority or blockers — update_todo does that and records its own " +
				"reason, so a comment describing a change you did not actually make is a false record on the one " +
				"surface the user trusts to be true. It also does not claim the todo; claim_todo does that.",
			returns:
				"Confirmation naming the todo and the identity the comment was signed with. It does not return the " +
				"todo's other comments, so it tells you nothing about what anyone else has written there.",
		}),
		parameters: commentParameters,
		execute: async (_toolCallId, params) => {
			const author = agentAuthor(context.getModel());
			const todo = await requireTodo(context, params.todo_id);
			const response = await context.backend.addTodoComment({
				userId: context.userId,
				todoId: params.todo_id,
				content: params.content,
				author,
				commentType: params.kind ?? "progress",
			});
			if (!response.comment) {
				// Names the consequence, not just the fact. The model's next move
				// after a failed comment depends entirely on what the comment was
				// FOR — a finding can be put in the answer instead, an attribution
				// cannot — and it cannot make that call from "the backend returned
				// none".
				throw new Error(
					`The comment was NOT recorded on [${shortIdOf(todo)}]: the backend accepted the request and ` +
						"returned no comment, so nothing was written. Nothing on the todo says what you found. Retry " +
						"once; if it fails again, put the finding in your answer to the user and tell them it could not " +
						"be filed.",
				);
			}
			return textResult(`Comment recorded on [${shortIdOf(todo)}] as ${author}.`, {
				todo_id: shortIdOf(todo),
				comment_id: response.comment.id,
			});
		},
	};

	// Read, then create, then take, then change, then annotate — the order work
	// actually moves in. Stable, because a tool list that reorders itself between
	// turns costs a prompt-cache hit on every one of them.
	return [listTool, createTool, claimTool, updateTool, commentTool];
}
