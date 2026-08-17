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

export interface TodoToolContext {
	backend: ShodhBackend;
	userId: string;
	/** Read at call time: the conversation's model can change mid-session. */
	getModel(): ModelRef;
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
		description:
			"List the work recorded against this profile, with status, priority, project, blockers and due dates. " +
			"Read this before claiming or updating anything: the short ids it returns are what the other todo tools take.",
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
				return textResult("No todos match those filters.", { count: 0 });
			}
			const lines = [`${response.todos.length} todo(s):`];
			for (const todo of response.todos) lines.push(`- ${formatTodoLine(todo)}`);
			return textResult(lines.join("\n"), { count: response.todos.length });
		},
	};

	const claimTool: AgentTool<typeof claimParameters> = {
		name: "claim_todo",
		label: "Claim a todo",
		description:
			"Take a todo: move it to in_progress and record, on the todo itself, that you took it and what you intend " +
			"to do. Refuses work that is already settled, or already claimed by another model — this system has no " +
			"assignee field, so a second claim cannot be made safely.",
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
		description:
			"Change a todo's status or priority and say why. The note is recorded on the todo under your model " +
			"identity, so the change is attributable. Marking a todo `blocked` requires naming what it is blocked on.",
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
		description:
			"Add a note to a todo — progress, a finding, a reason — recorded under your model identity. " +
			"Use it to leave behind what the next agent or the user would need to know.",
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
				throw new Error(`The comment was not recorded on [${shortIdOf(todo)}]; the backend returned none.`);
			}
			return textResult(`Comment recorded on [${shortIdOf(todo)}] as ${author}.`, {
				todo_id: shortIdOf(todo),
				comment_id: response.comment.id,
			});
		},
	};

	return [listTool, claimTool, updateTool, commentTool];
}
