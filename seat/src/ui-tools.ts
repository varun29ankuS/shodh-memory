/**
 * The agent operating the screen.
 *
 * Not by driving the DOM. A tool that clicks pixels is brittle, unauditable,
 * and cannot exist for a product that also ships as an embedded binary with no
 * browser attached. These tools emit the same intents the UI's own controls
 * emit, and the client applies them through its router and stores — so "the
 * agent opened the graph" and "the analyst clicked Graph" are the same code
 * path reaching the same state, with one of them recorded.
 *
 * Why this is worth having rather than a party trick: an analyst who asks
 * "what changed in Bangalore" should not then have to go and find the screen
 * themselves. The answer and the view of the answer are one request, and
 * splitting them across a chat panel and a human's mouse is the seam every
 * memory product currently leaves open.
 *
 * These are NOT exempt from policy, and the asymmetry with the memory tools is
 * deliberate. Recall and remember are what the seat IS. Moving an operator's
 * screen is not — it is precisely the capability someone approving a
 * deployment may want to withhold, and `ui_*` is one line in the policy file.
 */

import type { AgentTool, AgentToolResult } from "@earendil-works/pi-agent-core";
import { Type } from "@earendil-works/pi-ai";
import type { SeatEvent, UiView } from "./events.js";

export interface UiToolContext {
	emit(event: SeatEvent): void;
}

/** Every destination in the rail. A closed set, so a name the UI does not have
 *  becomes a tool error the model can see and correct rather than a silent
 *  no-op the operator reads as being ignored. */
const VIEWS = ["briefing", "chat", "recall", "graph", "geo", "anomalies", "tasks", "providers"] as const;

/** Shown to the operator beside the screen change. Not decoration: a view that
 *  changes by itself with no explanation reads as a fault. */
const REASON = Type.String({ description: "One short line, shown to the operator, explaining the change." });

const openParameters = Type.Object({
	view: Type.Union(
		VIEWS.map((v) => Type.Literal(v)),
		{ description: "Destination to open." },
	),
	reason: REASON,
});

const selectMemoryParameters = Type.Object({
	memory_id: Type.String({ description: "Full memory id, as returned by recall." }),
	reason: REASON,
});

const selectProfileParameters = Type.Object({
	profile: Type.String({ description: "Profile name, as listed by the profile switcher." }),
	reason: REASON,
});

function applied(text: string): AgentToolResult<{ applied: boolean }> {
	return { content: [{ type: "text", text }], details: { applied: true } };
}

export function createUiTools(ctx: UiToolContext): AgentTool<any>[] {
	const open: AgentTool<typeof openParameters> = {
		name: "ui_open",
		label: "Open view",
		description:
			"Move the operator's screen to a destination. Use when the answer is easier to see than " +
			"to describe — a graph, a map, a list of anomalies. Say what you opened and why in your " +
			"reply; a screen that changes without explanation is disorienting.",
		parameters: openParameters,
		execute: async (_toolCallId, params) => {
			const view = params.view as UiView;
			if (!(VIEWS as readonly string[]).includes(view)) {
				// Thrown, not returned: the loop surfaces a throw to the model as a
				// tool error it can correct on the next step, which is what a bad
				// destination should be.
				throw new Error(`Unknown view "${view}". One of: ${VIEWS.join(", ")}.`);
			}
			ctx.emit({ type: "ui_command", command: { kind: "open", view }, reason: params.reason });
			return applied(`Opened ${view}.`);
		},
	};

	const selectMemory: AgentTool<typeof selectMemoryParameters> = {
		name: "ui_select_memory",
		label: "Open memory",
		description:
			"Open one memory in the Inspector — the panel showing what it is, when it was recorded, " +
			"and what it connects to. Use it after citing a memory the operator will want to inspect, " +
			"rather than asking them to go and find it.",
		parameters: selectMemoryParameters,
		execute: async (_toolCallId, params) => {
			const id = params.memory_id.trim();
			if (!id) throw new Error("memory_id is required.");
			ctx.emit({ type: "ui_command", command: { kind: "select_memory", memory_id: id }, reason: params.reason });
			return applied(`Opened memory ${id} in the Inspector.`);
		},
	};

	const selectProfile: AgentTool<typeof selectProfileParameters> = {
		name: "ui_select_profile",
		label: "Switch profile",
		description:
			"Switch the active memory profile — the store every view reads from. This changes what " +
			"the operator is looking at wholesale, so name the profile in your reply.",
		parameters: selectProfileParameters,
		execute: async (_toolCallId, params) => {
			const profile = params.profile.trim();
			if (!profile) throw new Error("profile is required.");
			ctx.emit({ type: "ui_command", command: { kind: "select_profile", profile }, reason: params.reason });
			return applied(`Switched to profile ${profile}.`);
		},
	};

	return [open, selectMemory, selectProfile];
}
