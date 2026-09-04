import { useEffect, useMemo, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { Download, PanelLeft, PanelRight } from "lucide-react";
import { cn } from "@/lib/utils";
import type { Reachability } from "@/lib/api";
import { changeModel, getConversation, listConversations } from "@/lib/seat/client";
import type { ConversationSummary, SeatReachability } from "@/lib/seat/types";
import { formatCost, formatTokens } from "@/lib/format";
import { type ChatTurn, useChat } from "@/stores/chat";
import { useSession } from "@/stores/session";
import { Button } from "@/components/ui/button";
import { EmptyState } from "@/components/ui/empty-state";
import { Composer } from "./Composer";
import { EgressBadge } from "./EgressBadge";
import { useEgressServers } from "./useEgressServers";
import { costIsReal, useBillingLookup } from "./useBilling";
import { EvidencePanel } from "./EvidencePanel";
import { MessageList } from "./MessageList";
import { ModelPicker } from "./ModelPicker";
import { NewConversation } from "./NewConversation";
import { SessionList } from "./SessionList";

/**
 * The conversation seat — the product's primary surface.
 *
 * Three regions compete for 1440px: sessions, conversation, evidence. The
 * resolution, decided rather than discovered: the conversation is the only
 * region that must always be present; evidence is its peer and holds a real
 * column from 1024px up (it is the differentiator — hiding it to make room
 * for a session list would order the product backwards); the session list is
 * reference material, so it takes an in-flow column only at ≥1280px and
 * overlays below that. At 1440px the ledger is 56 rail + 240 sessions +
 * flexible conversation (~800) + 340 evidence. On a phone everything but the
 * conversation is an overlay.
 */

function exportMarkdown(title: string, turns: ChatTurn[]): void {
  const lines: string[] = [`# ${title}`, ""];
  for (const turn of turns) {
    lines.push(`## You`, "", turn.userText, "");
    for (const op of turn.ops) {
      if (op.type === "memory_recall" && op.scope === "user") {
        lines.push(
          `> recalled ${op.memories.length} memories for “${op.query}” (${op.took_ms}ms)`,
          ...op.memories.map((memory) => `> - [${memory.id.slice(0, 8)}] ${memory.experience.content}`),
          "",
        );
      } else if (op.type === "proactive_context" && op.memories.length > 0) {
        lines.push(
          `> auto-surfaced ${op.memories.length} memories`,
          ...op.memories.map((memory) => `> - [${memory.id.slice(0, 8)}] ${memory.content}`),
          "",
        );
      }
    }
    if (turn.assistantText) {
      const model = turn.usage ? ` (${turn.usage.model.name})` : "";
      lines.push(`## Assistant${model}`, "", turn.assistantText, "");
    }
  }
  const blob = new Blob([lines.join("\n")], { type: "text/markdown;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = `${title.replace(/[^\w\s-]+/g, "").trim().replace(/\s+/g, "-").slice(0, 60) || "conversation"}.md`;
  anchor.click();
  URL.revokeObjectURL(url);
}

export function ChatView({ reach, seat }: { reach: Reachability; seat: SeatReachability }) {
  // Every remote tool server is another way out of this machine; the badge
  // has to count them, not just the model.
  const egressServers = useEgressServers();
  const queryClient = useQueryClient();
  const profile = useSession((s) => s.profile);
  const activeId = useChat((s) => s.activeId);
  const setActive = useChat((s) => s.setActive);
  const adoptDetail = useChat((s) => s.adoptDetail);
  const send = useChat((s) => s.send);
  const setModel = useChat((s) => s.setModel);
  const convo = useChat((s) => (activeId ? (s.conversations[activeId] ?? null) : null));
  const evidenceOpen = useChat((s) => s.evidenceOpen);
  const toggleEvidence = useChat((s) => s.toggleEvidence);
  const sessionsOpen = useChat((s) => s.sessionsOpen);
  const toggleSessions = useChat((s) => s.toggleSessions);

  // Overlay variants for narrow viewports; independent of the wide toggles.
  const [sessionsOverlay, setSessionsOverlay] = useState(false);
  const [evidenceOverlay, setEvidenceOverlay] = useState(false);

  const seatUp = seat.state === "online";
  const backendUp = seatUp && seat.backendOk;

  const sessionsQuery = useQuery({
    queryKey: ["seat-sessions", profile],
    queryFn: ({ signal }) => listConversations(profile!, signal),
    enabled: seatUp && profile !== null,
    refetchInterval: 30_000,
  });
  const sessions: ConversationSummary[] = sessionsQuery.data?.conversations ?? [];

  const detailQuery = useQuery({
    queryKey: ["seat-conversation", activeId],
    queryFn: ({ signal }) => getConversation(activeId!, signal),
    enabled: seatUp && activeId !== null,
    staleTime: 5_000,
  });
  const detail = detailQuery.data;
  useEffect(() => {
    if (detail) adoptDetail(detail);
  }, [detail, adoptDetail]);

  // Opening the view with sessions but no selection lands on the newest one —
  // matching every session-bearing product; "new conversation" is one click.
  useEffect(() => {
    if (activeId === null && sessions.length > 0) {
      const first = sessions[0];
      if (first) setActive(first.conversation_id);
    }
    // Only on first arrival of the list, not on every refetch.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sessions.length === 0]);

  const invalidateSessions = () =>
    void queryClient.invalidateQueries({ queryKey: ["seat-sessions", profile] });

  const activeSummary = useMemo(
    () => sessions.find((session) => session.conversation_id === activeId) ?? null,
    [sessions, activeId],
  );
  const title =
    activeSummary?.title ?? detail?.title ?? (convo?.turns[0]?.userText.slice(0, 60) || "Conversation");

  // MUST stay above the `!seatUp` early return below. This is a hook, and it
  // was previously called after it — so the render where the seat came up went
  // from 66 hooks to more than 66 and React threw "Rendered more hooks than
  // during the previous render", blanking the whole app. It only reproduced on
  // the transition (seat down, then up), which is exactly the path a demo takes
  // and exactly the path a build and a typecheck do not.
  //
  // `useBillingLookup` already takes `seatUp` and returns a lookup that yields
  // null while the seat is down, so hoisting it costs nothing.
  const lookupModel = useBillingLookup(seatUp);

  if (!seatUp) {
    return (
      <EmptyState
        size="page"
        title="Seat not running"
        body="Conversations run through the seat harness, a local process. Start it (seat/README.md) and this screen picks up on its own — existing conversations resume where they stopped."
      />
    );
  }

  const openConversation = (id: string) => {
    setActive(id);
    setSessionsOverlay(false);
  };
  const startNew = () => {
    setActive(null);
    setSessionsOverlay(false);
  };

  const onCreated = (id: string) => {
    invalidateSessions();
    // Seed live state so the composer works before the first detail fetch.
    setActive(id);
  };

  const activeModelInfo = lookupModel(convo?.model ?? activeSummary?.model ?? null);

  const totals = convo?.totals ?? activeSummary?.usage ?? null;
  // Under a subscription, pi's cost numbers are list-price estimates of
  // flat-rate usage — a dollar figure would imply a bill that does not exist.
  // Tokens still show; only the currency is withheld.
  const totalCost =
    totals && costIsReal(activeModelInfo) ? formatCost(totals.cost_total) : null;

  const sessionsColumn = (
    <SessionList sessions={sessions} activeId={activeId} onOpen={openConversation} onNew={startNew} />
  );

  return (
    <div className="relative flex h-full min-h-0">
      {sessionsOpen ? (
        <div className="border-border hidden w-[240px] shrink-0 border-r xl:block">
          {sessionsColumn}
        </div>
      ) : null}

      <div className="flex min-h-0 min-w-0 flex-1 flex-col">
        <header className="border-border flex h-10 shrink-0 items-center gap-2 border-b pr-2 pl-2">
          <Button
            size="icon"
            variant="ghost"
            aria-label="Toggle session list"
            className="xl:hidden"
            onClick={() => setSessionsOverlay(true)}
          >
            <PanelLeft />
          </Button>
          <Button
            size="icon"
            variant="ghost"
            aria-label="Toggle session list"
            className="hidden xl:inline-flex"
            onClick={toggleSessions}
          >
            <PanelLeft />
          </Button>

          <h2 className="min-w-0 flex-1 truncate text-[12px] font-medium tracking-tight">
            {activeId ? title : "New conversation"}
          </h2>

          {totals && totals.total_tokens > 0 ? (
            <span
              className="text-muted-foreground mono hidden shrink-0 text-[10px] sm:inline"
              title={
                activeModelInfo?.billing === "subscription"
                  ? "Conversation totals. Usage counts against your plan; tokens are not billed individually."
                  : "Conversation totals, accumulated from per-message usage"
              }
            >
              {formatTokens(totals.total_tokens)} tok{totalCost ? ` · ${totalCost}` : ""}
            </span>
          ) : null}

          <EgressBadge info={activeModelInfo} servers={egressServers} />

          {activeId ? (
            <ModelPicker
              current={convo?.model ?? activeSummary?.model ?? null}
              disabled={convo?.streaming}
              swap
              onSelect={async (model) => {
                const applied = await changeModel(activeId, model.provider, model.id);
                setModel(activeId, applied);
                invalidateSessions();
              }}
            />
          ) : null}

          {activeId && convo && convo.turns.length > 0 ? (
            <Button
              size="icon"
              variant="ghost"
              aria-label="Export conversation as Markdown"
              onClick={() => exportMarkdown(title, convo.turns)}
            >
              <Download />
            </Button>
          ) : null}

          <Button
            size="icon"
            variant="ghost"
            aria-label="Toggle evidence panel"
            className="lg:hidden"
            onClick={() => setEvidenceOverlay(true)}
          >
            <PanelRight />
          </Button>
          <Button
            size="icon"
            variant="ghost"
            aria-label="Toggle evidence panel"
            aria-pressed={evidenceOpen}
            className={cn("hidden lg:inline-flex", evidenceOpen && "text-primary")}
            onClick={toggleEvidence}
          >
            <PanelRight />
          </Button>
        </header>

        {!backendUp ? (
          <p className="border-border text-warn shrink-0 border-b px-4 py-1.5 text-[11px]">
            Memory backend unreachable ({seat.state === "online" ? seat.backendDetail : ""}) — turns
            will run without recall until it is back.
          </p>
        ) : null}

        {activeId === null ? (
          <NewConversation
            profiles={reach.state === "online" ? reach.profiles : []}
            onCreated={onCreated}
          />
        ) : convo ? (
          <>
            <MessageList turns={convo.turns} conversationId={activeId} model={convo.model} />
            {convo.transportError ? (
              <p className="text-destructive shrink-0 px-4 pb-1 text-[11px]">
                Stream interrupted: {convo.transportError}
              </p>
            ) : null}
            <Composer
              disabled={convo.streaming}
              disabledReason="Waiting for the current turn to finish"
              onSend={(text) => void send(activeId, text, invalidateSessions)}
            />
          </>
        ) : detailQuery.isError ? (
          <EmptyState
            title="Could not load this conversation"
            body={
              detailQuery.error instanceof Error ? detailQuery.error.message : "Unknown error."
            }
          />
        ) : (
          <div className="flex-1" />
        )}
      </div>

      {evidenceOpen && activeId ? (
        <EvidencePanel
          conversationId={activeId}
          convo={convo}
          className="hidden w-[min(340px,26vw)] shrink-0 lg:flex"
        />
      ) : null}

      {/* Narrow-viewport overlays. Backdrop click closes; panels share the
          exact components the wide layout uses. */}
      {sessionsOverlay ? (
        <div className="absolute inset-0 z-30 lg:z-40">
          <button
            type="button"
            aria-label="Close session list"
            className="absolute inset-0 bg-black/50"
            onClick={() => setSessionsOverlay(false)}
          />
          <div className="bg-sidebar border-border absolute inset-y-0 left-0 w-[280px] max-w-[85vw] border-r shadow-2xl shadow-black/50">
            {sessionsColumn}
          </div>
        </div>
      ) : null}
      {evidenceOverlay && activeId ? (
        <div className="absolute inset-0 z-30 lg:hidden">
          <button
            type="button"
            aria-label="Close evidence panel"
            className="absolute inset-0 bg-black/50"
            onClick={() => setEvidenceOverlay(false)}
          />
          <EvidencePanel
            conversationId={activeId}
            convo={convo}
            onClose={() => setEvidenceOverlay(false)}
            className="absolute inset-y-0 right-0 w-[min(360px,92vw)] shadow-2xl shadow-black/50"
          />
        </div>
      ) : null}
    </div>
  );
}
