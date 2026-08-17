import { useEffect, useMemo, useState , useRef} from "react";
import { createPortal } from "react-dom";
import { useLocation, useNavigate } from "react-router-dom";
import { useQueryClient } from "@tanstack/react-query";
import { Maximize2, MessageSquare, Minus, X } from "lucide-react";
import { cn } from "@/lib/utils";
import { changeModel } from "@/lib/seat/client";
import type { SeatReachability } from "@/lib/seat/types";
import { formatCost, formatTokens } from "@/lib/format";
import { useChat } from "@/stores/chat";
import { Button } from "@/components/ui/button";
import { ProviderLogo } from "@/components/ui/provider-logo";
import { Composer } from "./Composer";
import { EgressBadge } from "./EgressBadge";
import { costIsReal, useBillingLookup } from "./useBilling";
import { MessageList } from "./MessageList";
import { ModelPicker } from "./ModelPicker";

/**
 * The conversation, available from every surface.
 *
 * "The conversation is the instrument": an analyst reading the graph or the map
 * should be able to ask about what they are looking at without navigating away
 * from it. A route-only chat forces a context switch precisely when context is
 * the thing being examined.
 *
 * NOT A FORK. This is a second MOUNT of the same conversation state — the same
 * `useChat` store, the same Composer, ModelPicker, EgressBadge and MessageList
 * the /chat route uses. Opening the overlay mid-stream on another route shows
 * that stream continuing, because there is only ever one conversation object
 * and both surfaces read it. The /chat route stays for deep links, session
 * management and full-screen work.
 *
 * COLLISION LAYOUT, decided rather than discovered: the overlay docks
 * bottom-RIGHT but sits ABOVE the Inspector in z-order and offsets itself past
 * the Inspector's reserved width on the routes that have one. Three options
 * were on the table — cover the Inspector, push the content, or dodge — and
 * dodging is the only one that keeps both readable: covering it hides the
 * detail pane exactly when a conversation is most likely to be about the
 * selected object, and pushing violates the same overlay rule the rail follows
 * (never reflow a force layout or a map because a panel opened). Below the
 * Inspector's breakpoint there is no Inspector to dodge, so it docks flush.
 *
 * Portal-rendered into `document.body` so no route's `overflow-hidden` or
 * stacking context can clip it — `main` clips its canvases deliberately, and a
 * panel that lives inside that box would be cut off at the stage edge.
 */

type Mode = "minimized" | "expanded";

/** Routes that render the Inspector, which the overlay must not sit on top of.
 *  Kept in step with App.tsx's ROUTES_WITH_INSPECTOR. */
const INSPECTOR_ROUTES = ["/recall", "/geo", "/graph"];

export function ConversationOverlay({ seat }: { seat: SeatReachability }) {
  const { pathname } = useLocation();
  const navigate = useNavigate();
  const queryClient = useQueryClient();

  const activeId = useChat((s) => s.activeId);
  const convo = useChat((s) => (activeId ? (s.conversations[activeId] ?? null) : null));
  const send = useChat((s) => s.send);
  const setModel = useChat((s) => s.setModel);

  const [mode, setMode] = useState<Mode>("minimized");

  const dockRef = useRef<HTMLElement | null>(null);


  // The stages' bottom hint strips live at the same edge this panel docks

  // to, and were being clipped underneath it (found during reactive-surface

  // verification: "13 l…"). Rather than teach three views the overlay's

  // geometry, the overlay publishes its measured footprint as a CSS custom

  // property and the strips pad by it — one source of truth, measured from

  // the DOM rect so every width variant (expanded/minimized/narrow

  // viewport) is exact. Cleared on unmount so /chat and dismissed states

  // reclaim the space.

  useEffect(() => {

    const el = dockRef.current;

    if (!el) return;

    const apply = () => {

      const inset = Math.max(0, window.innerWidth - el.getBoundingClientRect().left) + 12;

      document.documentElement.style.setProperty("--overlay-dock-inset", `${inset}px`);

    };

    apply();

    const ro = new ResizeObserver(apply);

    ro.observe(el);

    window.addEventListener("resize", apply);

    return () => {

      ro.disconnect();

      window.removeEventListener("resize", apply);

      document.documentElement.style.removeProperty("--overlay-dock-inset");

    };

  });
  const [dismissed, setDismissed] = useState(false);

  const lookupModel = useBillingLookup(seat.state === "online");
  const model = convo?.model ?? null;
  const modelInfo = lookupModel(model);

  const totals = convo?.totals ?? null;
  const totalCost = totals && costIsReal(modelInfo) ? formatCost(totals.cost_total) : null;

  const streaming = convo?.streaming ?? false;

  // A stream starting is the one event that should raise the panel on its own:
  // the whole point is that the answer is visible from wherever you are.
  useEffect(() => {
    if (streaming) setDismissed(false);
  }, [streaming]);

  // Escape minimizes rather than closes. Closing would look like it ended the
  // conversation, which it does not — the conversation keeps streaming.
  useEffect(() => {
    if (mode !== "expanded") return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setMode("minimized");
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [mode]);

  const rightOffset = useMemo(
    // Dodge the Inspector where one exists. Its width is `min(280px,36vw)`
    // (Inspector.tsx), so the same expression keeps the two in lockstep.
    () => (INSPECTOR_ROUTES.includes(pathname) ? "calc(min(280px,36vw) + 0.75rem)" : "0.75rem"),
    [pathname],
  );

  // The /chat route IS the conversation, full width. A floating copy of it on
  // top of itself is noise.
  if (pathname === "/chat") return null;
  if (seat.state !== "online") return null;
  /**
   * NO CONVERSATION, NO DOCK.
   *
   * This used to render whatever the state, so every screen in the product
   * carried a permanent bar in its bottom-right corner reading "No model" with
   * a dismiss X beside it — overlapping the content on each one. It was neither
   * of the two things a corner bar is allowed to be. Not a state indicator: the
   * one place this product states its connections is the status strip, said
   * once for the whole product, and a second permanent statement in the
   * opposite corner is the three-places-one-fact problem that strip exists to
   * have solved. Not a transient notice either, since it never went away and
   * offered a dismissal for a state that was not going to change.
   *
   * And it could not act. With no active conversation the expanded panel had no
   * model picker, no transcript and no composer — only a line telling the
   * reader to go to Conversations. A bar whose whole content is directions to
   * another screen is what the nav entry already is.
   *
   * So the dock exists exactly when there is a conversation for it to be about,
   * which is also when its header answers the two questions it is for — which
   * model, how many tokens. Nothing is orphaned: Conversations is still in the
   * rail, and a stream starting un-dismisses this from anywhere.
   */
  if (!activeId || !convo) return null;
  if (dismissed) return null;

  // Null for a real conversation whose model the seat has not reported yet —
  // said plainly rather than guessed at, and now a fact about something that
  // exists rather than a label on an empty shell.
  const label = model ? model.name || model.id : "No model";

  const body = (
    <section
      ref={dockRef}
      aria-label="Conversation"
      className={cn(
        "border-border bg-card fixed bottom-3 z-40 flex flex-col rounded-lg border shadow-2xl shadow-black/50",
        // `transition-[height,width]` only: transitioning `all` would animate
        // the offset change on every route switch, which reads as the panel
        // sliding around for no reason. Reduced motion is handled globally in
        // index.css, which collapses this to an instant state change.
        "transition-[height,width] duration-200 ease-out",
        mode === "expanded" ? "h-[min(560px,70vh)] w-[min(420px,calc(100vw-1.5rem))]" : "h-10 w-[min(340px,calc(100vw-1.5rem))]",
      )}
      style={{ right: rightOffset }}
    >
      {/* The always-visible answer to "which model, how many tokens" from any
          surface. It is the whole reason the minimized state is a bar rather
          than a bubble: a floating icon answers neither question. */}
      <header className="flex h-10 shrink-0 items-center gap-1.5 px-2">
        <button
          type="button"
          onClick={() => setMode(mode === "expanded" ? "minimized" : "expanded")}
          aria-expanded={mode === "expanded"}
          aria-label={mode === "expanded" ? "Minimize conversation" : "Expand conversation"}
          className="hover:bg-accent/60 focus-visible:ring-ring flex min-w-0 flex-1 items-center gap-1.5 rounded px-1 py-1 text-left transition-colors focus-visible:ring-2 focus-visible:outline-none"
        >
          {/* Logo first: it identifies the provider faster than the model id
              can be read, and it is what makes this bar scannable from across
              a room during a demo. Falls back to the message glyph only when
              there is no model to identify. */}
          {model ? (
            <ProviderLogo provider={model.provider} className="size-3.5" />
          ) : (
            <MessageSquare aria-hidden="true" className="text-muted-foreground size-3.5 shrink-0" />
          )}
          <span className="mono min-w-0 flex-1 truncate text-[11px]">{label}</span>
          {streaming ? (
            // Streaming is stated, not animated into a spinner: a spinner says
            // "busy", this says which conversation is producing tokens.
            <span className="text-primary shrink-0 text-[10px]">streaming</span>
          ) : null}
        </button>

        {totals && totals.total_tokens > 0 ? (
          <span
            className="text-muted-foreground mono shrink-0 text-[10px]"
            title="Conversation totals, accumulated from per-message usage"
          >
            {formatTokens(totals.total_tokens)} tok{totalCost ? ` · ${totalCost}` : ""}
          </span>
        ) : null}

        <EgressBadge info={modelInfo} />

        {mode === "expanded" ? (
          <>
            <Button
              size="icon"
              variant="ghost"
              aria-label="Open conversation full screen"
              onClick={() => navigate("/chat")}
            >
              <Maximize2 />
            </Button>
            <Button
              size="icon"
              variant="ghost"
              aria-label="Minimize conversation"
              onClick={() => setMode("minimized")}
            >
              <Minus />
            </Button>
          </>
        ) : (
          <Button
            size="icon"
            variant="ghost"
            aria-label="Hide conversation"
            onClick={() => setDismissed(true)}
          >
            <X />
          </Button>
        )}
      </header>

      {mode === "expanded" ? (
        <>
          <div className="border-border flex shrink-0 items-center gap-1.5 border-y px-2 py-1.5">
            <ModelPicker
              current={model}
              disabled={streaming}
              swap
              onSelect={async (m) => {
                const applied = await changeModel(activeId, m.provider, m.id);
                setModel(activeId, applied);
                void queryClient.invalidateQueries({ queryKey: ["seat-sessions"] });
              }}
            />
          </div>

          <div className="min-h-0 flex-1 overflow-y-auto">
            <MessageList turns={convo.turns} conversationId={activeId} model={model} />
          </div>

          <div className="border-border shrink-0 border-t p-2">
            <Composer
              disabled={streaming}
              disabledReason={streaming ? "Waiting for the current turn" : undefined}
              onSend={(text) => void send(activeId, text)}
            />
          </div>
        </>
      ) : null}
    </section>
  );

  return createPortal(body, document.body);
}
