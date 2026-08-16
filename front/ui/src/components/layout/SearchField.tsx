import { useEffect, useRef, useState } from "react";
import { Search } from "lucide-react";
import { cn } from "@/lib/utils";
import type { Reachability } from "@/lib/api";
import { useSession } from "@/stores/session";

/**
 * The search control.
 *
 * Submit-driven. Recall runs vector, keyword and graph retrieval and then fuses
 * and re-ranks them; firing that on every keystroke would spend the work
 * repeatedly to answer prefixes nobody asked about.
 *
 * The draft is local and the committed query is global, because they are
 * genuinely different things: the draft is what someone is typing, the
 * committed query is what the results on screen are an answer to. Keeping them
 * as one value makes the result list contradict its own heading mid-word.
 *
 * WHY IT IS THIS BIG. Searching is the primary verb of this product and this is
 * the only control that performs it, and as a bare 280px input at the right
 * edge of a 48px bar it was missed outright — by the person who built it, on a
 * screen whose whole left column was captioned "search above". Size, a leading
 * glyph and a stated shortcut are what make a control read as the thing to use
 * rather than as chrome. It is still one input in the header: this is a
 * workbench, and a hero search box would claim the screen the results need.
 *
 * The keyboard hint is not decoration — it is the shortcut's only discovery
 * surface, and it doubles as the strongest "this is a search field" signal at a
 * glance. It hides while the field has focus, where it would be advice on how
 * to reach somewhere you already are.
 */

/** Focus keys. `/` is the convention every reading surface uses; ⌘/Ctrl-K is the
 *  convention every command surface uses. Supporting both costs one branch. */
function isFocusKey(e: KeyboardEvent): boolean {
  if (e.key === "k" && (e.metaKey || e.ctrlKey)) return true;
  return e.key === "/" && !e.metaKey && !e.ctrlKey && !e.altKey;
}

/** Typing must never be stolen. The conversation composer is mounted on every
 *  destination (ConversationOverlay), so "/" inside it has to stay a slash. */
function isTyping(target: EventTarget | null): boolean {
  const el = target as HTMLElement | null;
  if (!el || !el.tagName) return false;
  const tag = el.tagName.toLowerCase();
  return tag === "input" || tag === "textarea" || tag === "select" || el.isContentEditable;
}

export function SearchField({ reach }: { reach: Reachability }) {
  const profile = useSession((s) => s.profile);
  const activeQuery = useSession((s) => s.activeQuery);
  const setActiveQuery = useSession((s) => s.setActiveQuery);
  const setCueDraft = useSession((s) => s.setCueDraft);
  const [draft, setDraft] = useState(activeQuery);
  const [focused, setFocused] = useState(false);
  const input = useRef<HTMLInputElement>(null);

  // The committed query can now be set from outside this component — a cue chip
  // on the empty Recall state commits one directly. Without this the field would
  // sit empty while the list below it showed that query's results, which is the
  // same "heading contradicts its own list" failure the draft/committed split
  // exists to prevent, only inverted.
  //
  // This replaces an unconditional clear on profile change. That clear was aimed
  // at the right problem and solved the wrong half of it: `setProfile` does not
  // touch `activeQuery` (stores/session.ts:65-71), so the committed query
  // survives the switch and re-runs against the new corpus — emptying the field
  // hid a search that was still on screen. Re-seeding from the committed value
  // states what the results actually are, on both paths.
  useEffect(() => setDraft(activeQuery), [activeQuery, profile]);

  const usable = reach.state === "online" && profile !== null;

  useEffect(() => {
    if (!usable) return;
    const onKey = (e: KeyboardEvent) => {
      if (!isFocusKey(e) || isTyping(e.target)) return;
      e.preventDefault();
      input.current?.focus();
      input.current?.select();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [usable]);

  return (
    <form
      role="search"
      onSubmit={(e) => {
        e.preventDefault();
        setActiveQuery(draft.trim());
        // Committing is the end of the typing task. Blurring hands the keyboard
        // back to the page so the results can be reached without a Tab out of
        // a field that has already done its job.
        input.current?.blur();
      }}
      className="flex min-w-0 flex-1 justify-end"
    >
      {/* The wrapper, not the input, carries the frame: the icon and the hint
          sit inside the field's box, so the box has to be the thing that draws
          it. `focus-within` puts the ring on the same box, which is what makes
          the whole control read as focused rather than just its text area. */}
      <div
        className={cn(
          "border-input bg-input/30 flex h-8 w-full max-w-[420px] min-w-0 items-center gap-2",
          "rounded-md border pr-2 pl-2.5 transition-colors duration-100",
          focused ? "border-ring ring-ring/50 ring-[3px]" : "hover:border-ring/40",
          !usable && "opacity-50",
        )}
      >
        <Search
          aria-hidden="true"
          className="text-muted-foreground size-[15px] shrink-0"
          strokeWidth={1.8}
        />
        <input
          ref={input}
          type="search"
          name="q"
          value={draft}
          onChange={(e) => {
            setDraft(e.target.value);
            // Published live: the graph rings matches as you type. The
            // committed query, and the retrieval it triggers, still waits
            // for Enter.
            setCueDraft(e.target.value);
          }}
          onFocus={() => setFocused(true)}
          onBlur={() => setFocused(false)}
          onKeyDown={(e) => {
            // Escape leaves the field without committing. A search box that
            // traps the keyboard is the reason people reach for the mouse.
            if (e.key === "Escape") input.current?.blur();
          }}
          disabled={!usable}
          placeholder={usable ? "Search this memory…" : "Search unavailable"}
          aria-label="Search memory"
          className={cn(
            "placeholder:text-muted-foreground min-w-0 flex-1 bg-transparent text-[13px]",
            "outline-none disabled:cursor-not-allowed",
            // Chrome draws its own clear affordance on type=search, in its own
            // colours, on a surface it knows nothing about.
            "[&::-webkit-search-cancel-button]:appearance-none",
          )}
        />
        {usable && !focused ? (
          <kbd
            aria-hidden="true"
            className="border-border text-muted-foreground/70 mono shrink-0 rounded border px-1 text-[10px] leading-[16px]"
          >
            /
          </kbd>
        ) : null}
      </div>
    </form>
  );
}
