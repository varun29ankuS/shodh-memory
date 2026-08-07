import { useEffect } from "react";
import { cn } from "@/lib/utils";
import { NOTICE_MS, useActivity } from "@/stores/activity";

/**
 * What the conversation just did to the corpus, said where the corpus is.
 *
 * A write or a reinforcement changes what every memory surface draws, and the
 * conversation may well be minimised to a 40px bar when it happens. Without
 * this, the graph, the map and the result list silently describe a corpus that
 * moved under them.
 *
 * DELIBERATELY NOT THE RECORD. Each line is transient and carries no control:
 * the durable entry — with the ledger id and its revert — stays in the
 * conversation block, because an action a person may need to undo must not
 * expire after six seconds. This says only "that happened", from the fields the
 * event carried (stores/activity.ts builds the text), and then gets out of the
 * way.
 *
 * `aria-live="polite"`: a screen reader gets told, at the next pause, and is
 * never interrupted mid-sentence for something with no action attached. The
 * region itself is ALWAYS mounted, empty or not — a live region that appears in
 * the same commit as its first child is a new node rather than a mutation, and
 * assistive technology generally will not announce it.
 */
export function ActivityNotices({ className }: { className?: string }) {
  const notices = useActivity((s) => s.notices);
  const expireNotice = useActivity((s) => s.expireNotice);

  useEffect(() => {
    if (notices.length === 0) return;
    // Absolute deadlines, not fresh countdowns: this effect re-runs whenever
    // the list changes, and restarting each timer would keep an old notice
    // alive for as long as new ones kept arriving.
    const timers = notices.map((notice) =>
      window.setTimeout(
        () => expireNotice(notice.id),
        Math.max(0, notice.at + NOTICE_MS - Date.now()),
      ),
    );
    return () => timers.forEach((t) => clearTimeout(t));
  }, [notices, expireNotice]);

  return (
    <div
      aria-live="polite"
      className={cn(
        "pointer-events-none absolute top-3 right-4 z-10 flex flex-col items-end gap-1.5",
        className,
      )}
    >
      {notices.map((notice) => (
        <p
          key={notice.id}
          // Muted on the card surface, no hue. These are facts about the
          // corpus, not alarms — the destructive and warn tokens have to keep
          // meaning "something is wrong" or they stop meaning anything.
          className="border-border bg-card/90 text-muted-foreground mono rounded border px-2 py-1 text-[10px] shadow-lg backdrop-blur-sm"
        >
          {notice.text}
        </p>
      ))}
    </div>
  );
}
