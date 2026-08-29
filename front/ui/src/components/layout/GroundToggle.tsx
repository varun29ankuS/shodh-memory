import { cn } from "@/lib/utils";
import { useTheme } from "@/stores/theme";

/**
 * PAPER / NIGHT.
 *
 * Two buttons for three states, because the third — following the system — is
 * reached by pressing the ground you are already on. That is less discoverable
 * than a third button and much less clutter, and the pressed state makes the
 * current ground legible either way: when neither reads as pressed, the system
 * is deciding.
 *
 * Named for the grounds rather than "light/dark" because Paper is a specific
 * ground with a specific palette, not a generic light mode.
 */
export function GroundToggle() {
  const ground = useTheme((s) => s.ground);
  const select = useTheme((s) => s.select);

  return (
    <div className="flex items-center font-mono text-[11px] tracking-wider" role="group" aria-label="Ground">
      {(["paper", "night"] as const).map((g) => (
        <button
          key={g}
          type="button"
          onClick={() => select(g)}
          aria-pressed={ground === g}
          title={
            ground === g
              ? "Press again to follow the system"
              : `Switch to ${g === "paper" ? "the paper ground" : "the night ground"}`
          }
          className={cn(
            "border-border border px-2 py-0.5 uppercase transition-colors",
            "first:rounded-l-sm first:border-r-0 last:rounded-r-sm",
            ground === g
              ? "bg-foreground text-background"
              : "text-muted-foreground hover:text-foreground",
          )}
        >
          {g}
        </button>
      ))}
    </div>
  );
}
