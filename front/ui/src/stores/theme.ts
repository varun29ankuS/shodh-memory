import { create } from "zustand";

/**
 * Ground selection.
 *
 * Three states, not two. "system" is the one most theme switches forget they
 * have, and dropping it means a machine set to light at 6am and dark at 6pm
 * gets whatever the analyst last clicked instead. Clicking the ground you are
 * already on returns to system — the mockup's behaviour, kept because it is the
 * only affordance that reaches the third state without a third button.
 */
export type Ground = "paper" | "night" | "system";

const KEY = "shodh.ground";

/**
 * `<html data-theme>` carries the choice:
 *   data-theme="paper"  → paper tokens
 *   absent              → the dark values in :root
 *
 * "system" resolves at apply time rather than being written through, so a
 * machine that flips at dusk flips with it. `prefers-color-scheme: dark` means
 * night, which means *no* attribute — the default ground is already dark.
 */
function apply(ground: Ground) {
  const root = document.documentElement;
  const wantsPaper =
    ground === "paper" ||
    (ground === "system" && !window.matchMedia("(prefers-color-scheme: dark)").matches);

  if (wantsPaper) root.setAttribute("data-theme", "paper");
  else root.removeAttribute("data-theme");
}

function load(): Ground {
  try {
    const saved = localStorage.getItem(KEY);
    if (saved === "paper" || saved === "night" || saved === "system") return saved;
  } catch {
    // Private windows and locked-down browsers throw on access rather than
    // returning null. A theme is not worth a blank screen.
  }
  return "system";
}

type ThemeStore = {
  ground: Ground;
  /** Selecting the ground already in effect returns to following the system. */
  select: (next: Exclude<Ground, "system">) => void;
};

export const useTheme = create<ThemeStore>((set, get) => ({
  ground: load(),
  select: (next) => {
    const ground: Ground = get().ground === next ? "system" : next;
    set({ ground });
    apply(ground);
    try {
      localStorage.setItem(KEY, ground);
    } catch {
      // Same as above: the choice just does not survive the session.
    }
  },
}));

/**
 * Applies the stored ground before first paint and keeps "system" live.
 * Called once from the app shell; safe to call again.
 */
export function initTheme(): () => void {
  apply(useTheme.getState().ground);

  const scheme = window.matchMedia("(prefers-color-scheme: dark)");
  const onScheme = () => {
    // Only "system" tracks the OS. An explicit choice stays put.
    if (useTheme.getState().ground === "system") apply("system");
  };
  scheme.addEventListener("change", onScheme);
  return () => scheme.removeEventListener("change", onScheme);
}
