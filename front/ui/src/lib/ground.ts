import { useCallback, useEffect, useSyncExternalStore } from "react";

/**
 * Which ground the product is read on.
 *
 * PAPER IS THE DEFAULT, AND IT IS NOT THE SYSTEM'S DECISION. index.css states
 * the reason at length: this product is read, not monitored, and paper is what
 * reading has always been. A ground chosen by `prefers-color-scheme` would
 * make the product's default appearance a property of the reader's operating
 * system, which is precisely the decision the design already took — and it
 * would flip every destination to the night ground for anyone whose machine
 * happens to be set that way, without them asking for it here.
 *
 * So this is a two-state control with a stated default, not a three-state one
 * that defers. The night ground is reached by asking for it, and the asking is
 * remembered: a ground that resets on reload is a bug.
 *
 * Three things are stamped together and must never be allowed to disagree:
 *
 *   - `data-theme`, which selects the token set. index.css declares the paper
 *     values on a bare `:root` and the night values behind
 *     `[data-theme="dark"]`, so an unstamped document is paper.
 *   - the `dark` class, because index.css declares
 *     `@custom-variant dark (&:is(.dark *))` — any component pulled from the
 *     registry with a `dark:` utility reads the CLASS, not the attribute, and
 *     a class left permanently on (as index.html shipped it) would style those
 *     components for a ground the tokens are not on.
 *   - `color-scheme`, which is what the browser paints form controls, the
 *     caret and default scrollbars from. Paper tokens under a dark
 *     `color-scheme` gives a light page with a dark text field, which reads as
 *     a rendering fault rather than as a theme.
 *
 * It is applied at boot from `main.tsx` rather than from the screen that owns
 * the control: deep-linking to any other destination must not land on a ground
 * the person did not pick, and a screen that corrects the ground one frame
 * after it mounts is a flash nobody asked for.
 */

export type Ground = "light" | "dark";

const STORAGE_KEY = "shodh.ground";
const DEFAULT: Ground = "light";

/** Storage is not guaranteed: private modes and embedded webviews throw on
 *  access rather than returning null. A ground preference is not worth taking
 *  the app down for, so every access is guarded and failure degrades to the
 *  default ground. */
function readStored(): Ground {
  try {
    const v = localStorage.getItem(STORAGE_KEY);
    return v === "light" || v === "dark" ? v : DEFAULT;
  } catch {
    return DEFAULT;
  }
}

function writeStored(ground: Ground) {
  try {
    localStorage.setItem(STORAGE_KEY, ground);
  } catch {
    /* A preference that cannot be saved is still worth honouring this session. */
  }
}

/** Stamp the document. Idempotent, so it is safe to call on every render pass. */
export function applyGround(ground: Ground) {
  const root = document.documentElement;
  if (ground === "dark") {
    root.setAttribute("data-theme", "dark");
    root.classList.add("dark");
  } else {
    root.setAttribute("data-theme", "light");
    root.classList.remove("dark");
  }
  root.style.colorScheme = ground;
}

// =============================================================================
// THE STORE
// =============================================================================

/**
 * A short external store rather than a zustand slice.
 *
 * The choice is already persisted outside React and already has to be applied
 * before React mounts, so the durable copy is `localStorage` and the module
 * holds the live one. `useSyncExternalStore` reads that without the value
 * existing in two places that can disagree.
 */
let current: Ground = readStored();
const listeners = new Set<() => void>();

function subscribe(listener: () => void): () => void {
  listeners.add(listener);
  return () => {
    listeners.delete(listener);
  };
}

export function setGround(ground: Ground) {
  if (ground === current) return;
  current = ground;
  writeStored(ground);
  applyGround(ground);
  for (const l of listeners) l();
}

/** Called once from `main.tsx`, before the first paint. */
export function applyStoredGround() {
  applyGround(current);
}

/** The ground on screen, and the way to change it. Consumers that paint —
 *  a canvas reading tokens through `getComputedStyle` — depend on the returned
 *  value so a ground change re-runs them; CSS reaches everything else. */
export function useGround(): { ground: Ground; setGround: (g: Ground) => void } {
  const ground = useSyncExternalStore(
    subscribe,
    () => current,
    () => DEFAULT,
  );

  useEffect(() => {
    applyGround(ground);
  }, [ground]);

  return { ground, setGround: useCallback((g: Ground) => setGround(g), []) };
}
