import { create } from "zustand";

/**
 * Whether the conversation's account of a move is currently on screen.
 *
 * ONE FIELD, AND IT IS A TRACE'S IDENTITY RATHER THAN A BOOLEAN. The block is
 * shown while the key it was opened for is still the key the view bus produces,
 * so every way a move can stop being true closes it without anything having to
 * notice: an offer accepted, an offer refused, a turn boundary expiring one, a
 * hand on an axis killing the notice that covered it. A boolean would need each
 * of those sites to remember to clear it, and the one that forgot would leave a
 * block on screen describing a view that had moved on. See `traceKey` in
 * `lib/view/presence.ts`.
 *
 * WHY IT IS NOT IN `stores/view.ts`. That store is the bus: what the view IS and
 * who is allowed to move it. This is what a person has looked at, which is not
 * view state at all — it is read by two components that must agree on it (the
 * block, and the header line that yields its place while the block is up) and by
 * nothing else. Folding it in would put a presentation timer inside the module
 * whose whole discipline is that nothing in it renders or navigates.
 */
interface TraceState {
  /** The trace key on screen, or null when nothing is. */
  shown: string | null;
  open: (key: string) => void;
  close: () => void;
}

export const useTrace = create<TraceState>((set) => ({
  shown: null,
  open: (key) => set({ shown: key }),
  close: () => set({ shown: null }),
}));
