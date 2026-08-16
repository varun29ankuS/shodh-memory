import { create } from "zustand";

/**
 * The two pieces of state that are genuinely global.
 *
 * WORKFLOWS.md: "Selection is global state, not per-panel. One selected object
 * at a time." Every list in the product feeds one Inspector, so the selected id
 * cannot live inside whichever list produced it — the graph, the result list
 * and the lineage edges all set and read the same value.
 *
 * The active profile is global for a harder reason. Every recall carries a
 * `user_id`, and `MultiUserMemoryManager::get_user_memory` (src/handlers/state.rs)
 * *creates* a store on demand for an id it has not seen — it does not 404. So a
 * wrong or invented profile does not fail loudly; it silently provisions an
 * empty RocksDB directory and returns no results. The active profile is
 * therefore only ever set from the list the server itself returned, and is
 * `null` until one arrives.
 */
interface SessionState {
  /** A value the server listed in `GET /api/users` — or, exactly once removed
   *  from that guarantee, a profile someone deliberately named on the seat's
   *  new-conversation screen (which the backend provisions on first write). */
  profile: string | null;
  /** True when `profile` was set by a deliberate act (switcher, seat creation)
   *  rather than adopted from the server list. A pinned profile survives
   *  reconciliation while the server has not yet listed it — the seat creates
   *  the store on the first turn, and wiping the selection in that window
   *  would tear down the conversation that is about to populate it. */
  profilePinned: boolean;
  /** Id of the memory currently open in the Inspector. */
  selectedMemoryId: string | null;
  /**
   * Id of the ENTITY currently open in the Inspector — a `UniverseStar.id` from
   * the knowledge graph, which is a different kind of object from a memory and
   * cannot share the field.
   *
   * Two ids rather than one tagged union because the two are genuinely
   * different objects reached from different destinations, and every existing
   * caller of `select`/`selectedMemoryId` would otherwise have to learn about a
   * kind it never encounters. They are mutually exclusive: selecting either
   * clears the other, so the Inspector still shows exactly one thing and the
   * "one selected object at a time" rule in WORKFLOWS.md holds.
   */
  selectedEntityId: string | null;
  /** The query the results on screen came from, for the Inspector's header. */
  activeQuery: string;
  /**
   * What is being typed RIGHT NOW, before it is committed.
   *
   * Deliberately separate from `activeQuery`. Recall is a multi-leg retrieval
   * -- vector, BM25 and graph, then fusion and re-rank -- so it stays
   * submit-driven; firing it per keystroke would spend all of that answering
   * prefixes nobody asked about. But highlighting the graph is a local string
   * match over names already in memory, and it costs nothing, so it should
   * happen as you type. One field, two costs, two cadences.
   */
  cueDraft: string;
  /**
   * Extra terms the cue matches on, beyond `cueDraft` itself.
   *
   * The second producer of a cue is the conversation: a `memory_recall` carries
   * the query the model asked AND the keyword lists of the facts it got back
   * (`RecallFact.related_entities`), and those terms name entities the query
   * string does not contain. One field could not carry both — a cue is one
   * string typed by a person, and joining a dozen keywords into it would make
   * the substring match test a sentence that appears nowhere.
   *
   * Same channel, not a second visual language: the canvas unions these into
   * the set it already rings in the accent, so an agent-driven narrowing and a
   * typed one look identical, which is the point. Cleared by any typed cue —
   * the person's own cue supersedes the model's rather than compounding with it.
   */
  cueEntities: string[];

  setProfile: (profile: string | null) => void;
  select: (memoryId: string | null) => void;
  selectEntity: (entityId: string | null) => void;
  setActiveQuery: (query: string) => void;
  setCueDraft: (cue: string) => void;
  /** Set both halves of the cue at once — the only way `cueEntities` becomes
   *  non-empty. Called by the view bus when an agent cue command applies; never
   *  from a keystroke, which has no entity list. */
  setCue: (cue: string, entities: string[]) => void;
  /** Adopt the server's list: keep the current profile if it is still offered
   *  (or pinned, see above), otherwise fall back to the first. Prevents a
   *  stale auto-adopted selection surviving a backend restart that dropped it. */
  reconcileProfiles: (profiles: string[]) => void;
}

/**
 * The active profile, remembered across reloads.
 *
 * A refresh used to drop you back onto whichever profile the server listed
 * first — usually an empty one — so a reload mid-demo silently swapped the
 * corpus under you and every screen went blank for no stated reason.
 *
 * SAFE BECAUSE RECONCILIATION STILL DECIDES. A restored name is a suggestion,
 * not an authority: it is written back unpinned, so `reconcileProfiles` keeps
 * it only while the server still lists it and otherwise falls back. That
 * preserves the rule this store exists to enforce — a profile is only ever one
 * the server named, because an invented one silently provisions an empty store
 * instead of failing.
 *
 * Only the profile persists. The selections do not: they name objects inside a
 * corpus, and restoring an id whose memory may have been rewritten since would
 * open the Inspector on something that no longer means what it did.
 */
const PROFILE_KEY = "shodh.profile";

function readStoredProfile(): string | null {
  try {
    return window.localStorage.getItem(PROFILE_KEY);
  } catch {
    // Storage can throw outright in private modes and hardened embeddings.
    // Forgetting the profile is a far smaller failure than not loading.
    return null;
  }
}

function storeProfile(profile: string | null): void {
  try {
    if (profile === null) window.localStorage.removeItem(PROFILE_KEY);
    else window.localStorage.setItem(PROFILE_KEY, profile);
  } catch {
    /* see readStoredProfile */
  }
}

export const useSession = create<SessionState>((set) => ({
  profile: readStoredProfile(),
  profilePinned: false,
  selectedMemoryId: null,
  selectedEntityId: null,
  activeQuery: "",
  cueDraft: "",
  cueEntities: [],

  setProfile: (profile) => {
    storeProfile(profile);
    return set({
      profile,
      profilePinned: profile !== null,
      selectedMemoryId: null,
      selectedEntityId: null,
    });
  },
  setCueDraft: (cueDraft) => set({ cueDraft, cueEntities: [] }),
  setCue: (cueDraft, cueEntities) => set({ cueDraft, cueEntities }),

  select: (selectedMemoryId) => set({ selectedMemoryId, selectedEntityId: null }),
  selectEntity: (selectedEntityId) => set({ selectedEntityId, selectedMemoryId: null }),
  // Committing a query starts a new answer, and the selected object belonged to
  // the previous one. Left standing, it stays open in the Inspector beside a
  // result set it is not part of — verified in the browser: search from the
  // pre-query corpus listing with a row selected and the detail pane holds a
  // memory that appears nowhere in the results, which reads as a stale panel
  // rather than as a deliberate carry-over. Clearing both is the same rule
  // `setProfile` and `reconcileProfiles` already follow for the same reason.
  setActiveQuery: (activeQuery) =>
    set({
      activeQuery,
      cueDraft: activeQuery,
      // A committed search is the person taking the cue back; the model's
      // entity terms would otherwise keep lighting nodes the new query never
      // mentioned, under a heading that names the new query.
      cueEntities: [],
      selectedMemoryId: null,
      selectedEntityId: null,
    }),

  reconcileProfiles: (profiles) =>
    set((s) => {
      if (s.profile && (profiles.includes(s.profile) || s.profilePinned)) return s;
      // Both selections clear on a profile change: they name objects in the
      // previous profile's corpus, and an id from one store means nothing in
      // another.
      if (profiles.length === 0) {
        // AN EMPTY LIST IS NOT A DECISION. It is what arrives before the
        // profile query resolves, and what arrives when the read fails. The
        // in-memory profile still clears -- there is nothing to search against
        // -- but the REMEMBERED one must survive, or the first render forgets
        // the choice before the server has had a chance to confirm it. That
        // was the bug: persistence was added, and then erased by the load it
        // was meant to survive.
        if (s.profile === null) return s;
        return { profile: null, selectedMemoryId: null, selectedEntityId: null };
      }
      storeProfile(profiles[0]);
      return {
        profile: profiles[0],
        profilePinned: false,
        selectedMemoryId: null,
        selectedEntityId: null,
      };
    }),
}));
