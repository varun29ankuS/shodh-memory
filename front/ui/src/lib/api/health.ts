import { api, ApiError, NetworkError } from "./client";

/**
 * Reachability, authorization, and the profile list — from one request.
 *
 * `GET /api/users` is verified in src/handlers/router.rs and sits behind auth
 * (src/auth.rs::auth_middleware exempts only /health and /webhook/*), so a 401
 * here proves the key is wrong. /health would prove nothing about the key,
 * which is the failure this has to distinguish.
 *
 * The handler is `Json<Vec<String>>` — src/handlers/users.rs, `list_users`
 * returns `state.list_users()` — so the body is a plain array of profile ids.
 * The previous version of this probe fetched exactly this and threw the body
 * away, then the UI hardcoded an identity string next to it.
 */
/**
 * Backend profiles that belong to a person, as opposed to machinery.
 *
 * The seat stores its lessons under `<user>.seat-harness` — a real per-user
 * store on the backend (that is what gives it watertight isolation), which
 * means GET /api/users lists it like any other profile. Every human-facing
 * surface — the switcher, the profile count, and especially the auto-select
 * that picks a profile on first load — must see only human profiles, or a
 * fresh session could silently open ON the harness scope and read machinery
 * as memory.
 */
/**
 * `/api/users` returns every directory in the store, which includes machinery
 * that is not a corpus anyone wants to look at. Left unfiltered the app opens
 * on whichever of those sorts first — `.mcp-shims` — and then every surface
 * truthfully reports an empty profile: graph fails to load, geo has no
 * coordinates, tasks answers 400. That reads as a broken product when nothing
 * is wrong except the selection.
 *
 * Excluded: seat-harness scratch profiles, anything dot-prefixed (internal
 * directories, never user data), and test fixtures left behind by PR work.
 */
export const isHumanProfile = (profile: string): boolean =>
  !profile.endsWith(".seat-harness") &&
  !profile.startsWith(".") &&
  profile !== "test" &&
  !profile.startsWith("test-") &&
  !profile.endsWith("-test");

export type Reachability =
  | { state: "online"; profiles: string[] }
  | { state: "unauthorized"; status: number }
  | { state: "offline"; detail: string };

/* ------------------------------------------------------------------ *
 * WHY A SCREEN IS EMPTY
 *
 * THE PROBE ALREADY TELLS THESE APART AND THE SCREENS DID NOT. Every view
 * gated on `reach.state !== "online"` and then said one thing — "…once the
 * memory server is running" — which is the OFFLINE sentence printed over the
 * UNAUTHORIZED state as well. Verified in the browser: with a wrong key the
 * status strip correctly read `Key rejected — 401 — set SHODH_API_KEY to the
 * server's key` while the body of the same screen told the reader to start a
 * server that was already running. Two diagnoses on one screen, and the larger
 * one was wrong.
 *
 * That collapse is exactly what `probeBackend` exists to prevent: it asks
 * `/api/users` rather than `/health` precisely so that a 401 proves the key is
 * wrong, and the distinction was being thrown away one branch later in five
 * views at once. So the discrimination lives HERE, next to the union that
 * carries it, and the views spend a sentence rather than a branch.
 *
 * `Key rejected` IS THE STATUS STRIP'S OWN STATE NAME, VERBATIM. That is the
 * state the two surfaces were contradicting each other about, so it is the one
 * where a reader who looks up at the corner and down at the stage must read
 * the same words, not two paraphrases they have to reconcile.
 *
 * THE UNREACHABLE TITLE DELIBERATELY DIVERGES. The strip says `Not running`,
 * which is a sharper remedy than "Offline" and right for a 26px strip that has
 * room for one. It is not right here, because `offline` covers two different
 * things: no answer at all, and an answer this client could not use
 * (`backend returned 500` — see `probeBackend`). A server that replied 500 IS
 * running, and a full-page heading asserting otherwise would send a reader to
 * restart something healthy — the same class of mistake this whole function
 * exists to stop. `Not connected` is true of both, and the specific evidence
 * (`reach.detail`) is one click away.
 *
 * CONNECTED-AND-GENUINELY-EMPTY IS NOT THIS FUNCTION'S CASE, and deliberately.
 * "This profile holds nothing" is a claim about a corpus, which only the view
 * that queried it can make; it returns null here so that the view's own empty
 * state — the one that can say what would put data there — is reached.
 * ------------------------------------------------------------------ */

export interface Outage {
  /** The state, in the status strip's words. */
  title: string;
  /** What it means for this screen, in one sentence. */
  body: string;
  /** The evidence for the diagnosis and the fix, behind the info affordance. */
  more: string;
}

/**
 * The account a screen should give of itself when it cannot render.
 *
 * `absent` is the caller's own one-sentence description of what would be on
 * the screen if the server were reachable — the sentence each view already
 * wrote. It is used for the offline case ONLY, because it is only true there:
 * over a rejected key it describes a server that is running fine.
 *
 * Null when the backend is reachable and authorized, which is the caller's
 * signal to carry on and decide for itself whether it has data.
 */
export function outageOf(reach: Reachability, absent: string): Outage | null {
  if (reach.state === "online") return null;

  if (reach.state === "unauthorized") {
    return {
      title: "Key rejected",
      // The server IS running, and saying so is the whole correction: it stops
      // a reader from restarting a healthy backend to fix an authentication
      // problem.
      body: `The memory server is running and answered ${reach.status}. It did not accept this key, so nothing on this screen could be read.`,
      more:
        "Set SHODH_API_KEY to the same key the shodh backend was started with, and reload. " +
        "Nothing is wrong with the profile or with what it holds — the request never got past authentication, so no part of this screen has been read yet.",
    };
  }

  return {
    title: "Not connected",
    body: absent,
    more: `The server did not answer: ${reach.detail}. Until it does, this screen has nothing to read and is not reporting an empty profile.`,
  };
}

export async function probeBackend(signal?: AbortSignal): Promise<Reachability> {
  try {
    const profiles = await api.get<string[]>("/api/users", signal);
    return { state: "online", profiles };
  } catch (err) {
    if (err instanceof ApiError) {
      return err.isAuthFailure
        ? { state: "unauthorized", status: err.status }
        : { state: "offline", detail: `backend returned ${err.status}` };
    }
    if (err instanceof NetworkError) return { state: "offline", detail: err.message };
    throw err;
  }
}
