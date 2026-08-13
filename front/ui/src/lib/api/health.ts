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
