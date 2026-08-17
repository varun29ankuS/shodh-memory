import { api } from "@/lib/api";
import type { ReminderItem } from "./prospective";

/**
 * The one reminder route this product's UI is allowed to call.
 *
 * IN-FENCE RATHER THAN IN `lib/api`, on the precedent this repo already set for
 * `features/tasks/api.ts`: a route with exactly one caller and a contract worth
 * arguing about in comments belongs beside the screen that argues it.
 *
 * ─────────────────────────────────────────────────────────────────────────────
 * THREE OF THE EIGHT REMINDER ROUTES ARE READS THAT MUTATE. THEY ARE NOT
 * CALLED FROM HERE, AND THIS IS NOT A STYLE PREFERENCE.
 *
 * `GET`-shaped as they look, `/api/reminders/due` and `/api/reminders/check`
 * both take `mark_triggered` with `#[serde(default = "default_true")]`
 * (src/handlers/todos.rs:117-123, 137-138) and, when it is set, walk their
 * results calling `mark_triggered` on each (todos.rs:715-740). A dashboard that
 * polled `/due` on a sixty-second clock would silently flip every reminder in
 * the store from `Pending` to `Triggered` — and the background poller
 * (src/server.rs) and the MCP tools are the only things that have ever surfaced
 * these to anyone. The web UI would be quietly consuming the notification that
 * an agent was supposed to deliver.
 *
 * `POST /api/reminders` is the read with no side effect: `list_reminders`
 * (todos.rs:644-699) calls `prospective_store.list_for_user` and maps. That is
 * the only route below.
 *
 * DISMISS AND DELETE ARE ALSO ABSENT, AND FOR A DIFFERENT REASON. There are six
 * handlers — create, list, due, check, dismiss, delete — and NO inverse of
 * dismiss: `ProspectiveTaskStatus::Dismissed` has no route back to `Pending`.
 * Tasks holds the line that every action is reversible from where it lands
 * (see `TaskActions`), and a dismissal that cannot be undone would be the one
 * irreversible control in the product, shipped on a front page, on the object a
 * person is least likely to be able to reconstruct. So this section reads and
 * does not act, which is also `Learning.tsx`'s "no controls, deliberately".
 * ─────────────────────────────────────────────────────────────────────────────
 */

/** `ListRemindersResponse` — src/handlers/todos.rs:107-111. */
export interface ListRemindersResponse {
  reminders: ReminderItem[];
  count: number;
}

/**
 * `POST /api/reminders` — src/handlers/router.rs:372.
 *
 * `status` accepts `pending | triggered | dismissed | expired | all`, and an
 * unrecognised value is a 400 rather than a silent empty list
 * (todos.rs:655-666). `"all"` is asked for deliberately: the section shows what
 * is still standing, which is TWO of those statuses, and the request takes one.
 * Two calls to filter server-side would double the requests to save a filter
 * the screen has to be able to do anyway — `standingReminders` decides what is
 * standing, in one place, under test.
 *
 * There is no limit or cursor on `ListRemindersRequest` (todos.rs:84-88), so
 * this returns a profile's whole reminder history including closed ones. That
 * is two rows on the largest profile on this instance; the cap that matters is
 * on what is DRAWN, and it lives in `standingReminders`.
 */
export function listReminders(
  userId: string,
  signal?: AbortSignal,
): Promise<ListRemindersResponse> {
  return api.post<ListRemindersResponse>("/api/reminders", { user_id: userId, status: "all" }, signal);
}
