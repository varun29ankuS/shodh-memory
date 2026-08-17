import { describe, expect, it } from "vitest";
import {
  isStanding,
  lateBy,
  reminderMeta,
  spanLabel,
  standingReminders,
  type ReminderItem,
} from "./prospective";

/**
 * The failure this file exists to catch is a context reminder drawn as an
 * overdue one.
 *
 * `due_at` is null on every standing reminder on the live profile, so every
 * lateness path here is exercised against data that does not have one. A bug
 * that treats "no due instant" as "due at epoch" renders two five-month-old
 * asks in the alarm colour and looks entirely plausible on screen.
 */

const NOW = Date.parse("2026-08-17T12:00:00Z");

/**
 * The live shape, taken from `POST /api/reminders` on `claude-code`: two rows,
 * both `context`-triggered, both `pending`, `due_at` and `triggered_at` null on
 * both, asked for on 2026-03-20 and 2026-02-13. The timed cases below are
 * synthetic because this instance holds none — which is exactly why they are
 * tested rather than assumed.
 */
const reminder = (extra: Partial<ReminderItem> = {}): ReminderItem => ({
  id: "r1",
  content: "Check if mrmartan confirmed the RocksDB lock fix works (PR #117/#118)",
  trigger_type: "context",
  status: "pending",
  due_at: null,
  created_at: "2026-03-20T16:18:36.279482Z",
  triggered_at: null,
  ...extra,
});

describe("spanLabel", () => {
  it("names months rather than counting days at that distance", () => {
    // 150 days. "150 days ago" is a number a reader has to convert.
    expect(spanLabel(150 * 24 * 3600_000)).toBe("5 months");
  });

  it("never prints a zero-valued span", () => {
    expect(spanLabel(0)).toBe("moments");
    expect(spanLabel(30_000)).toBe("moments");
    expect(spanLabel(-5_000)).toBe("moments");
  });

  it("uses the article rather than the digit at one of each unit", () => {
    expect(spanLabel(90 * 60_000)).toBe("an hour");
    expect(spanLabel(36 * 3600_000)).toBe("a day");
    expect(spanLabel(9 * 24 * 3600_000)).toBe("a week");
  });
});

describe("isStanding", () => {
  it("keeps a triggered reminder, which was raised but never acknowledged", () => {
    expect(isStanding(reminder({ status: "triggered" }))).toBe(true);
  });

  it("drops the two closed statuses", () => {
    expect(isStanding(reminder({ status: "dismissed" }))).toBe(false);
    expect(isStanding(reminder({ status: "expired" }))).toBe(false);
  });
});

describe("lateBy", () => {
  it("is null for a context reminder, which has no due instant at all", () => {
    // The whole point. OnContext#due_at() returns None (types.rs:3637-3639),
    // so there is nothing for "now" to be past.
    expect(lateBy(reminder(), NOW)).toBeNull();
  });

  it("is null for a timed reminder whose instant has not passed", () => {
    expect(lateBy(reminder({ trigger_type: "time", due_at: "2026-08-18T12:00:00Z" }), NOW)).toBeNull();
  });

  it("measures a timed reminder that has passed", () => {
    expect(lateBy(reminder({ trigger_type: "time", due_at: "2026-08-14T12:00:00Z" }), NOW)).toBe(
      3 * 24 * 3600_000,
    );
  });
});

describe("reminderMeta", () => {
  it("says when a context reminder fires, and never a date", () => {
    // Floored, never rounded up: this ask is 4 months and 28 days old and the
    // gutter says four. A span may under-report its age; it may never claim to
    // be older than it is.
    expect(reminderMeta(reminder(), NOW)).toEqual(["when it comes up", "asked 4 months ago"]);
  });

  it("states lateness for a timed reminder", () => {
    const meta = reminderMeta(
      reminder({ trigger_type: "time", due_at: "2026-08-14T12:00:00Z" }),
      NOW,
    );
    expect(meta[0]).toBe("3 days late");
  });

  it("counts down to a timed reminder still ahead", () => {
    const meta = reminderMeta(
      reminder({ trigger_type: "duration", due_at: "2026-08-19T12:00:00Z" }),
      NOW,
    );
    expect(meta[0]).toBe("due in 2 days");
  });

  it("reports when it was raised in place of when it was asked", () => {
    const meta = reminderMeta(
      reminder({ status: "triggered", triggered_at: "2026-08-15T12:00:00Z" }),
      NOW,
    );
    expect(meta).toEqual(["when it comes up", "raised 2 days ago"]);
  });
});

describe("standingReminders", () => {
  it("returns null rather than an empty board when nothing is standing", () => {
    // A heading over no rows is furniture, and "0 reminders" is a claim about a
    // subsystem most profiles have never touched.
    expect(standingReminders([], NOW)).toBeNull();
    expect(standingReminders([reminder({ status: "dismissed" })], NOW)).toBeNull();
  });

  it("flags a passed instant and nothing else, on a page holding all three kinds", () => {
    // The context row and the not-yet-due row must be indistinguishable here:
    // "has no deadline" and "has one that has not arrived" are both NOT LATE,
    // and a rule keyed on trigger type rather than on the instant would colour
    // `ahead` red the moment it was created.
    const board = standingReminders(
      [
        reminder({ id: "ctx", created_at: "2026-03-20T09:00:00Z" }),
        reminder({ id: "ahead", trigger_type: "time", due_at: "2026-09-01T12:00:00Z" }),
        reminder({ id: "passed", trigger_type: "time", due_at: "2026-08-10T12:00:00Z" }),
      ],
      NOW,
    );
    expect(board?.rows.map((r) => [r.id, r.late])).toEqual([
      ["passed", true],
      ["ahead", false],
      ["ctx", false],
    ]);
  });

  it("ranks every deadline by how soon it is owed, then the most recent ask", () => {
    // Ascending due instant is the whole rule: it puts the two that have passed
    // above the two that have not, most overdue first, with no separate pass
    // for lateness. Context reminders cannot enter this band at all.
    const board = standingReminders(
      [
        reminder({ id: "ctx-old", created_at: "2026-02-01T09:00:00Z" }),
        reminder({ id: "ctx-new", created_at: "2026-07-01T09:00:00Z" }),
        reminder({ id: "soon", trigger_type: "time", due_at: "2026-08-18T12:00:00Z" }),
        reminder({ id: "later", trigger_type: "time", due_at: "2026-09-01T12:00:00Z" }),
        reminder({ id: "late-a", trigger_type: "time", due_at: "2026-08-16T12:00:00Z" }),
        reminder({ id: "late-b", trigger_type: "time", due_at: "2026-06-01T12:00:00Z" }),
      ],
      NOW,
      6,
    );
    expect(board?.rows.map((r) => r.id)).toEqual([
      "late-b",
      "late-a",
      "soon",
      "later",
      "ctx-new",
      "ctx-old",
    ]);
  });

  it("counts what the cap left off rather than showing a subset as the whole", () => {
    const many = Array.from({ length: 7 }, (_, i) =>
      reminder({ id: `r${i}`, created_at: `2026-0${i + 1}-01T09:00:00Z` }),
    );
    const board = standingReminders(many, NOW, 4);
    expect(board?.rows).toHaveLength(4);
    expect(board?.hidden).toBe(3);
  });

  it("collapses a stored newline so one reminder stays one row", () => {
    const board = standingReminders([reminder({ content: "ask Varun\n\nabout  the deck" })], NOW);
    expect(board?.rows[0].text).toBe("ask Varun about the deck");
  });
});
