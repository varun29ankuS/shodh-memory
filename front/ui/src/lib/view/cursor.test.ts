import { describe, expect, it } from "vitest";
import type { SeatEvent } from "@/lib/seat/types";
import { EMPTY_CURSOR, advance, type LiveTurn } from "./cursor";

const op = (id: string): SeatEvent => ({
  type: "memory_write",
  scope: "user",
  memory_id: id,
  memory_type: "Observation",
  content_preview: id,
  ledger_event_id: `l-${id}`,
});

const turn = (over: Partial<LiveTurn> = {}): LiveTurn => ({
  key: "c1#0",
  pending: true,
  ops: [],
  ...over,
});

describe("advance", () => {
  it("opens the authority window on a live turn it has not seen", () => {
    const step = advance(EMPTY_CURSOR, turn({ ops: [op("a")] }));
    expect(step.beginTurn).toBe(true);
    expect(step.fresh).toEqual([op("a")]);
    expect(step.cursor).toEqual({ key: "c1#0", consumed: 1 });
  });

  it("translates each op exactly once as the turn grows", () => {
    const first = advance(EMPTY_CURSOR, turn({ ops: [op("a")] }));
    const second = advance(first.cursor, turn({ ops: [op("a"), op("b")] }));

    expect(second.beginTurn).toBe(false);
    expect(second.fresh).toEqual([op("b")]);
  });

  it("does nothing when the turn is flushed with no new ops", () => {
    const first = advance(EMPTY_CURSOR, turn({ ops: [op("a")] }));
    const again = advance(first.cursor, turn({ ops: [op("a")] }));

    expect(again.fresh).toEqual([]);
    expect(again.beginTurn).toBe(false);
  });

  it("never translates a replayed turn, and never reopens the window for one", () => {
    // Reopening a conversation from the session list rebuilds every past turn's
    // ops with pending false. Translating them would walk the view through
    // everything the model has ever recalled, on a click that asked to READ.
    const step = advance(EMPTY_CURSOR, turn({ pending: false, ops: [op("a"), op("b")] }));

    expect(step.beginTurn).toBe(false);
    expect(step.fresh).toEqual([]);
    expect(step.cursor).toEqual({ key: "c1#0", consumed: 2 });
  });

  it("stops translating once the turn settles", () => {
    const live = advance(EMPTY_CURSOR, turn({ ops: [op("a")] }));
    // The seat's persisted detail arrives and replaces the turn with one more
    // op than the live reducer saw.
    const settled = advance(live.cursor, turn({ pending: false, ops: [op("a"), op("b")] }));

    expect(settled.fresh).toEqual([]);
    expect(settled.cursor.consumed).toBe(2);
  });

  it("treats a new turn in the same conversation as a new window", () => {
    const first = advance(EMPTY_CURSOR, turn({ ops: [op("a")] }));
    const next = advance(first.cursor, turn({ key: "c1#1", ops: [op("b")] }));

    expect(next.beginTurn).toBe(true);
    expect(next.fresh).toEqual([op("b")]);
  });

  it("treats a different conversation as a new window", () => {
    const first = advance(EMPTY_CURSOR, turn({ ops: [op("a")] }));
    const other = advance(first.cursor, turn({ key: "c2#0", ops: [op("b")] }));

    expect(other.beginTurn).toBe(true);
  });
});
