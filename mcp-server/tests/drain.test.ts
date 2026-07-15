import { afterEach, describe, expect, it, vi } from "vitest";
import { DrainController, type TimerHandle } from "../drain";

type Result = { content: { type: string; text: string }[]; isError?: boolean };

const ABANDON: Result = {
  content: [{ type: "text", text: "ABANDONED" }],
  isError: true,
};

function real(text: string): Result {
  return { content: [{ type: "text", text }] };
}

function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (reason?: unknown) => void;
  const promise = new Promise<T>((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}

interface FakeTimer {
  fn: () => void;
  ms: number;
  handle: number;
  cleared: boolean;
}

function makeHarness(opts?: { graceMs?: number; writable?: boolean }) {
  const graceMs = opts?.graceMs ?? 60_000;
  const shutdownCalls: string[] = [];
  const logs: string[] = [];
  const state = { writable: opts?.writable ?? true };
  const timers: FakeTimer[] = [];
  let nextHandle = 1;

  const controller = new DrainController<Result>({
    graceMs,
    abandonResult: ABANDON,
    isOutputWritable: () => state.writable,
    shutdown: (reason) => shutdownCalls.push(reason),
    log: (message) => logs.push(message),
    setTimer: (fn, ms) => {
      const handle = nextHandle++;
      timers.push({ fn, ms, handle, cleared: false });
      return handle as unknown as TimerHandle;
    },
    clearTimer: (handle) => {
      const timer = timers.find((t) => t.handle === (handle as unknown as number));
      if (timer) timer.cleared = true;
    },
  });

  return {
    controller,
    shutdownCalls,
    logs,
    state,
    timers,
    activeTimers: () => timers.filter((t) => !t.cleared),
    fireGraceTimer: () => {
      const active = timers.filter((t) => !t.cleared);
      if (active.length === 0) throw new Error("no active grace timer to fire");
      active[0].fn();
    },
  };
}

describe("DrainController — nothing in flight", () => {
  it("shuts down immediately on stdin EOF and schedules no grace timer", () => {
    const h = makeHarness();
    expect(h.controller.isDraining).toBe(false);
    h.controller.onStdinClose("eof");
    expect(h.shutdownCalls).toEqual(["eof"]);
    expect(h.timers).toHaveLength(0);
  });

  it("shuts down exactly once when both stdin end and close fire (double-fire guard)", () => {
    const h = makeHarness();
    h.controller.onStdinClose("end-fired");
    h.controller.onStdinClose("close-fired");
    expect(h.shutdownCalls).toEqual(["end-fired"]);
  });
});

describe("DrainController — draining in-flight calls", () => {
  it("delivers the real in-flight response before exiting on stdin EOF", async () => {
    const h = makeHarness();
    const work = deferred<Result>();
    const tracked = h.controller.track(() => work.promise);
    expect(h.controller.inFlightCount).toBe(1);

    // Host closes stdin mid-flight: must NOT exit yet, must arm the grace timer.
    h.controller.onStdinClose("eof");
    expect(h.controller.isDraining).toBe(true);
    expect(h.shutdownCalls).toHaveLength(0);
    expect(h.activeTimers()).toHaveLength(1);
    expect(h.logs.some((l) => l.includes("draining before shutdown"))).toBe(true);

    // Backend finishes: caller gets the REAL result, not the abandon result.
    work.resolve(real("REAL"));
    await expect(tracked).resolves.toEqual(real("REAL"));

    // Only now do we shut down, and the grace timer is cleared.
    expect(h.shutdownCalls).toHaveLength(1);
    expect(h.timers[0].cleared).toBe(true);
    expect(h.controller.inFlightCount).toBe(0);

    // A late output-loss signal after completion is a harmless no-op.
    h.controller.onOutputLost("late");
    expect(h.shutdownCalls).toHaveLength(1);
  });

  it("waits for ALL in-flight calls before shutting down", async () => {
    const h = makeHarness();
    const w1 = deferred<Result>();
    const w2 = deferred<Result>();
    const t1 = h.controller.track(() => w1.promise);
    const t2 = h.controller.track(() => w2.promise);
    expect(h.controller.inFlightCount).toBe(2);

    h.controller.onStdinClose("eof");

    w1.resolve(real("a"));
    await expect(t1).resolves.toEqual(real("a"));
    expect(h.shutdownCalls).toHaveLength(0); // one still in flight
    expect(h.controller.inFlightCount).toBe(1);

    w2.resolve(real("b"));
    await expect(t2).resolves.toEqual(real("b"));
    expect(h.shutdownCalls).toHaveLength(1);
  });

  it("decrements the in-flight count even when the tool call throws", async () => {
    const h = makeHarness();
    const tracked = h.controller.track(async () => {
      throw new Error("boom");
    });
    await expect(tracked).rejects.toThrow("boom");
    expect(h.controller.inFlightCount).toBe(0);
  });

  it("does not shut down when a call settles before any stdin EOF", async () => {
    const h = makeHarness();
    await expect(h.controller.track(async () => real("ok"))).resolves.toEqual(real("ok"));
    expect(h.shutdownCalls).toHaveLength(0);
    expect(h.controller.isDraining).toBe(false);
  });
});

describe("DrainController — grace window expiry", () => {
  it("abandons a hung in-flight call with an error result, then exits", async () => {
    const h = makeHarness();
    const work = deferred<Result>(); // backend never responds
    const tracked = h.controller.track(() => work.promise);

    h.controller.onStdinClose("eof");
    expect(h.shutdownCalls).toHaveLength(0);

    // Grace window elapses.
    h.fireGraceTimer();
    await expect(tracked).resolves.toEqual(ABANDON);
    expect(h.shutdownCalls).toHaveLength(1);
    expect(h.shutdownCalls[0]).toContain("grace window");
  });

  it("returns the abandon result immediately for calls that start after expiry", async () => {
    const h = makeHarness();
    const work = deferred<Result>();
    const t1 = h.controller.track(() => work.promise);
    h.controller.onStdinClose("eof");
    h.fireGraceTimer();
    await expect(t1).resolves.toEqual(ABANDON);

    const t2 = h.controller.track(async () => real("late"));
    await expect(t2).resolves.toEqual(ABANDON);
  });
});

describe("DrainController — output stream gone", () => {
  it("shuts down immediately if stdout is already unwritable when stdin closes mid-flight", () => {
    const h = makeHarness({ writable: false });
    const work = deferred<Result>();
    void h.controller.track(() => work.promise);

    h.controller.onStdinClose("eof");
    expect(h.shutdownCalls).toHaveLength(1);
    expect(h.shutdownCalls[0]).toContain("not writable");
    expect(h.timers).toHaveLength(0); // no drain attempted
  });

  it("abandons and exits if stdout dies during the drain window", async () => {
    const h = makeHarness();
    const work = deferred<Result>();
    const tracked = h.controller.track(() => work.promise);
    h.controller.onStdinClose("eof");
    expect(h.shutdownCalls).toHaveLength(0);

    // stdout dies mid-drain: cannot deliver, so exit now (no abandon write).
    h.state.writable = false;
    h.controller.onOutputLost("stdout gone");
    expect(h.shutdownCalls).toEqual(["stdout gone"]);

    // The underlying work still resolves to its real value; process is exiting.
    work.resolve(real("late"));
    await expect(tracked).resolves.toEqual(real("late"));
  });

  it("ignores output loss when not draining", () => {
    const h = makeHarness();
    h.controller.onOutputLost("noise");
    expect(h.shutdownCalls).toHaveLength(0);
  });
});

describe("DrainController — default wiring (no injected log/timers)", () => {
  afterEach(() => {
    vi.useRealTimers();
    vi.restoreAllMocks();
  });

  it("uses console.error and real timers, clearing the timer on a normal drain", async () => {
    vi.useFakeTimers();
    const errSpy = vi.spyOn(console, "error").mockImplementation(() => {});
    const shutdownCalls: string[] = [];
    const controller = new DrainController<Result>({
      graceMs: 5_000,
      abandonResult: ABANDON,
      isOutputWritable: () => true,
      shutdown: (reason) => shutdownCalls.push(reason),
    });

    const work = deferred<Result>();
    const tracked = controller.track(() => work.promise);
    controller.onStdinClose("eof");
    expect(errSpy).toHaveBeenCalled(); // default log sink

    work.resolve(real("ok"));
    await expect(tracked).resolves.toEqual(real("ok"));
    expect(shutdownCalls).toHaveLength(1);

    // The real grace timer was cleared, so advancing time triggers nothing more.
    vi.advanceTimersByTime(10_000);
    expect(shutdownCalls).toHaveLength(1);
  });

  it("fires the default real timer on grace expiry", async () => {
    vi.useFakeTimers();
    vi.spyOn(console, "error").mockImplementation(() => {});
    const shutdownCalls: string[] = [];
    const controller = new DrainController<Result>({
      graceMs: 5_000,
      abandonResult: ABANDON,
      isOutputWritable: () => true,
      shutdown: (reason) => shutdownCalls.push(reason),
    });

    const work = deferred<Result>(); // hung
    const tracked = controller.track(() => work.promise);
    controller.onStdinClose("eof");

    vi.advanceTimersByTime(5_000);
    await expect(tracked).resolves.toEqual(ABANDON);
    expect(shutdownCalls).toHaveLength(1);
  });
});
