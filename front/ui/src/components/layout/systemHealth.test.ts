import { describe, expect, it } from "vitest";
import {
  alertsOf,
  describeAge,
  freshnessOf,
  readAssistant,
  readMemory,
  ribbonToneOf,
  STALE_AFTER_MS,
  type Freshness,
  type ServiceReading,
} from "./systemHealth";

const fresh: Freshness = { kind: "fresh", ageMs: 4_000 };
const stale: Freshness = { kind: "stale", ageMs: 180_000 };
const unprobed: Freshness = { kind: "unprobed" };

describe("freshnessOf", () => {
  it("calls a reading that has never resolved unprobed, not stale", () => {
    // `dataUpdatedAt` is 0 before the first probe resolves. Treating that as a
    // very old reading would paint an alarm on every page load, before anything
    // has been asked.
    expect(freshnessOf(0, 1_000_000)).toEqual({ kind: "unprobed" });
  });

  it("holds a reading fresh right up to the threshold", () => {
    expect(freshnessOf(1_000, 1_000 + STALE_AFTER_MS)).toEqual({
      kind: "fresh",
      ageMs: STALE_AFTER_MS,
    });
  });

  it("goes stale one millisecond past it", () => {
    expect(freshnessOf(1_000, 1_000 + STALE_AFTER_MS + 1)).toEqual({
      kind: "stale",
      ageMs: STALE_AFTER_MS + 1,
    });
  });

  it("survives a full poll interval without flickering", () => {
    // The probes poll every 10s. If one is slow or dropped, the ribbon must not
    // strobe between confirmed and unconfirmed — the threshold is 2.5 intervals
    // for exactly this reason.
    expect(freshnessOf(0 + 1, 1 + 20_000).kind).toBe("fresh");
  });

  it("reads a clock that moved backwards as age zero, never a negative age", () => {
    // A negative age would print as "-3s ago", which reads as a bug in the
    // product rather than a bug in the clock.
    expect(freshnessOf(9_000, 5_000)).toEqual({ kind: "fresh", ageMs: 0 });
  });
});

describe("describeAge", () => {
  it("says just now below three seconds, so a 1s tick does not jitter", () => {
    expect(describeAge(0)).toBe("just now");
    expect(describeAge(2_999)).toBe("just now");
  });

  it("counts seconds, which is the scale a 10s poll lives at", () => {
    // The reason lib/format.ts::relativeDay is not reused: its finest step is a
    // minute, so every age this ribbon reports would collapse to "now".
    expect(describeAge(3_000)).toBe("3s ago");
    expect(describeAge(59_999)).toBe("59s ago");
  });

  it("rolls up to minutes, hours and days", () => {
    expect(describeAge(60_000)).toBe("1m ago");
    expect(describeAge(59 * 60_000)).toBe("59m ago");
    expect(describeAge(60 * 60_000)).toBe("1h ago");
    expect(describeAge(23 * 3_600_000)).toBe("23h ago");
    expect(describeAge(24 * 3_600_000)).toBe("1d ago");
  });
});

describe("readMemory", () => {
  it("reports a reachable server with profiles as live, and counts them", () => {
    const r = readMemory({ state: "online", profiles: ["varun", "gdelt-bridge"] }, fresh);
    expect(r.tone).toBe("live");
    expect(r.state).toBe("Live");
    expect(r.evidence).toContain("2 profiles");
    // Nothing is lost, so nothing is claimed to be lost. A consequence in the
    // healthy case is what turns a status line into noise.
    expect(r.consequence).toBeNull();
    expect(r.remedy).toBeNull();
  });

  it("counts only human profiles, so machinery does not read as a corpus", () => {
    const r = readMemory(
      { state: "online", profiles: ["varun", ".mcp-shims", "varun.seat-harness", "test"] },
      fresh,
    );
    expect(r.evidence).toContain("1 profile");
    expect(r.evidence).not.toContain("4 profile");
  });

  it("does not pluralise a single profile", () => {
    expect(readMemory({ state: "online", profiles: ["varun"] }, fresh).evidence).toContain(
      "1 profile ",
    );
  });

  it("treats a reachable server holding nothing as waiting-on, not wrong", () => {
    const r = readMemory({ state: "online", profiles: [".mcp-shims"] }, fresh);
    expect(r.state).toBe("No profiles");
    expect(r.tone).toBe("warn");
    expect(r.consequence).toContain("nothing is stored yet");
    expect(r.remedy).not.toBeNull();
  });

  it("blames the key, not the server, when the key is rejected", () => {
    const r = readMemory({ state: "unauthorized", status: 401 }, fresh);
    expect(r.state).toBe("Key rejected");
    expect(r.tone).toBe("alarm");
    // The correction that matters: the server IS running, so nobody is sent to
    // restart a healthy backend over an authentication problem.
    expect(r.consequence).toContain("is running");
    expect(r.remedy).toContain("SHODH_API_KEY");
    expect(r.evidence).toContain("401");
  });

  it("quotes 403 as 403 — the other auth failure the client classifies", () => {
    expect(readMemory({ state: "unauthorized", status: 403 }, fresh).evidence).toContain("403");
  });

  it("tells someone to START a server only when nothing answered", () => {
    const r = readMemory({ state: "offline", detail: "Failed to fetch" }, fresh);
    expect(r.state).toBe("Not running");
    expect(r.tone).toBe("warn");
    expect(r.remedy).toBe("Start the shodh backend.");
    expect(r.evidence).toContain("Failed to fetch");
  });

  it("REFUSES to tell someone to start a server that answered 500", () => {
    // THE DEFECT THIS FILE EXISTS FOR. The old strip printed
    // `Not running — start the shodh backend` over a backend that had answered
    // 500 and was therefore already up. lib/api/health.ts documents the same
    // distinction at length for the full-page case and the strip threw it away.
    const r = readMemory(
      { state: "offline", detail: "backend returned 500", answered: 500 },
      fresh,
    );
    expect(r.state).not.toBe("Not running");
    expect(r.state).toBe("Erroring");
    expect(r.remedy).not.toContain("Start the shodh backend");
    expect(r.remedy).toContain("already up");
    expect(r.consequence).toContain("is running");
    expect(r.evidence).toContain("500");
  });

  it("escalates an erroring server above an absent one", () => {
    // Absent is ordinary — it is a local process someone starts by hand.
    // Answering wrongly is not ordinary, and the tones must not agree.
    const absent = readMemory({ state: "offline", detail: "Failed to fetch" }, fresh);
    const erroring = readMemory({ state: "offline", detail: "…", answered: 502 }, fresh);
    expect(absent.tone).toBe("warn");
    expect(erroring.tone).toBe("alarm");
  });

  it("says nothing it cannot support before the first probe resolves", () => {
    const r = readMemory({ state: "offline", detail: "not probed yet" }, unprobed);
    expect(r.state).toBe("Checking…");
    expect(r.tone).toBe("unknown");
    // Crucially NOT "Not running": at this point nothing has been asked, and
    // announcing an outage on every page load is how a status surface teaches
    // people to ignore it.
    expect(r.tone).not.toBe("warn");
    expect(r.consequence).toBeNull();
  });

  it("WITHDRAWS the green claim when the reading has gone stale", () => {
    // "Live" asserts the server is answering NOW. This app stops polling in a
    // background tab and does not refetch on focus, so a green ribbon can be an
    // hour old and describe a server that has since died.
    const r = readMemory({ state: "online", profiles: ["varun"] }, stale);
    expect(r.state).toBe("Unconfirmed");
    expect(r.tone).toBe("unknown");
    expect(r.tone).not.toBe("live");
    expect(r.consequence).toContain("stopped hearing");
    expect(r.remedy).not.toBeNull();
  });

  it("does NOT withdraw a failure claim when it goes stale", () => {
    // The asymmetry is deliberate: a server that was down two minutes ago is
    // still, in all likelihood, down. Downgrading that to "not sure" would hide
    // a real outage.
    const r = readMemory({ state: "offline", detail: "Failed to fetch" }, stale);
    expect(r.state).toBe("Not running");
    expect(r.tone).toBe("warn");
  });

  it("puts the age of every reading in its evidence", () => {
    expect(readMemory({ state: "online", profiles: ["v"] }, fresh).evidence).toContain(
      "checked 4s ago",
    );
    expect(readMemory({ state: "offline", detail: "x" }, stale).evidence).toContain(
      "checked 3m ago",
    );
  });
});

describe("readAssistant", () => {
  it("reports a seat whose backend answers as ready", () => {
    const r = readAssistant({ state: "online", backendOk: true, backendDetail: "healthy" }, true, fresh);
    expect(r.tone).toBe("live");
    expect(r.state).toBe("Ready");
    expect(r.consequence).toBeNull();
  });

  it("says what is LOST when the seat is not running, not merely that it is not running", () => {
    // The omission the whole ribbon exists to fix: the old strip reported the
    // memory server only, so "Connected" was displayed over a dead assistant.
    const r = readAssistant({ state: "offline", detail: "seat answered 502" }, true, fresh);
    expect(r.state).toBe("Not running");
    expect(r.tone).toBe("warn");
    expect(r.consequence).toContain("move this view");
    expect(r.consequence).toContain("tasks");
    expect(r.remedy).toContain("3141");
    expect(r.evidence).toContain("seat answered 502");
  });

  it("raises an ALARM when the seat cannot reach memory that this page can read", () => {
    // Silently fatal and invisible everywhere else: the assistant answers
    // normally and remembers nothing, because it is pointed at a different
    // backend or holds a different key.
    const r = readAssistant(
      { state: "online", backendOk: false, backendDetail: "connection refused" },
      true,
      fresh,
    );
    expect(r.state).toBe("No memory");
    expect(r.tone).toBe("alarm");
    expect(r.consequence).toContain("recall and remember nothing");
    expect(r.remedy).toContain("SHODH_API_URL");
    expect(r.evidence).toContain("connection refused");
  });

  it("does NOT double-report one outage seen from two vantage points", () => {
    // The memory server being down is already stated by the memory row, with a
    // better remedy. The seat process itself is genuinely fine, and a second
    // alarm for the same fact is how a status surface becomes wallpaper.
    const r = readAssistant(
      { state: "online", backendOk: false, backendDetail: "connection refused" },
      false,
      fresh,
    );
    expect(r.tone).toBe("live");
    expect(r.state).toBe("Ready");
    expect(r.state).not.toBe("No memory");
    // It is still honest about why: the evidence records that it is waiting on
    // the same server this page is.
    expect(r.evidence).toContain("waiting on the memory server");
  });

  it("withdraws its own green claim when stale, on the same rule", () => {
    const r = readAssistant({ state: "online", backendOk: true, backendDetail: "healthy" }, true, stale);
    expect(r.tone).toBe("unknown");
    expect(r.state).toBe("Unconfirmed");
  });

  it("is checking, not broken, before its first probe resolves", () => {
    const r = readAssistant({ state: "offline", detail: "not probed yet" }, false, unprobed);
    expect(r.state).toBe("Checking…");
    expect(r.tone).toBe("unknown");
  });
});

describe("ribbonToneOf", () => {
  const reading = (tone: ServiceReading["tone"]): ServiceReading => ({
    id: "memory",
    service: "x",
    state: "x",
    tone,
    consequence: null,
    remedy: null,
    evidence: "x",
  });

  it("takes the worst service, so one healthy one cannot mask a dead one", () => {
    expect(ribbonToneOf([reading("live"), reading("alarm")])).toBe("alarm");
    expect(ribbonToneOf([reading("warn"), reading("live")])).toBe("warn");
    expect(ribbonToneOf([reading("alarm"), reading("warn")])).toBe("alarm");
  });

  it("never paints an unconfirmed reading as health", () => {
    expect(ribbonToneOf([reading("live"), reading("unknown")])).toBe("unknown");
  });

  it("ranks a known failure above an unconfirmed reading", () => {
    expect(ribbonToneOf([reading("unknown"), reading("warn")])).toBe("warn");
  });

  it("is live only when every service is live", () => {
    expect(ribbonToneOf([reading("live"), reading("live")])).toBe("live");
  });
});

describe("alertsOf", () => {
  const reading = (id: ServiceReading["id"], tone: ServiceReading["tone"]): ServiceReading => ({
    id,
    service: id,
    state: "x",
    tone,
    consequence: null,
    remedy: null,
    evidence: "x",
  });

  it("stays silent while everything is healthy", () => {
    expect(alertsOf([reading("memory", "live"), reading("assistant", "live")])).toEqual([]);
  });

  it("stays silent on the first frame, when both services are merely unchecked", () => {
    // Otherwise the banner opens and closes on every page load, announcing a
    // problem that does not exist.
    expect(alertsOf([reading("memory", "unknown"), reading("assistant", "unknown")])).toEqual([]);
  });

  it("surfaces only the services that have something to say", () => {
    const alerts = alertsOf([reading("memory", "live"), reading("assistant", "warn")]);
    expect(alerts.map((a) => a.id)).toEqual(["assistant"]);
  });

  it("puts the wrong thing above the merely-absent thing", () => {
    const alerts = alertsOf([reading("memory", "warn"), reading("assistant", "alarm")]);
    expect(alerts.map((a) => a.id)).toEqual(["assistant", "memory"]);
  });
});
