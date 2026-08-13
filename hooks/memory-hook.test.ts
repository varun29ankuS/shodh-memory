import { describe, expect, it } from "bun:test";
import {
  buildPreToolContext,
  formatRelativeTime,
  formatMemoriesForContext,
  isErrorOutput,
  describeKeyOrigin,
  reportAuthFailure,
} from "./memory-hook";

describe("formatRelativeTime", () => {
  it("returns today for current date", () => {
    const nowIso = new Date().toISOString();
    expect(formatRelativeTime(nowIso)).toBe("today");
  });

  it("returns yesterday for one day old date", () => {
    const d = new Date(Date.now() - 24 * 60 * 60 * 1000).toISOString();
    expect(formatRelativeTime(d)).toBe("yesterday");
  });

  it("returns Xd ago for dates under a week", () => {
    const d = new Date(Date.now() - 3 * 24 * 60 * 60 * 1000).toISOString();
    expect(formatRelativeTime(d)).toBe("3d ago");
  });

  it("returns calendar date for older memories", () => {
    const d = new Date(Date.now() - 10 * 24 * 60 * 60 * 1000).toISOString();
    const value = formatRelativeTime(d);
    expect(value).not.toContain("ago");
    expect(value).not.toBe("today");
    expect(value).not.toBe("yesterday");
  });
});

describe("formatMemoriesForContext", () => {
  const memory = (over: Partial<Record<string, unknown>> = {}) => ({
    id: "m1",
    content: "Remember to review deployment checklist before release",
    memory_type: "Task",
    score: 0.83,
    importance: 0.7,
    created_at: new Date().toISOString(),
    tags: ["deploy"],
    relevance_reason: "matches query",
    matched_entities: ["release"],
    ...over,
  });

  // Returns SurfaceResult | null, not a string: callers need `meta` for the
  // surfacing decision, and null is how "nothing worth surfacing" is expressed.
  it("returns null for empty input", () => {
    expect(formatMemoriesForContext([])).toBeNull();
  });

  it("returns null when the best score is under the noise floor", () => {
    expect(formatMemoriesForContext([memory({ score: 0.01 })])).toBeNull();
  });

  // Displayed percentages are normalised across the set, so a lone memory is
  // always 100% — its raw score survives on meta.bestScore.
  it("normalises the displayed score and keeps the raw one in meta", () => {
    const out = formatMemoriesForContext([memory()]);
    expect(out).not.toBeNull();
    expect(out!.text).toContain("100% match");
    expect(out!.text).toContain("today");
    expect(out!.text).toContain("Remember to review deployment checklist");
    expect(out!.meta.bestScore).toBe(83);
    expect(out!.meta.count).toBe(1);
  });

  it("spreads the set across the normalised range", () => {
    const now = new Date().toISOString();
    const out = formatMemoriesForContext([
      memory({ id: "m1", content: "A", score: 0.2, created_at: now }),
      memory({ id: "m2", content: "B", score: 0.3, created_at: now }),
    ]);
    expect(out).not.toBeNull();
    expect((out!.text.match(/\u2022/g) || []).length).toBe(2);
    expect(out!.text).toContain("0% match");
    expect(out!.text).toContain("100% match");
  });

  it("truncates long content at 120 chars with an ellipsis", () => {
    const out = formatMemoriesForContext([memory({ content: "x".repeat(130) })]);
    expect(out).not.toBeNull();
    expect(out!.text).toContain("...");
    expect(out!.text).toContain("x".repeat(120));
    expect(out!.text).not.toContain("x".repeat(121));
  });
});

describe("buildPreToolContext", () => {
  it("builds edit context", () => {
    expect(buildPreToolContext("Edit", { file_path: "src/main.ts" })).toBe("Editing file: src/main.ts");
  });

  it("builds write context from the same branch", () => {
    expect(buildPreToolContext("Write", { file_path: "src/new.ts" })).toBe("Editing file: src/new.ts");
  });

  // Bash deliberately has no branch of its own. handlePreToolUse returns before
  // calling this for anything but Edit/Write (see the guard in that handler),
  // and routing a raw shell command through here would send it to the memory
  // server — commands carry credentials often enough that the narrow surface
  // is the point, not an oversight.
  it("falls through to the generic form for tools it no longer specialises", () => {
    expect(buildPreToolContext("Bash", { command: "curl -H 'X-API-Key: sk-live-abc' https://api" }))
      .toBe("About to use Bash");
    expect(buildPreToolContext("Read", {})).toBe("About to use Read");
  });

  it("falls back when the expected input field is absent", () => {
    expect(buildPreToolContext("Edit", {})).toBe("About to use Edit");
  });
});

describe("isErrorOutput", () => {
  it("detects the error shapes it targets", () => {
    expect(isErrorOutput("error[E0308]: mismatched types")).toBe(true);
    expect(isErrorOutput("Operation FAILED quickly")).toBe(true);
    expect(isErrorOutput("thread 'main' panicked at src/lib.rs:4")).toBe(true);
    expect(isErrorOutput("fatal: not a git repository")).toBe(true);
    expect(isErrorOutput("process exited with exit code 1")).toBe(true);
    expect(isErrorOutput("bash: foo: command not found")).toBe(true);
  });

  // Cargo, npm and tsc all prefix diagnostics with a lowercase "error:", so a
  // case-sensitive match missed the most common failure line in this repo.
  // Only this pattern is case-insensitive — see the trade pinned below.
  it("detects lowercase error: diagnostics", () => {
    expect(isErrorOutput("error: could not compile `shodh-memory`")).toBe(true);
    expect(isErrorOutput("Error: connect ECONNREFUSED")).toBe(true);
  });

  it("returns false on clean output", () => {
    expect(isErrorOutput("Command completed successfully")).toBe(false);
  });

  // The matching is case-sensitive on FAILED by design. Relaxing it would make
  // "0 tests failed" — an ordinary green-build line — read as a failure, and a
  // false positive here mislabels a successful run in memory, which is worse
  // than missing one. These cases pin that trade so it is not relaxed by accident.
  it("does not fire on clean lines that merely contain the words", () => {
    expect(isErrorOutput("0 tests failed")).toBe(false);
    expect(isErrorOutput("0 errors, 0 warnings")).toBe(false);
    expect(isErrorOutput("compiled without errors")).toBe(false);
    expect(isErrorOutput("build succeeded")).toBe(false);
  });
});

describe("describeKeyOrigin", () => {
  const SOURCES = ["env", "shared-key-file", "legacy-dev-fallback"] as const;

  // The regression this pins: the origin string is written to stderr AND into
  // the user-visible systemMessage on an auth failure. A previous version
  // interpolated the resolved key-file path, which sits under the operator's
  // home directory and therefore carries their username.
  it("never carries a filesystem path", () => {
    for (const source of SOURCES) {
      const text = describeKeyOrigin(source);
      expect(text).not.toContain("/");
      expect(text).not.toContain("\\");
      expect(text).not.toContain(".api-key");
    }
  });

  it("still distinguishes every source", () => {
    const described = SOURCES.map(describeKeyOrigin);
    expect(new Set(described).size).toBe(SOURCES.length);
    expect(describeKeyOrigin("env")).toContain("SHODH_API_KEY");
  });

  it("is pure — no call ordering can change the answer", () => {
    const before = describeKeyOrigin("shared-key-file");
    describeKeyOrigin("env");
    describeKeyOrigin("legacy-dev-fallback");
    expect(describeKeyOrigin("shared-key-file")).toBe(before);
  });
});

describe("reportAuthFailure output", () => {
  // describeKeyOrigin is pure and easy to assert, but what ships is what this
  // function writes: a stderr block AND a systemMessage on stdout that the
  // client surfaces to the user. Both were flagged on main. Assert the bytes.
  const capture = (fn: () => void) => {
    const err: string[] = [];
    const out: string[] = [];
    const origErr = console.error;
    const origLog = console.log;
    console.error = (...a: unknown[]) => { err.push(a.join(" ")); };
    console.log = (...a: unknown[]) => { out.push(a.join(" ")); };
    try { fn(); } finally { console.error = origErr; console.log = origLog; }
    return { err: err.join("\n"), out: out.join("\n") };
  };

  // The "<data-dir>/.api-key" placeholder in the fix instructions is fine and
  // deliberate — it tells the operator what to look for. What must never appear
  // is a RESOLVED path, which carries the home directory and therefore the
  // account name.
  it("emits no resolved filesystem path on either stream", () => {
    const { err, out } = capture(() => reportAuthFailure(401));
    const home = process.env.HOME ?? "\u0000never";
    for (const stream of [err, out]) {
      expect(stream).not.toContain(home);
      expect(stream).not.toContain("Application Support");
      expect(stream).not.toContain("/.local/share");
      // No absolute path of any kind: the only "/" allowed is in the placeholder.
      expect(stream.replace(/<data-dir>\/\.api-key/g, "")).not.toMatch(/\s\/[A-Za-z]/);
    }
  });

  it("still tells the operator the status and where the key came from", () => {
    const { err, out } = capture(() => reportAuthFailure(403));
    expect(err).toContain("403");
    expect(err).toContain("Key came from");
    const parsed = JSON.parse(out);
    expect(parsed.systemMessage).toContain("403");
    expect(parsed.hookSpecificOutput.hookEventName).toBe("SessionStart");
  });

  it("emits a single parseable JSON object on stdout", () => {
    const { out } = capture(() => reportAuthFailure(401));
    expect(() => JSON.parse(out)).not.toThrow();
  });
});
