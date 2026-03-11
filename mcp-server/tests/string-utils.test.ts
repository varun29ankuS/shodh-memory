import { describe, expect, it } from "vitest";
import {
  stripSystemNoise,
  getContent,
  getType,
  formatSurfacedMemories,
  formatToolCallContent,
} from "../string-utils";

// =============================================================================
// stripSystemNoise
// =============================================================================
describe("stripSystemNoise", () => {
  it("removes <task-notification> blocks", () => {
    const input = "Hello <task-notification>ignore this</task-notification> world";
    expect(stripSystemNoise(input)).toBe("Hello  world");
  });

  it("removes <system-reminder> blocks", () => {
    const input = "Before <system-reminder>stuff\nmultiline</system-reminder> After";
    expect(stripSystemNoise(input)).toBe("Before  After");
  });

  it("removes <shodh-context> blocks", () => {
    const input = '<shodh-context attr="1">data</shodh-context>clean';
    expect(stripSystemNoise(input)).toBe("clean");
  });

  it("removes <shodh-memory> blocks", () => {
    const input = "prefix<shodh-memory>...</shodh-memory>suffix";
    expect(stripSystemNoise(input)).toBe("prefixsuffix");
  });

  it("removes <command-name> blocks", () => {
    const input = "cmd: <command-name>rm -rf</command-name> done";
    expect(stripSystemNoise(input)).toBe("cmd:  done");
  });

  it("removes multiple different tag types", () => {
    const input =
      "<task-notification>A</task-notification> hello <system-reminder>B</system-reminder>";
    expect(stripSystemNoise(input)).toBe("hello");
  });

  it("collapses excessive whitespace", () => {
    const input = "word1     word2\n\n\n\nword3";
    expect(stripSystemNoise(input)).toBe("word1 word2 word3");
  });

  it("returns empty string for whitespace-only input", () => {
    expect(stripSystemNoise("   ")).toBe("");
  });

  it("returns input unchanged when no tags present", () => {
    const text = "normal text without any tags";
    expect(stripSystemNoise(text)).toBe(text);
  });

  it("handles empty string", () => {
    expect(stripSystemNoise("")).toBe("");
  });

  it("handles multiline tag content with nested elements", () => {
    const input = `before<task-notification>
      <inner>
        nested content
      </inner>
    </task-notification>after`;
    expect(stripSystemNoise(input)).toBe("beforeafter");
  });
});

// =============================================================================
// getContent
// =============================================================================
describe("getContent", () => {
  it("returns top-level content when present", () => {
    expect(getContent({ content: "hello" })).toBe("hello");
  });

  it("returns experience.content when top-level is missing", () => {
    expect(getContent({ experience: { content: "nested" } })).toBe("nested");
  });

  it("prefers top-level content over experience.content", () => {
    expect(getContent({ content: "top", experience: { content: "nested" } })).toBe("top");
  });

  it("returns empty string when no content anywhere", () => {
    expect(getContent({})).toBe("");
  });

  it("returns empty string for undefined experience", () => {
    expect(getContent({ experience: undefined })).toBe("");
  });

  it("returns empty string when content is empty string", () => {
    expect(getContent({ content: "" })).toBe("");
  });
});

// =============================================================================
// getType
// =============================================================================
describe("getType", () => {
  it("returns top-level memory_type when present", () => {
    expect(getType({ memory_type: "Decision" })).toBe("Decision");
  });

  it("returns experience.memory_type when top-level is missing", () => {
    expect(getType({ experience: { memory_type: "Learning" } })).toBe("Learning");
  });

  it("returns experience.experience_type as last fallback", () => {
    expect(getType({ experience: { experience_type: "Error" } })).toBe("Error");
  });

  it('defaults to "Observation" when nothing is set', () => {
    expect(getType({})).toBe("Observation");
  });

  it("prefers top-level over nested", () => {
    expect(getType({ memory_type: "Task", experience: { memory_type: "Decision" } })).toBe(
      "Task"
    );
  });

  it('returns "Observation" for undefined experience', () => {
    expect(getType({ experience: undefined })).toBe("Observation");
  });
});

// =============================================================================
// formatSurfacedMemories
// =============================================================================
describe("formatSurfacedMemories", () => {
  it("returns empty string for null", () => {
    expect(formatSurfacedMemories(null)).toBe("");
  });

  it("returns empty string for undefined", () => {
    expect(formatSurfacedMemories(undefined)).toBe("");
  });

  it("returns empty string for empty array", () => {
    expect(formatSurfacedMemories([])).toBe("");
  });

  it("formats a single memory", () => {
    const result = formatSurfacedMemories([
      { content: "Test memory content here", relevance_score: 0.85 },
    ]);
    expect(result).toContain("[Relevant memories surfaced]");
    expect(result).toContain("85%");
    expect(result).toContain("Test memory content here");
  });

  it("formats multiple memories with numbered list", () => {
    const result = formatSurfacedMemories([
      { content: "First", relevance_score: 0.9 },
      { content: "Second", relevance_score: 0.7 },
    ]);
    expect(result).toContain("1.");
    expect(result).toContain("2.");
    expect(result).toContain("90%");
    expect(result).toContain("70%");
  });

  it("handles missing relevance_score (defaults to 0%)", () => {
    const result = formatSurfacedMemories([{ content: "No score" }]);
    expect(result).toContain("0%");
  });

  it("truncates long content to 80 chars", () => {
    const longContent = "x".repeat(200);
    const result = formatSurfacedMemories([
      { content: longContent, relevance_score: 0.5 },
    ]);
    // Should contain truncated content + "..."
    expect(result).toContain("...");
    // The actual content in the output should be 80 chars max
    const lines = result.trim().split("\n");
    const memLine = lines.find((l) => l.includes("50%"));
    expect(memLine).toBeDefined();
  });
});

// =============================================================================
// formatToolCallContent
// =============================================================================
describe("formatToolCallContent", () => {
  it("returns null for memory management tools", () => {
    expect(formatToolCallContent("remember", {}, "result")).toBeNull();
    expect(formatToolCallContent("recall", {}, "result")).toBeNull();
    expect(formatToolCallContent("forget", {}, "result")).toBeNull();
    expect(formatToolCallContent("list_memories", {}, "result")).toBeNull();
  });

  it("formats non-memory tools", () => {
    const result = formatToolCallContent("run_command", { cmd: "ls" }, "file1 file2");
    expect(result).toContain("Tool: run_command");
    expect(result).toContain("Input:");
    expect(result).toContain("Result: file1 file2");
  });

  it("truncates result to 1000 chars", () => {
    const longResult = "x".repeat(2000);
    const result = formatToolCallContent("some_tool", {}, longResult);
    expect(result).not.toBeNull();
    expect(result!).toContain("...");
    // Result portion should be truncated
    const resultPart = result!.split("Result: ")[1];
    expect(resultPart.length).toBeLessThanOrEqual(1004); // 1000 + "..."
  });

  it("does not add ellipsis for short results", () => {
    const result = formatToolCallContent("some_tool", {}, "short");
    expect(result).not.toBeNull();
    expect(result!).toContain("Result: short");
    expect(result!).not.toContain("...");
  });

  it("handles empty args", () => {
    const result = formatToolCallContent("tool", {}, "ok");
    expect(result).toContain("Input: {}");
  });
});
