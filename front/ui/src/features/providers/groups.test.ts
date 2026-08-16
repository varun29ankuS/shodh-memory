import { describe, expect, it } from "vitest";
import type { ProviderInfo } from "@/lib/seat/types";
import { groupProviders, isIdleLocal, matchesFilter, showAvailable } from "./groups";

/**
 * Two failures this file exists to prevent, both observed on the real install:
 * a local endpoint that nothing is running counted as "Ready", and a filter
 * whose single match stayed folded away behind a collapsed disclosure while
 * the screen said "No provider is configured yet".
 */

const provider = (extra: Partial<ProviderInfo> = {}): ProviderInfo => ({
  id: "acme",
  name: "Acme",
  configured: false,
  source: null,
  auth_type: null,
  stored: false,
  accepts_api_key: true,
  oauth_available: false,
  oauth_subscription: false,
  oauth_label: null,
  model_count: 4,
  local: false,
  ...extra,
});

const anthropic = provider({
  id: "anthropic",
  name: "Anthropic",
  configured: true,
  stored: true,
  source: "stored key",
  model_count: 13,
});
const ollama = provider({
  id: "ollama",
  name: "Ollama",
  configured: true,
  accepts_api_key: false,
  local: true,
  model_count: 0,
});
const lmstudioRunning = provider({
  id: "lmstudio",
  name: "LM Studio",
  configured: true,
  accepts_api_key: false,
  local: true,
  model_count: 3,
});
const deepseek = provider({ id: "deepseek", name: "DeepSeek", model_count: 2 });

describe("isIdleLocal", () => {
  it("flags a keyless local endpoint the seat has no models from", () => {
    // `configured` is true here — the seat's check is credential presence, and
    // a keyless provider always passes it. Only model_count knows.
    expect(ollama.configured).toBe(true);
    expect(isIdleLocal(ollama)).toBe(true);
  });

  it("leaves a local endpoint alone once it is actually serving models", () => {
    expect(isIdleLocal(lmstudioRunning)).toBe(false);
  });

  it("never applies to a remote provider, however few models it lists", () => {
    expect(isIdleLocal(provider({ model_count: 0 }))).toBe(false);
  });
});

describe("groupProviders", () => {
  it("keeps a dead local endpoint out of Ready", () => {
    const g = groupProviders([anthropic, ollama, deepseek], "");
    expect(g.ready.map((p) => p.id)).toEqual(["anthropic"]);
    expect(g.idle.map((p) => p.id)).toEqual(["ollama"]);
    expect(g.available.map((p) => p.id)).toEqual(["deepseek"]);
  });

  it("counts a running local endpoint as ready", () => {
    const g = groupProviders([lmstudioRunning], "");
    expect(g.ready.map((p) => p.id)).toEqual(["lmstudio"]);
    expect(g.idle).toEqual([]);
  });

  it("reports totals and matches apart, so a filtered count can say so", () => {
    const all = [anthropic, ollama, deepseek];
    expect(groupProviders(all, "").filtering).toBe(false);
    const g = groupProviders(all, "deep");
    expect(g.filtering).toBe(true);
    expect(g.matched).toBe(1);
    expect(g.total).toBe(3);
  });

  it("treats whitespace as no filter rather than as a term nothing matches", () => {
    const g = groupProviders([anthropic, deepseek], "   ");
    expect(g.filtering).toBe(false);
    expect(g.matched).toBe(2);
  });

  it("matches on the provider id as well as its display name", () => {
    // "google-vertex" is reached by id; its name is "Google Vertex AI".
    const vertex = provider({ id: "google-vertex", name: "Google Vertex AI" });
    expect(matchesFilter(vertex, "vertex")).toBe(true);
    expect(groupProviders([vertex], "GOOGLE-VER").matched).toBe(1);
  });

  it("returns a zero-match group rather than falling back to everything", () => {
    const g = groupProviders([anthropic, deepseek], "zzz");
    expect(g.matched).toBe(0);
    expect(g.ready).toEqual([]);
    expect(g.available).toEqual([]);
    expect(g.filtering).toBe(true);
  });
});

describe("showAvailable", () => {
  it("reveals the folded pile whenever a filter is active", () => {
    // This is the dead end: one match, folded away, under a Ready group that
    // was rendering "No provider is configured yet".
    const g = groupProviders([anthropic, deepseek], "deepseek");
    expect(g.available).toHaveLength(1);
    expect(showAvailable(g, false)).toBe(true);
  });

  it("respects the fold when nothing is being filtered", () => {
    const g = groupProviders([anthropic, deepseek], "");
    expect(showAvailable(g, false)).toBe(false);
    expect(showAvailable(g, true)).toBe(true);
  });
});
