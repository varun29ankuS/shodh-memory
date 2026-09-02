import { describe, expect, it } from "vitest";

import { computeEgress } from "./egress";
import type { McpServerInfo, SeatModelInfo } from "@/lib/seat/types";

const localModel = { billing: "none", provider: "ollama" } as unknown as SeatModelInfo;
const hostedModel = { billing: "api", provider: "anthropic" } as unknown as SeatModelInfo;

const server = (over: Partial<McpServerInfo>): McpServerInfo =>
  ({ name: "s", status: "ready", transport: "stdio", command: "node x.js" , ...over }) as unknown as McpServerInfo;

/**
 * These pin the claim, not the wording. Each corresponds to a way the badge
 * could read "Local" while something is leaving the machine — which is the
 * only failure of this component that matters.
 */
describe("computeEgress", () => {
  it("is local when the model is local and every server is a local process", () => {
    const e = computeEgress(localModel, [server({ transport: "stdio" })]);
    expect(e?.local).toBe(true);
  });

  it("is NOT local when a remote connector is live, even with a local model", () => {
    const e = computeEgress(localModel, [server({ name: "gmail", transport: "http" })]);
    expect(e?.local).toBe(false);
    expect(e?.title).toContain("gmail");
  });

  it("counts sse connectors as remote", () => {
    expect(computeEgress(localModel, [server({ transport: "sse" })])?.local).toBe(false);
  });

  it("ignores connectors that never came up", () => {
    const e = computeEgress(localModel, [server({ transport: "http", status: "failed" })]);
    expect(e?.local).toBe(true);
  });

  it("counts the model and connectors together", () => {
    const e = computeEgress(hostedModel, [
      server({ name: "gmail", transport: "http" }),
      server({ name: "fs", transport: "stdio" }),
    ]);
    expect(e?.label).toBe("2 exits");
  });

  it("renders nothing for an unresolved model", () => {
    expect(computeEgress(null, [])).toBeNull();
  });
});
