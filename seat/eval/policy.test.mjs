import assert from "node:assert/strict";
import test from "node:test";

import { ToolPolicy, PolicyError, loadPolicy } from "../dist/policy.js";

/**
 * These pin the properties the policy is *for*, not its implementation.
 *
 * Each one corresponds to a way the constraint could look like it works and
 * not: a default that fails open, a wildcard that reaches further than the
 * server it names, an unreadable file degrading to "unrestricted", and rule
 * order that quietly stops mattering.
 */

test("an unmatched tool is withheld by default", () => {
  const p = new ToolPolicy({ rules: [{ match: "mcp__a__read", effect: "allow" }] });
  assert.equal(p.decide("mcp__a__read").allowed, true);
  assert.equal(p.decide("mcp__a__delete").allowed, false, "default must not fail open");
  assert.equal(p.decide("mcp__a__delete").by, "default");
});

test("first matching rule wins, so file order is the policy", () => {
  const p = new ToolPolicy({
    default: "allow",
    rules: [
      { match: "mcp__fs__*", effect: "withhold", reason: "no filesystem" },
      { match: "mcp__fs__read", effect: "allow", reason: "unreachable — shadowed above" },
    ],
  });
  const d = p.decide("mcp__fs__read");
  assert.equal(d.allowed, false);
  assert.equal(d.by, "mcp__fs__*");
  assert.equal(d.reason, "no filesystem");
});

test("a wildcard does not leak past the server it names", () => {
  const p = new ToolPolicy({ rules: [{ match: "mcp__shodh__*", effect: "allow" }] });
  assert.equal(p.decide("mcp__shodh__recall").allowed, true);
  assert.equal(p.decide("mcp__shodh__deep__nested").allowed, true, "* spans __");
  assert.equal(p.decide("mcp__shodhx__recall").allowed, false, "must not match a longer server name");
  assert.equal(p.decide("evil__mcp__shodh__recall").allowed, false, "must be anchored at the start");
});

test("regex metacharacters in a pattern are literal", () => {
  const p = new ToolPolicy({ rules: [{ match: "mcp__a__get.thing", effect: "allow" }] });
  assert.equal(p.decide("mcp__a__get.thing").allowed, true);
  assert.equal(p.decide("mcp__a__getXthing").allowed, false, "'.' must not act as a wildcard");
});

test("apply splits the list and returns the withheld half as evidence", () => {
  const p = new ToolPolicy({
    rules: [{ match: "mcp__a__read", effect: "allow" }],
  });
  const { allowed, withheld } = p.apply([
    { name: "mcp__a__write" },
    { name: "mcp__a__read" },
    { name: "mcp__a__delete" },
  ]);
  assert.deepEqual(allowed.map((t) => t.name), ["mcp__a__read"]);
  assert.deepEqual(
    withheld.map((d) => d.tool),
    ["mcp__a__delete", "mcp__a__write"],
    "withheld is sorted so the same decision logs identically twice",
  );
});

test("a named-but-unreadable policy file is fatal, never 'unrestricted'", () => {
  assert.throws(
    () => loadPolicy({ SEAT_POLICY: "./definitely-not-here.json" }),
    PolicyError,
    "failing open on a missing policy is the dangerous failure",
  );
});

test("no SEAT_POLICY means unrestricted, so upgrading does not break deployments", () => {
  const p = loadPolicy({});
  assert.equal(p.decide("mcp__anything__at_all").allowed, true);
});

test("an invalid effect is rejected at load, not at call time", () => {
  assert.throws(
    () => new ToolPolicy({ rules: [{ match: "x", effect: "maybe" }] }),
    PolicyError,
  );
});
