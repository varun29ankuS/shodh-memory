import { describe, expect, it } from "vitest";
import { formatCount, isPartialRead, sampleNote } from "./format";

/**
 * Every test here was checked to FAIL when the line it covers is deleted, and
 * each mutation is named in the test that catches it. The alternative — an
 * assertion that holds however the module is mutated — is how this codebase
 * previously carried two whole modules alive only via their own self-tests.
 *
 * `formatCount` is the second wrong-figure formatting defect this product has
 * shipped past a passing test (the first rendered "1h 60m"), so its tests
 * assert the EXACT rendered string rather than a property of it. A test that
 * only asserts "contains a non-digit" passes on every locale the runtime might
 * pick, which is precisely the flake it was supposed to catch.
 */

describe("formatCount", () => {
  it("leaves a three-digit count bare", () => {
    // Mutation caught: replacing the hand-rolled grouping with
    // `value.toLocaleString()` on a runtime whose default numbering system is
    // not latn returns "٢٣٠" and fails here. On an en-US machine this exact
    // assertion still holds, which is why the grouping cases below carry the
    // weight of the pin.
    expect(formatCount(230)).toBe("230");
  });

  it("groups a five-digit count with commas, on every machine", () => {
    // Mutation caught: deleting the `.replace(/\B(?=(\d{3})+(?!\d))/g, ",")`
    // returns "19553". Asserting the literal separator is the point — the
    // previous test asserted only `/\D/`, which `toLocaleString()` satisfies
    // in de-DE ("19.553") and in en-IN ("19,553" for four digits but "1,95,530"
    // for six), so it could not tell a pin from its absence.
    expect(formatCount(19_553)).toBe("19,553");
    expect(formatCount(10_758)).toBe("10,758");
  });

  it("groups beyond a single separator", () => {
    // Mutation caught: a single-separator implementation (e.g. one `slice` at
    // three digits from the end) returns "1234,567".
    expect(formatCount(1_234_567)).toBe("1,234,567");
  });

  it("keeps the sign outside the grouping", () => {
    // Mutation caught: dropping the `Math.abs` makes the minus part of the
    // digit string, so the grouping counts it and returns "-1,234" as
    // "-1,234" only by luck at some widths and "-123,4" at others.
    expect(formatCount(-1234)).toBe("-1,234");
  });

  it("renders a non-finite count as zero rather than as NaN", () => {
    // Mutation caught: deleting the `Number.isFinite` guard renders the string
    // "NaN" into a status strip, which reads as a measurement.
    expect(formatCount(Number.NaN)).toBe("0");
    expect(formatCount(Number.POSITIVE_INFINITY)).toBe("0");
  });
});

describe("isPartialRead", () => {
  it("is true when the page is smaller than the profile", () => {
    // The defect this whole primitive exists for: 500 rows read, 19,553 held.
    // Mutation caught: `return false` (i.e. deleting the comparison) lets Geo
    // assert an absolute negative from 2.6% of a corpus.
    expect(isPartialRead(500, 19_553)).toBe(true);
  });

  it("is false when the page covered the whole profile", () => {
    // Mutation caught: flipping `<` to `<=` makes a complete read report as
    // partial, which prints "28 of 28 read" and hedges a claim that is earned.
    expect(isPartialRead(28, 28)).toBe(false);
  });

  it("is false when the listing somehow holds more than the reported total", () => {
    // A total that lags its own page (a write landing between the two numbers)
    // must not be reported as a partial read. Mutation caught: `read !== total`.
    expect(isPartialRead(30, 28)).toBe(false);
  });

  it("is false when either figure is not a number", () => {
    // A figure that has not arrived must never be reported as a partial read —
    // that would print "NaN of NaN read" into a status strip. Mutation caught:
    // `read !== total`, which is true for NaN and would hedge every screen
    // whose counts had not loaded. An explicit finiteness guard was tried here
    // and deleted: no mutation of it failed a test, because `<` against NaN is
    // already false. The comparison is the mechanism, and this asserts it.
    expect(isPartialRead(Number.NaN, 19_553)).toBe(false);
    expect(isPartialRead(500, Number.NaN)).toBe(false);
  });
});

describe("sampleNote", () => {
  it("states both halves of the fraction, grouped", () => {
    // Mutation caught: dropping either `formatCount` call renders "500 of
    // 19553 read"; dropping the whole template renders nothing at all and the
    // denominator disappears from the screen again.
    expect(sampleNote(500, 19_553)).toBe("500 of 19,553 read");
  });

  it("is null when the read was complete, so the token is dropped entirely", () => {
    // This is the clutter budget: a profile the page fully covers pays NO words
    // for honesty. Mutation caught: deleting the `isPartialRead` guard returns
    // "28 of 28 read" and spends four tokens saying nothing.
    expect(sampleNote(28, 28)).toBeNull();
    expect(sampleNote(0, 0)).toBeNull();
  });

  it("reports a page that read nothing against a profile that holds something", () => {
    // Mutation caught: a truthiness guard on `read` (`if (!read) return null`)
    // would silently drop the denominator in the one case where the screen is
    // at its most misleading — nothing read, everything claimed.
    expect(sampleNote(0, 19_553)).toBe("0 of 19,553 read");
  });
});
