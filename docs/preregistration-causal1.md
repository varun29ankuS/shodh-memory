# causal1 pre-registration (written BEFORE any model call)

Change under test: backend commit 0f385c0b (causal-language lineage handshake)
on branch feat/causal-chain-lineage; seat + scorer + case set = origin/main
689a89a7, byte-identical (zero diffs under seat/eval/).

Comparator: mech1 arm B (bundle, claude-haiku-4-5, 6 repeats) = 111/126,
forward-trace 1/6, propulsion-chain 2/6.

Protocol: arm B only (--mech on --guidance off), claude-haiku-4-5 via
anthropic, 6 repeats, fresh users causal1-b1..b6, full frozen 21-case set,
backend = freshly built exe from 0f385c0b (BACKEND_EXE explicit), rescore
with the untouched main scorer.

PASS bar (all three required):
1. forward-trace >= 4/6
2. propulsion-chain >= 4/6
3. No regression: no non-chain case drops >= 2 passes vs its mech1 B count,
   and overall B pass-rate >= 85% (mech1 was 88.1% +/- 4.0pp).

Gate checks pre-registered on the seeded census (before the model run):
- language-tier edges (conf >= 0.5) <= 10 on the 71-memory corpus
- zero conf >= 0.5 edges between two haystack memories (m19+)
- m1->m2 present as Caused at 0.70; m0->m1 upgraded to Caused >= 0.70
- recall probe for the forward-trace question returns m2 among results with
  the m1->m2 edge in the lineage payload

NO scoring changes, no case changes, no LLM judge. If the bar is missed the
result is reported as-is.
