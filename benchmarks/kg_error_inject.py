#!/usr/bin/env python3
"""KG error-detection fixture builder (FB15k-237 + constrained noise injection).

Standard protocol from the KG error-detection literature (CKRL, KGTtm, CAGED):
take a clean knowledge graph, corrupt a fixed fraction of triples, then ask the
detector to rank all triples by trustworthiness — injected errors should sink.
We use the CONSTRAINED corruption mode ("harder" noise, Xie et al. 2018): the
replacement entity must already appear in the SAME slot of the SAME relation
somewhere in the graph, so a corrupted triple is type-plausible and cannot be
dismissed by relation-slot statistics alone.

Source data is pinned to the KG-BERT repository (yao8839836/kg-bert) at commit
62b76ed5652bf3da5a355cbe9c1109f58290756f, which carries the canonical
FB15k-237 split plus the entity2text mapping we need for the embedding signal.

Deterministic: fixed RNG seed (--seed, default 29), stable iteration order.
Output (in --out, default tests/recall/kg_error/):
  triples.tsv       head_mid <TAB> relation <TAB> tail_mid <TAB> label
                    label 1 = original triple, 0 = injected corruption
  entity_names.tsv  mid <TAB> human-readable name
  meta.json         counts, ratio, seed, source pin
"""

import argparse
import json
import pathlib
import random
import urllib.request

PIN = "62b76ed5652bf3da5a355cbe9c1109f58290756f"
BASE = f"https://raw.githubusercontent.com/yao8839836/kg-bert/{PIN}/data/FB15k-237"


def fetch(name: str, cache: pathlib.Path) -> str:
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / name
    if not path.exists():
        print(f"downloading {name} @ {PIN[:8]} ...")
        urllib.request.urlretrieve(f"{BASE}/{name}", path)
    return path.read_text(encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ratio", type=float, default=0.10,
                    help="fraction of triples to corrupt (literature: 0.05/0.10/0.20)")
    ap.add_argument("--seed", type=int, default=29)
    ap.add_argument("--limit", type=int, default=None,
                    help="cap the number of CLEAN triples first (cheap pilot)")
    ap.add_argument("--out", default="tests/recall/kg_error")
    ap.add_argument("--cache", default=".kg-error-cache")
    args = ap.parse_args()

    cache = pathlib.Path(args.cache)
    triples = []
    for line in fetch("train.tsv", cache).splitlines():
        parts = line.strip().split("\t")
        if len(parts) == 3:
            triples.append(tuple(parts))
    # Deterministic order regardless of source ordering quirks.
    triples.sort()

    rng = random.Random(args.seed)
    if args.limit and args.limit < len(triples):
        triples = rng.sample(triples, args.limit)
        triples.sort()

    # Slot pools per relation for constrained corruption.
    heads_of, tails_of = {}, {}
    for h, r, t in triples:
        heads_of.setdefault(r, []).append(h)
        tails_of.setdefault(r, []).append(t)
    for pool in (heads_of, tails_of):
        for r in pool:
            pool[r] = sorted(set(pool[r]))

    existing = set(triples)
    n_corrupt = int(len(triples) * args.ratio)
    victims = set(rng.sample(range(len(triples)), n_corrupt))

    out_rows, injected = [], 0
    for i, (h, r, t) in enumerate(triples):
        if i not in victims:
            out_rows.append((h, r, t, 1))
            continue
        corrupted = None
        for _ in range(50):
            if rng.random() < 0.5:
                cand = (rng.choice(heads_of[r]), r, t)
            else:
                cand = (h, r, rng.choice(tails_of[r]))
            if cand not in existing and cand[0] != cand[2]:
                corrupted = cand
                break
        if corrupted is None:  # degenerate slot pool; keep the original as clean
            out_rows.append((h, r, t, 1))
            continue
        existing.add(corrupted)
        out_rows.append((*corrupted, 0))
        injected += 1

    names = {}
    for line in fetch("entity2text.txt", cache).splitlines():
        parts = line.strip().split("\t")
        if len(parts) >= 2:
            names[parts[0]] = parts[1]

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "triples.tsv", "w", encoding="utf-8", newline="\n") as f:
        for h, r, t, label in out_rows:
            f.write(f"{h}\t{r}\t{t}\t{label}\n")
    used = {e for h, _, t, _ in out_rows for e in (h, t)}
    with open(out / "entity_names.tsv", "w", encoding="utf-8", newline="\n") as f:
        for mid in sorted(used):
            f.write(f"{mid}\t{names.get(mid, mid)}\n")
    meta = {
        "dataset": "FB15k-237 train split",
        "source_pin": PIN,
        "triples": len(out_rows),
        "injected_errors": injected,
        "ratio_requested": args.ratio,
        "seed": args.seed,
        "corruption": "constrained (same relation slot), no duplicates, no self-loops",
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
