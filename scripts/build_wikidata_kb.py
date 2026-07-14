#!/usr/bin/env python3
"""Build the CC0 domain-KB asset for shodh's retrieval-based entity linking (ER Task 3.2).

The KB is a STATIC asset: each entity carries a precomputed MiniLM embedding of
`label + description`, so the Rust runtime (`src/kb.rs`) needs no embedder and stays
deterministic + on-device. Linking is then a type-blocked cosine nearest-neighbour of a
mention's embedding against the KB, plus an exact alias fast-path.

## Source (CC0)

Wikidata is CC0 (public domain) — the sovereignty-friendly choice. Produce the *input*
JSONL (one raw entity per line, no embeddings) by ONE of:

  1. Wikidata5M (Wang et al.) — a pruned ~4.6M-entity subset with aliases + descriptions.
     Convert `wikidata5m_entity.txt` (QID<TAB>alias1<TAB>alias2...) + `wikidata5m_text.txt`
     (QID<TAB>description) into the input format below.
  2. wdumper / the Wikidata Query Service — filter `instance-of (P31) ∈ {person, organization,
     location, event, product}`, keep label + also-known-as + description + P31, optionally
     P749 (parent-org) for corporate hierarchy (the real `Google→Alphabet` link).
  3. Domain / mission KB — a curated set you own (more sovereign, higher precision).

## Input format (JSONL, one per line; NO embedding)

    {"id": "Q20800404", "label": "Alphabet Inc.",
     "aliases": ["Google", "Alphabet", "GOOGL"],
     "description": "American multinational technology conglomerate",
     "entity_type": "organization"}

## Output format (JSONL, what shodh loads via SHODH_KB_PATH)

    {... same fields ..., "embedding": [<384 floats>]}

## IMPORTANT — embedder parity

The embedding model here MUST match shodh's runtime embedder (all-MiniLM-L6-v2, 384-d),
or cosine scores between a mention embedding and a KB embedding are meaningless. Do not
change the model without re-checking parity against shodh's MiniLM.

Usage:
    pip install sentence-transformers
    python build_wikidata_kb.py --in wikidata_raw.jsonl --out domain_kb.jsonl
    # then:  SHODH_KB_PATH=domain_kb.jsonl SHODH_KB_LINKING=1 <run shodh>
"""

import argparse
import json
import sys

MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # must match shodh's runtime embedder


def coarse_type(raw: str) -> str:
    """Normalise assorted type labels to shodh's coarse blocking types."""
    r = (raw or "").strip().lower()
    if r in ("person", "human", "q5"):
        return "person"
    if r in ("organization", "organisation", "company", "business", "agency"):
        return "organization"
    if r in ("location", "place", "city", "country", "geographic"):
        return "location"
    if r in ("event",):
        return "event"
    if r in ("product", "work", "software"):
        return "product"
    return r  # leave as-is; empty type → searched across all types at link time


def main() -> int:
    ap = argparse.ArgumentParser(description="Embed a raw KB JSONL into shodh's KB asset.")
    ap.add_argument("--in", dest="inp", required=True, help="raw entities JSONL (no embeddings)")
    ap.add_argument("--out", dest="out", required=True, help="output KB JSONL (with embeddings)")
    ap.add_argument("--batch", type=int, default=256)
    args = ap.parse_args()

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        sys.exit("pip install sentence-transformers")

    model = SentenceTransformer(MODEL)

    rows = []
    with open(args.inp, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            label = (e.get("label") or "").strip()
            if not label:
                continue
            e["entity_type"] = coarse_type(e.get("entity_type", ""))
            e["aliases"] = [a for a in e.get("aliases", []) if a and a.strip()]
            e["description"] = (e.get("description") or "").strip()
            rows.append(e)

    # Retrieval text = label + description (the same text a mention is compared against).
    texts = [f"{e['label']} {e['description']}".strip() for e in rows]
    written = 0
    with open(args.out, "w", encoding="utf-8") as out:
        for i in range(0, len(rows), args.batch):
            chunk = rows[i : i + args.batch]
            embs = model.encode(
                texts[i : i + args.batch],
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            for e, emb in zip(chunk, embs):
                e["embedding"] = [round(float(x), 6) for x in emb]
                out.write(json.dumps(e, ensure_ascii=False) + "\n")
                written += 1

    print(f"wrote {written} KB entities → {args.out}  (model {MODEL})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
