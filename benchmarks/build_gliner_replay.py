#!/usr/bin/env python3
"""Build a SHODH_NER_REPLAY map {text -> [entities]} from a harness corpus using
GLiNER, so the REAL shodh recall pipeline can run with GLiNER NER instead of the
bundled TinyBERT — no Rust port, no parallel code (ner.rs replay hook does the rest).

Usage: python build_gliner_replay.py "<corpora glob>" <out.json>
  e.g.  python build_gliner_replay.py "tests/recall/longmemeval/corpora/*.jsonl" gliner_replay.json

Replay JSON shape (from ner.rs): {"<text>": [{"text","type"(PER|ORG|LOC|MISC),"start","end","conf"}]}
"""
import json, sys, glob, time

# GLiNER schema -> shodh's NER types (PER/ORG/LOC/MISC).
LABELS = ["person", "organization", "company", "government agency",
          "location", "product", "event", "technology"]
TMAP = {"person": "PER", "organization": "ORG", "company": "ORG",
        "government agency": "ORG", "location": "LOC"}
def to_type(label): return TMAP.get(label, "MISC")


def main():
    corpus_glob, out = sys.argv[1], sys.argv[2]
    from gliner import GLiNER
    model = GLiNER.from_pretrained("urchade/gliner_multi-v2.1")

    texts, seen = [], set()
    for f in glob.glob(corpus_glob):
        for line in open(f, encoding="utf-8"):
            c = (json.loads(line).get("content") or "").strip()
            if c and c not in seen:
                seen.add(c); texts.append(c)
    print(f"{len(texts)} unique corpus texts to extract", flush=True)

    replay, t0, nents = {}, time.time(), 0
    for i, text in enumerate(texts):
        ents = model.predict_entities(text, LABELS, threshold=0.5)
        replay[text] = [{"text": e["text"], "type": to_type(e["label"]),
                         "start": e["start"], "end": e["end"],
                         "conf": round(float(e["score"]), 3)} for e in ents]
        nents += len(ents)
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(texts)} texts, {nents} entities, {time.time()-t0:.0f}s", flush=True)
    json.dump(replay, open(out, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"GLiNER replay map: {len(replay)} texts, {nents} entities -> {out} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
