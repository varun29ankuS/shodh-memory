#!/usr/bin/env python3
"""Task 7 — before/after validation harness for the GLiNER bi-edge typer (D12 gate).

Answers the question the branch exists to answer: does swapping the 4-class
bert-tiny CoNLL tagger (+ `classify_misc_entity` regex heuristic) for the
gliner-bi-edge-v2.0 schema-driven typer actually produce better types, and
does the finer typing help or hurt canonicalization (the D12 gate)?

Two real, independently-sourced corpora, honestly labeled (not conflated):

  BEFORE — `demos/gdelt-bridge/cooc_graph.json`, an export of the real
  production engine graph ("bridge_test" user) that was typed by bert-tiny +
  the MISC regex heuristic before this branch's typer swap. 669 entity nodes /
  1820 mentions over a ~184-episode GDELT Baltimore-bridge corpus.

  AFTER — a fresh run of the shipped `models/gliner-bi-edge` ONNX bi-encoder
  over `demos/gdelt-bridge/passages_100.jsonl` (100 passages), replicating
  `src/embeddings/gliner.rs::GlinerTyper` span-for-span in Python (same word
  splitter regex, same 7 ONNX inputs, same greedy non-overlap decode, same
  threshold=0.3/max_width=12). The total span count this harness produces is
  cross-checked against `models/gliner-bi-edge/parity.json` and the Rust
  integration test's own count (1269) as a correctness gate on the harness
  itself — if that number drifts, the Python replica has diverged from the
  Rust decode and every downstream number here is suspect.

  These two corpora OVERLAP (the 184-episode graph was built from a superset
  that includes most of the 100-passage sample — confirmed by substring
  matching, see task-7-report.md) but are NOT byte-identical, so dimension 3
  (canonicalization) reports both a same-corpus "official" bert-tiny number
  AND a ratio-normalized comparison against the GLiNER 100-passage run. No
  number here is fabricated across a corpus the harness didn't actually run
  on — see task-7-report.md for exactly which corpus backs which number.

Usage:
    python scripts/validate_typer.py

Writes `demos/gdelt-bridge/validate_typer_out.json` with every raw number
this script prints, so the before/after tables in
`shodh-vault/04-Research/Entity-Type-Taxonomy.md` and
`.superpowers/sdd/task-7-pr-numbers.md` can be cited back to a reproducible
artifact.
"""
from __future__ import annotations

import json
import math
import os
import re
import time
from collections import Counter, defaultdict

import numpy as np
import onnxruntime as ort
import spacy
from tokenizers import Tokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

MODEL_DIR = os.path.join(ROOT, "models", "gliner-bi-edge")
SCHEMA_PATH = os.path.join(ROOT, "src", "entity_type", "entity-type-schema.json")
PASSAGES_PATH = os.path.join(ROOT, "demos", "gdelt-bridge", "passages_100.jsonl")
COOC_GRAPH_PATH = os.path.join(ROOT, "demos", "gdelt-bridge", "cooc_graph.json")
PARITY_PATH = os.path.join(MODEL_DIR, "parity.json")
OUT_JSON = os.path.join(ROOT, "demos", "gdelt-bridge", "validate_typer_out.json")

THRESHOLD = 0.3
MAX_WIDTH = 12

KEY_ENTITIES = [
    "dali",
    "baltimore",
    "francis scott key bridge",
    "patapsco river",
    "port of baltimore",
    "ntsb",
    "cargo ship",
]

WORD_RE = re.compile(r"\w+(?:[-_]\w+)*|\S")


def whitespace_split(text: str):
    """Mirror gliner's WhitespaceTokenSplitter (`\\w+(?:[-_]\\w+)*|\\S`)."""
    return [(m.group(0), m.start(), m.end()) for m in WORD_RE.finditer(text)]


def probe_matches(probe: str, surface: str) -> bool:
    """Word-boundary containment, probe-phrase-in-surface direction only: the
    key-entity table asks "what type did the model give THIS specific named
    thing" (e.g. "cargo ship"), so a bare "ship" span must NOT count as a
    "cargo ship" hit (that would silently blend two different questions).
    Plain substring containment (`"a" in "dali"`) would additionally
    spuriously match single-character spans against any probe that happens to
    contain that character — word boundaries close that hole too.
    """
    p, s = probe.lower(), surface.lower().strip()
    if not s:
        return False
    return re.search(r"(?<!\w)" + re.escape(p) + r"(?!\w)", s) is not None


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


class GlinerBiEdge:
    """Python replica of `src/embeddings/gliner.rs::GlinerTyper`.

    Same 7 ONNX inputs, same word splitter, same words_mask construction, same
    greedy non-overlapping decode, same threshold/max_width defaults. This is
    NOT a from-scratch reimplementation of gliner — it is a byte-for-byte port
    of the Rust production code path so the "after" numbers in this report are
    what the shipped typer actually produces, not an approximation of it.
    """

    def __init__(self, model_dir: str = MODEL_DIR, threshold: float = THRESHOLD):
        self.threshold = threshold
        self.session = ort.InferenceSession(
            os.path.join(model_dir, "model.onnx"), providers=["CPUExecutionProvider"]
        )
        self.tokenizer = Tokenizer.from_file(os.path.join(model_dir, "tokenizer.json"))

        schema = json.load(open(SCHEMA_PATH, encoding="utf-8"))
        self.fine_labels = [f["label"] for f in schema["fine"]]
        self.coarse_of = {f["label"]: f["coarse"] for f in schema["fine"]}
        self.num_labels = len(self.fine_labels)
        assert self.num_labels == 141, f"expected 141 fine labels, got {self.num_labels}"

        raw = open(os.path.join(model_dir, "label_embeddings.bin"), "rb").read()
        arr = np.frombuffer(raw, dtype="<f4")
        self.hidden = arr.size // self.num_labels
        assert self.hidden * self.num_labels == arr.size, "label_embeddings.bin size mismatch"
        self.label_embeds = arr.reshape(self.num_labels, self.hidden).astype(np.float32)

    def extract(self, text: str):
        words = whitespace_split(text)
        if not words:
            return []
        num_words = len(words)
        word_texts = [w[0] for w in words]

        enc = self.tokenizer.encode(word_texts, is_pretokenized=True, add_special_tokens=True)
        ids = enc.ids
        word_ids = enc.word_ids
        seq = len(ids)

        input_ids = np.array([ids], dtype=np.int64)
        attention_mask = np.ones((1, seq), dtype=np.int64)

        # words_mask: 1-based word index at the FIRST subword of each word, 0
        # for continuation subwords and special tokens.
        words_mask = np.zeros((1, seq), dtype=np.int64)
        prev = None
        seen = 0
        for i, wid in enumerate(word_ids):
            if wid is None:
                prev = wid
                continue
            if wid != prev:
                seen += 1
                words_mask[0, i] = seen
            prev = wid

        text_lengths = np.array([[num_words]], dtype=np.int64)

        span_idx = []
        span_mask = []
        for start in range(num_words):
            for offset in range(MAX_WIDTH):
                end = start + offset
                span_idx.append([start, end])
                span_mask.append(end < num_words)
        span_idx_arr = np.array([span_idx], dtype=np.int64)
        span_mask_arr = np.array([span_mask], dtype=bool)

        outputs = self.session.run(
            None,
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "words_mask": words_mask,
                "text_lengths": text_lengths,
                "span_idx": span_idx_arr,
                "span_mask": span_mask_arr,
                "labels_embeds": self.label_embeds,
            },
        )
        logits = outputs[0]  # [1, L, K, C]
        _, l_dim, k_dim, c_dim = logits.shape
        probs = sigmoid(logits[0])  # [L, K, C]

        cands = []
        for s in range(l_dim):
            for k in range(k_dim):
                if s + k + 1 > num_words:
                    continue
                row = probs[s, k]
                above = np.nonzero(row > self.threshold)[0]
                for c in above:
                    cands.append((float(row[c]), s, s + k, int(c)))

        # Greedy non-overlapping selection (flat NER, single-label), matching
        # gliner's SpanDecoder / the Rust `decode` function exactly.
        cands.sort(key=lambda x: -x[0])
        selected = []
        for score, start, end, cls in cands:
            overlaps = False
            for _, ss, se, _ in selected:
                if (ss == start and se == end) or not (start > se or ss > end):
                    overlaps = True
                    break
            if not overlaps:
                selected.append((score, start, end, cls))
        selected.sort(key=lambda x: x[1])

        spans = []
        for score, start, end, cls in selected:
            if start >= num_words or end >= num_words:
                continue
            fine = self.fine_labels[cls] if cls < len(self.fine_labels) else None
            if not fine:
                continue
            coarse = self.coarse_of.get(fine, "other")
            start_byte = words[start][1]
            end_byte = words[end][2]
            surface = text[start_byte:end_byte]
            spans.append(
                dict(text=surface, fine=fine, coarse=coarse, score=score, start=start_byte, end=end_byte)
            )
        return spans


def entropy(counts: Counter) -> float:
    total = sum(counts.values())
    if total == 0:
        return 0.0
    h = 0.0
    for c in counts.values():
        if c == 0:
            continue
        p = c / total
        h -= p * math.log2(p)
    return h


# ---------------------------------------------------------------------------
# Dimension 1/2: run GLiNER over passages_100.jsonl, collect type distribution
# and key-entity spans.
# ---------------------------------------------------------------------------

def run_gliner_over_passages():
    typer = GlinerBiEdge()
    passages = [json.loads(l) for l in open(PASSAGES_PATH, encoding="utf-8") if l.strip()]

    t0 = time.time()
    all_spans = []
    per_passage_spans = []
    for doc in passages:
        text = doc.get("text", "")
        spans = typer.extract(text)
        per_passage_spans.append(spans)
        all_spans.extend(spans)
    elapsed = time.time() - t0

    fine_counts = Counter(s["fine"] for s in all_spans)
    coarse_counts = Counter(s["coarse"] for s in all_spans)

    key_hits = {}
    for probe in KEY_ENTITIES:
        matches = [s for s in all_spans if probe_matches(probe, s["text"])]
        fine_for_probe = Counter(s["fine"] for s in matches)
        key_hits[probe] = dict(
            count=len(matches),
            fine_dist=fine_for_probe.most_common(5),
            example_surfaces=sorted({s["text"] for s in matches})[:8],
        )

    return dict(
        passages=len(passages),
        total_spans=len(all_spans),
        elapsed_s=elapsed,
        ms_per_passage=1000 * elapsed / max(len(passages), 1),
        fine_types_used=len(fine_counts),
        coarse_types_used=len(coarse_counts),
        fine_dist=fine_counts.most_common(30),
        coarse_dist=coarse_counts.most_common(20),
        fine_entropy=entropy(fine_counts),
        coarse_entropy=entropy(coarse_counts),
        top_coarse_frac=coarse_counts.most_common(1)[0][1] / len(all_spans) if all_spans else 0.0,
        top_fine_frac=fine_counts.most_common(1)[0][1] / len(all_spans) if all_spans else 0.0,
        key_entities=key_hits,
        all_spans=all_spans,
    ), typer


# ---------------------------------------------------------------------------
# Dimension 1/2 (before): read the shipped bert-tiny production graph.
# ---------------------------------------------------------------------------

def load_bert_tiny_baseline():
    G = json.load(open(COOC_GRAPH_PATH, encoding="utf-8"))
    ents = [n for n in G["nodes"] if n.get("type") == "entity"]

    primary_unique = Counter()
    primary_mentions = Counter()
    total_mentions = 0
    for n in ents:
        labs = (n.get("attributes") or {}).get("labels") or []
        mc = int((n.get("attributes") or {}).get("mention_count", 1))
        total_mentions += mc
        if labs:
            primary_unique[labs[0]] += 1
            primary_mentions[labs[0]] += mc

    key_hits = {}
    for probe in KEY_ENTITIES:
        matches = [n for n in ents if probe_matches(probe, n["label"])]
        fine_for_probe = Counter(
            ((n.get("attributes") or {}).get("labels") or ["?"])[0] for n in matches
        )
        key_hits[probe] = dict(
            count=len(matches),
            total_mentions=sum(int((n.get("attributes") or {}).get("mention_count", 1)) for n in matches),
            fine_dist=fine_for_probe.most_common(5),
            example_surfaces=sorted({n["label"] for n in matches})[:8],
        )

    return dict(
        graph=G,
        entity_nodes=ents,
        unique_entities=len(ents),
        total_mentions=total_mentions,
        primary_unique_dist=primary_unique.most_common(20),
        primary_mentions_dist=primary_mentions.most_common(20),
        unique_entropy=entropy(primary_unique),
        mentions_entropy=entropy(primary_mentions),
        top_bucket_frac_unique=primary_unique.most_common(1)[0][1] / len(ents) if ents else 0.0,
        top_bucket_frac_mentions=primary_mentions.most_common(1)[0][1] / total_mentions if total_mentions else 0.0,
        n_buckets=len(primary_unique),
        key_entities=key_hits,
    )


# ---------------------------------------------------------------------------
# Dimension 3: type-blocked canonicalization (D12 gate). Mirrors
# demos/gdelt-bridge/entity_resolve_v2.py's head-blocking + modifier-compat +
# rare-shared-modifier merge rules (the causal-fingerprint rule is included
# for the bert-tiny run, where real typed edges exist; it naturally
# contributes 0 merges on the GLiNER run below since no relation-extraction
# graph was built over the 100-passage sample here — a typer-only harness,
# not a full re-ingest).
# ---------------------------------------------------------------------------

DET = {"the", "a", "an", "this", "that", "these", "those", "its", "their", "'s", "'s"}
JUNK_HEAD = {
    "tuesday", "monday", "morning", "night", "day", "week", "time", "hour",
    "thing", "area", "way", "number", "people", "one", "early", "today",
}
STOP = DET | {"of", "for", "in", "on", "at", "to", "and", "&"}
CAUSAL_REL = {"Triggers", "Causes", "Struck", "Damaged", "Killed", "Closed",
              "Disrupted", "Halted", "Blocked", "Suspended"}
MAX_SHARE = 4
RARE = 3

_nlp = None


def nlp():
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm")
    return _nlp


def parse_mention(label: str):
    doc = nlp()(label)
    toks = [t for t in doc if t.is_alpha or t.text.isupper()]
    if not toks:
        return None
    roots = [t for t in doc if t.head == t]
    root = roots[0] if roots else toks[-1]
    if root.pos_ in ("VERB", "AUX"):
        subj = [c for c in root.children if c.dep_ in ("nsubj", "nsubjpass", "dobj") and c.pos_ in ("NOUN", "PROPN")]
        if subj:
            root = subj[0]
    head = root.lemma_.lower()
    hpos = root.pos_
    mods = frozenset(
        t.lemma_.lower() for t in toks
        if t.lemma_.lower() != head and t.lemma_.lower() not in STOP
        and not t.is_stop and t.pos_ in ("NOUN", "PROPN", "ADJ", "NUM", "X")
    )
    return dict(head=head, mods=mods, hpos=hpos)


def canonicalize(mentions, edges=None):
    """mentions: list of dict(id, label, types:set[str], mention_count:int, proper:bool).
    edges (optional): list of dict(source, target, label) — node ids referring
    to `id` above — used only for the causal-fingerprint merge rule.
    Returns (clusters: dict[root_id -> list[id]], merge_counts: Counter, valid_ids: set).
    """
    info = {}
    for m in mentions:
        p = parse_mention(m["label"])
        if not p:
            continue
        p.update(id=m["id"], label=m["label"], types=frozenset(m["types"]),
                  mention_count=m.get("mention_count", 1), proper=m.get("proper", False))
        info[m["id"]] = p

    valid = {i for i, d in info.items() if d["head"] and d["head"] not in JUNK_HEAD and d["hpos"] in ("NOUN", "PROPN")}

    fp = defaultdict(set)
    if edges:
        for e in edges:
            if e.get("label") in CAUSAL_REL and e["source"] in valid and e["target"] in valid:
                fp[e["source"]].add((e["label"], "out", info[e["target"]]["head"]))
                fp[e["target"]].add((e["label"], "in", info[e["source"]]["head"]))
    fp_holders = defaultdict(set)
    for i in valid:
        for f in fp[i]:
            fp_holders[f].add(i)

    def causal_shared(a, b):
        common = fp[a] & fp[b]
        return any(len(fp_holders[f]) <= MAX_SHARE for f in common)

    parent = {i: i for i in valid}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    def mods_compat(a, b):
        ma, mb = info[a]["mods"], info[b]["mods"]
        return (not ma) or (not mb) or ma <= mb or mb <= ma

    mod_df = Counter(m for i in valid for m in info[i]["mods"])

    def rare_shared_mod(a, b):
        return any(m in info[b]["mods"] and mod_df[m] <= RARE for m in info[a]["mods"])

    by_head = defaultdict(list)
    for i in valid:
        by_head[info[i]["head"]].append(i)

    merges = Counter()
    for head, grp in by_head.items():
        for x in range(len(grp)):
            for y in range(x + 1, len(grp)):
                a, b = grp[x], grp[y]
                if not (info[a]["types"] & info[b]["types"]):
                    continue
                if mods_compat(a, b):
                    union(a, b)
                    merges["head+mods"] += 1
                elif rare_shared_mod(a, b):
                    union(a, b)
                    merges["head+rare-mod"] += 1
                elif causal_shared(a, b):
                    union(a, b)
                    merges["head+causal"] += 1

    clusters = defaultdict(list)
    for i in valid:
        clusters[find(i)].append(i)

    return clusters, merges, valid, info


def canonicalize_bert_tiny(baseline):
    ents = baseline["entity_nodes"]
    mentions = [
        dict(
            id=n["id"],
            label=n["label"],
            types=set((n.get("attributes") or {}).get("labels") or []),
            mention_count=int((n.get("attributes") or {}).get("mention_count", 1)),
            proper=bool((n.get("attributes") or {}).get("is_proper_noun")),
        )
        for n in ents
    ]
    clusters, merges, valid, info = canonicalize(mentions, edges=baseline["graph"]["edges"])
    return dict(
        input_mentions=len(ents),
        valid_mentions=len(valid),
        dropped_junk=len(ents) - len(valid),
        clusters=len(clusters),
        merges_by_rule=dict(merges),
        mentions_per_cluster=len(valid) / len(clusters) if clusters else 0.0,
        top_clusters=top_clusters_repr(clusters, info),
    )


def top_clusters_repr(clusters, info, n=12):
    big = sorted(clusters.values(), key=len, reverse=True)[:n]
    out = []
    for ms in big:
        members = sorted({info[i]["label"] for i in ms})
        out.append(dict(size=len(ms), members=members[:8]))
    return out


def canonicalize_gliner(gliner_result, block_on: str = "fine"):
    """block_on='fine' mirrors entity_resolve_v2.py literally (type-block on
    whatever the typer's single "types" field holds — for GLiNER that is the
    141-leaf fine label, per Entity-Type-Taxonomy.md's own worked example:
    "GLiNER types all five bridge, so type-blocking puts them in one block").
    block_on='coarse' is the schema's OWN prescribed remedy if the fine-label
    gate regresses ("two coupled vocabularies... coarse rollup for blocking")
    — included so a regression has an immediate, measured next step instead
    of only a verdict.
    """
    spans = gliner_result["all_spans"]
    key_field = "fine" if block_on == "fine" else "coarse"
    agg = Counter()
    for s in spans:
        agg[(s["text"], s[key_field])] += 1
    mentions = []
    for idx, ((surface, label), count) in enumerate(agg.items()):
        mentions.append(
            dict(
                id=f"g{idx}",
                label=surface,
                types={label},
                mention_count=count,
                proper=surface[:1].isupper(),
            )
        )
    # No relation-extraction graph over this 100-passage sample -> no causal
    # edges available; causal-fingerprint rule naturally contributes 0.
    clusters, merges, valid, info = canonicalize(mentions, edges=None)
    return dict(
        block_on=block_on,
        input_mentions=len(mentions),
        valid_mentions=len(valid),
        dropped_junk=len(mentions) - len(valid),
        clusters=len(clusters),
        merges_by_rule=dict(merges),
        mentions_per_cluster=len(valid) / len(clusters) if clusters else 0.0,
        top_clusters=top_clusters_repr(clusters, info),
    )


def main():
    print("=" * 78)
    print("TASK 7 — GLiNER typer before/after validation (D12 gate)")
    print("=" * 78)

    print("\n[1/4] Loading bert-tiny BEFORE baseline (real shipped production graph)…")
    before = load_bert_tiny_baseline()
    print(f"  {before['unique_entities']} unique entity nodes, {before['total_mentions']} total mentions, "
          f"{before['n_buckets']} coarse buckets used")
    print(f"  top bucket (unique-entity basis): {before['primary_unique_dist'][0]} "
          f"= {before['top_bucket_frac_unique']:.1%}")
    print(f"  top bucket (mention-weighted basis): {before['primary_mentions_dist'][0]} "
          f"= {before['top_bucket_frac_mentions']:.1%}")

    print("\n[2/4] Running GLiNER bi-edge AFTER over passages_100.jsonl…")
    after, typer = run_gliner_over_passages()
    print(f"  {after['passages']} passages, {after['total_spans']} total typed spans "
          f"({after['ms_per_passage']:.0f} ms/passage)")
    print(f"  {after['fine_types_used']}/141 fine types used, {after['coarse_types_used']}/18 coarse used")
    print(f"  top coarse bucket: {after['coarse_dist'][0]} = {after['top_coarse_frac']:.1%}")

    if os.path.exists(PARITY_PATH):
        parity = json.load(open(PARITY_PATH, encoding="utf-8"))
        ref_total = parity.get("parity_fp32", {}).get("ref_total_spans")
        print(f"\n  [harness self-check] parity.json ref_total_spans={ref_total}, "
              f"this run's total_spans={after['total_spans']} "
              f"({'MATCH' if ref_total == after['total_spans'] else 'DIVERGED — investigate before trusting numbers below'})")

    print("\n[3/4] Canonicalization (D12 gate) — bert-tiny BEFORE…")
    before_canon = canonicalize_bert_tiny(before)
    print(f"  {before_canon['valid_mentions']} valid mentions -> {before_canon['clusters']} clusters "
          f"({before_canon['mentions_per_cluster']:.2f} mentions/cluster), merges={before_canon['merges_by_rule']}")

    print("\n[3/4] Canonicalization (D12 gate) — GLiNER AFTER, fine-label blocking (100-passage sample)…")
    after_canon = canonicalize_gliner(after, block_on="fine")
    print(f"  {after_canon['valid_mentions']} valid mentions -> {after_canon['clusters']} clusters "
          f"({after_canon['mentions_per_cluster']:.2f} mentions/cluster), merges={after_canon['merges_by_rule']}")

    print("\n[3/4] Canonicalization (D12 gate) — GLiNER AFTER, coarse-rollup blocking (supplementary)…")
    after_canon_coarse = canonicalize_gliner(after, block_on="coarse")
    print(f"  {after_canon_coarse['valid_mentions']} valid mentions -> {after_canon_coarse['clusters']} clusters "
          f"({after_canon_coarse['mentions_per_cluster']:.2f} mentions/cluster), merges={after_canon_coarse['merges_by_rule']}")

    gate_pass = after_canon["mentions_per_cluster"] >= before_canon["mentions_per_cluster"]
    print(f"\n  D12 GATE (fine-label blocking, mentions/cluster must not drop): "
          f"{before_canon['mentions_per_cluster']:.2f} (before) -> {after_canon['mentions_per_cluster']:.2f} (after) "
          f"=> {'PASS' if gate_pass else 'FAIL'}")
    print(f"  supplementary (coarse-rollup blocking): "
          f"{before_canon['mentions_per_cluster']:.2f} (before) -> {after_canon_coarse['mentions_per_cluster']:.2f} (after) "
          f"=> {'PASS' if after_canon_coarse['mentions_per_cluster'] >= before_canon['mentions_per_cluster'] else 'still short, narrows the gap'}")

    print("\n  bert-tiny BEFORE — 8 largest clusters:")
    for c in before_canon["top_clusters"][:8]:
        print(f"    [{c['size']:2}] {c['members']}")
    print("\n  GLiNER AFTER — 8 largest clusters:")
    for c in after_canon["top_clusters"][:8]:
        print(f"    [{c['size']:2}] {c['members']}")

    print("\n[4/4] Key-entity table…")
    for probe in KEY_ENTITIES:
        b = before["key_entities"][probe]
        a = after["key_entities"][probe]
        print(f"  {probe:26} BEFORE ({b['total_mentions']} mentions): {b['fine_dist']!s:45} "
              f"AFTER ({a['count']} occurrences): {a['fine_dist']}")

    out = dict(
        before=before_canon | dict(
            unique_entities=before["unique_entities"],
            total_mentions=before["total_mentions"],
            n_buckets=before["n_buckets"],
            primary_unique_dist=before["primary_unique_dist"],
            primary_mentions_dist=before["primary_mentions_dist"],
            unique_entropy=before["unique_entropy"],
            mentions_entropy=before["mentions_entropy"],
            top_bucket_frac_unique=before["top_bucket_frac_unique"],
            top_bucket_frac_mentions=before["top_bucket_frac_mentions"],
            key_entities=before["key_entities"],
        ),
        after_coarse_blocked=after_canon_coarse,
        after=after_canon | dict(
            passages=after["passages"],
            total_spans=after["total_spans"],
            ms_per_passage=after["ms_per_passage"],
            fine_types_used=after["fine_types_used"],
            coarse_types_used=after["coarse_types_used"],
            fine_dist=after["fine_dist"],
            coarse_dist=after["coarse_dist"],
            fine_entropy=after["fine_entropy"],
            coarse_entropy=after["coarse_entropy"],
            top_coarse_frac=after["top_coarse_frac"],
            top_fine_frac=after["top_fine_frac"],
            key_entities=after["key_entities"],
        ),
        d12_gate_pass=gate_pass,
    )
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nWrote {OUT_JSON}")
    print("=" * 78)


if __name__ == "__main__":
    main()
