#!/usr/bin/env python3
"""Offline reranking pilot over a SHODH_POOL_EXPORT candidate pool.

Prices the direction-2 question -- "is the encoder's training objective the
ranking bottleneck?" -- without a CI run. Takes the pool.jsonl written by the
recall harness (per case: query, gold ids, RECALL_DIAG_K-deep pool in rank
order), joins candidate text from the corpus jsonl and gold labels from the
cases jsonl, and evaluates rescoring functions offline:

  baseline    the pool's own rank order (what the shipped fusion produced)
  oracle      gold-first ordering of the same pool = ceiling for ANY reranker
  cross-encoder  a query x candidate interaction model (default
              cross-encoder/ms-marco-MiniLM-L-2-v2 / -L-6-v2), the lower bound
              on the interaction-mechanism family: trained for question->passage
              relevance, not paraphrase similarity
  adapter     a query-side linear map W over FROZEN MiniLM embeddings, trained
              on a case-level split (report holdout only) -- the cheapest
              possible "fix the geometry, keep the encoder" arm

Gold ids are taken from the cases file, NOT from pool.jsonl's own gold field:
the first pool export wrote per-run Uuids there (fixed in a later commit), and
joining through the cases file works for both versions.

Usage:
  python scripts/rerank_pilot.py --pool pool.jsonl \
      [--corpus tests/recall/corpora/locomo.jsonl] \
      [--cases tests/recall/locomo_cases.jsonl] \
      [--arms baseline,oracle,ce,adapter] [--ce-model MODEL] [--rerank-depth 100]
      [--max-cases N] [--out results.json]

Every arm reports recall@10, ndcg@10, p@1 overall and per category, plus the
share of the oracle headroom (oracle - baseline) it captured. The CE arm also
reports pairs/sec and per-query latency at the chosen rerank depth, which is
the latency half of the cost/benefit verdict.
"""

import argparse
import json
import math
import sys
import time
from collections import defaultdict


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def ndcg_at_k(ranked_ids, gold_grades, k=10):
    """Binary-graded ndcg matching the harness's per-case shape closely enough
    for arm-to-arm comparison (graded when grades differ)."""
    dcg = 0.0
    for i, cid in enumerate(ranked_ids[:k]):
        g = gold_grades.get(cid, 0)
        if g > 0:
            dcg += (2**g - 1) / math.log2(i + 2)
    ideal = sorted(gold_grades.values(), reverse=True)[:k]
    idcg = sum((2**g - 1) / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def metrics_for(order_by_case, gold_by_case, grades_by_case, cats):
    per_cat = defaultdict(lambda: defaultdict(list))
    agg = defaultdict(list)
    for cid, ranked in order_by_case.items():
        gold = gold_by_case[cid]
        r10 = len(gold & set(ranked[:10])) / len(gold)
        p1 = 1.0 if ranked and ranked[0] in gold else 0.0
        nd = ndcg_at_k(ranked, grades_by_case[cid], 10)
        for key, val in (("recall@10", r10), ("p@1", p1), ("ndcg@10", nd)):
            agg[key].append(val)
            per_cat[cats[cid]][key].append(val)
    out = {k: sum(v) / len(v) for k, v in agg.items()}
    out["by_category"] = {
        c: {k: sum(v) / len(v) for k, v in m.items()} for c, m in per_cat.items()
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--corpus", default="tests/recall/corpora/locomo.jsonl")
    ap.add_argument("--cases", default="tests/recall/locomo_cases.jsonl")
    ap.add_argument("--arms", default="baseline,oracle,ce")
    ap.add_argument("--ce-model", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    ap.add_argument("--rerank-depth", type=int, default=100)
    ap.add_argument("--max-cases", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--adapter-epochs", type=int, default=30)
    ap.add_argument("--holdout-frac", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    arms = set(args.arms.split(","))

    corpus_text = {d["id"]: d["content"] for d in load_jsonl(args.corpus)}
    cases = load_jsonl(args.cases)
    gold_by_case = {
        d["id"]: set(r["corpus_item_id"] for r in d["relevant"]) for d in cases
    }
    grades_by_case = {
        d["id"]: {r["corpus_item_id"]: r.get("grade", 1) for r in d["relevant"]}
        for d in cases
    }

    pool_rows = load_jsonl(args.pool)
    if args.max_cases:
        pool_rows = pool_rows[: args.max_cases]
    cats = {d["case_id"]: d.get("category", "?") for d in pool_rows}
    queries = {d["case_id"]: d["query"] for d in pool_rows}
    pools = {d["case_id"]: d["pool"] for d in pool_rows}

    # Sanity: every pool id must join against corpus text.
    missing = sum(
        1 for p in pools.values() for cid in p if cid not in corpus_text
    )
    total_pool_ids = sum(len(p) for p in pools.values())
    print(
        f"cases={len(pools)} pool_ids={total_pool_ids} "
        f"unjoinable={missing} ({100.0 * missing / max(total_pool_ids, 1):.2f}%)",
        flush=True,
    )
    if missing / max(total_pool_ids, 1) > 0.01:
        sys.exit("pool ids do not join against the corpus -- wrong corpus file?")

    results = {"pool": args.pool, "cases": len(pools)}

    if "baseline" in arms:
        results["baseline"] = metrics_for(pools, gold_by_case, grades_by_case, cats)
        print("baseline ", json.dumps({k: round(v, 4) for k, v in results["baseline"].items() if isinstance(v, float)}), flush=True)

    if "oracle" in arms:
        oracle_order = {
            cid: sorted(p, key=lambda x: (x not in gold_by_case[cid],))
            for cid, p in pools.items()
        }
        results["oracle"] = metrics_for(oracle_order, gold_by_case, grades_by_case, cats)
        print("oracle   ", json.dumps({k: round(v, 4) for k, v in results["oracle"].items() if isinstance(v, float)}), flush=True)

    if "ce" in arms:
        from sentence_transformers import CrossEncoder

        model = CrossEncoder(args.ce_model, max_length=256, device="cpu")
        pairs, index = [], []
        for cid, p in pools.items():
            for rank, corpus_id in enumerate(p[: args.rerank_depth]):
                pairs.append((queries[cid], corpus_text.get(corpus_id, "")))
                index.append((cid, corpus_id))
        t0 = time.time()
        scores = model.predict(
            pairs, batch_size=args.batch_size, show_progress_bar=False
        )
        elapsed = time.time() - t0
        pairs_per_sec = len(pairs) / elapsed
        scored = defaultdict(dict)
        for (cid, corpus_id), s in zip(index, scores):
            scored[cid][corpus_id] = float(s)
        ce_order = {}
        for cid, p in pools.items():
            head = p[: args.rerank_depth]
            tail = p[args.rerank_depth:]
            head = sorted(head, key=lambda x: -scored[cid].get(x, -1e9))
            ce_order[cid] = head + tail
        results["ce"] = metrics_for(ce_order, gold_by_case, grades_by_case, cats)
        results["ce"]["model"] = args.ce_model
        results["ce"]["rerank_depth"] = args.rerank_depth
        results["ce"]["pairs_per_sec"] = pairs_per_sec
        results["ce"]["ms_per_query_at_depth"] = 1000.0 * args.rerank_depth / pairs_per_sec
        print(
            "ce       ",
            json.dumps({k: round(v, 4) for k, v in results["ce"].items() if isinstance(v, float)}),
            f"({pairs_per_sec:.0f} pairs/s, {results['ce']['ms_per_query_at_depth']:.0f} ms/query at depth {args.rerank_depth})",
            flush=True,
        )

    if "adapter" in arms:
        import numpy as np
        import torch
        from sentence_transformers import SentenceTransformer

        torch.manual_seed(args.seed)
        rng = np.random.default_rng(args.seed)
        st = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")

        needed_ids = sorted({cid for p in pools.values() for cid in p})
        cand_emb = st.encode(
            [corpus_text[c] for c in needed_ids],
            batch_size=256, convert_to_numpy=True, normalize_embeddings=True,
            show_progress_bar=False,
        )
        cand_ix = {c: i for i, c in enumerate(needed_ids)}
        case_ids = sorted(pools.keys())
        q_emb = st.encode(
            [queries[c] for c in case_ids],
            batch_size=256, convert_to_numpy=True, normalize_embeddings=True,
            show_progress_bar=False,
        )
        q_ix = {c: i for i, c in enumerate(case_ids)}

        # Case-level split. Holdout is the ONLY number reported: these 1531
        # cases are the same distribution other fitted components trained on,
        # so an in-sample adapter number would be worthless.
        perm = rng.permutation(len(case_ids))
        n_hold = int(len(case_ids) * args.holdout_frac)
        hold = {case_ids[i] for i in perm[:n_hold]}
        train = [c for c in case_ids if c not in hold]

        dim = q_emb.shape[1]
        W = torch.eye(dim, requires_grad=True)
        opt = torch.optim.Adam([W], lr=1e-3)
        qt = torch.tensor(q_emb)
        ct = torch.tensor(cand_emb)
        # In-pool contrastive: gold candidates are positives, the rest of the
        # SAME pool are negatives -- exactly the discrimination the pipeline
        # needs at ranks 11-100.
        train_rows = []
        for cid in train:
            g = gold_by_case[cid]
            pos = [cand_ix[x] for x in pools[cid] if x in g]
            neg = [cand_ix[x] for x in pools[cid] if x not in g]
            if pos and neg:
                train_rows.append((q_ix[cid], pos, neg))
        for _ in range(args.adapter_epochs):
            rng.shuffle(train_rows)
            for qi, pos, neg in train_rows:
                qv = qt[qi] @ W
                qv = qv / (qv.norm() + 1e-9)
                logits = ct[pos + neg] @ qv / 0.05
                target = torch.zeros(len(pos) + len(neg))
                target[: len(pos)] = 1.0 / len(pos)
                loss = -(torch.log_softmax(logits, 0) * target).sum()
                opt.zero_grad()
                loss.backward()
                opt.step()
        with torch.no_grad():
            Wn = W.detach()
            ad_order = {}
            for cid in hold:
                qv = qt[q_ix[cid]] @ Wn
                qv = qv / (qv.norm() + 1e-9)
                p = pools[cid]
                s = (ct[[cand_ix[x] for x in p]] @ qv).numpy()
                ad_order[cid] = [x for _, x in sorted(zip(-s, p))]
        hold_gold = {c: gold_by_case[c] for c in hold}
        hold_grades = {c: grades_by_case[c] for c in hold}
        results["adapter_holdout"] = metrics_for(ad_order, hold_gold, hold_grades, cats)
        results["adapter_holdout"]["holdout_cases"] = len(hold)
        # Baselines restricted to the SAME holdout so the comparison is fair.
        results["baseline_holdout"] = metrics_for(
            {c: pools[c] for c in hold}, hold_gold, hold_grades, cats
        )
        print("adapter(holdout) ", json.dumps({k: round(v, 4) for k, v in results["adapter_holdout"].items() if isinstance(v, float)}), flush=True)
        print("baseline(holdout)", json.dumps({k: round(v, 4) for k, v in results["baseline_holdout"].items() if isinstance(v, float)}), flush=True)

    if "baseline" in arms and "oracle" in arms and "ce" in arms:
        b = results["baseline"]["recall@10"]
        o = results["oracle"]["recall@10"]
        c = results["ce"]["recall@10"]
        if o > b:
            results["ce"]["headroom_captured"] = (c - b) / (o - b)
            print(f"ce captured {100.0 * (c - b) / (o - b):.1f}% of the oracle headroom ({b:.4f} -> {c:.4f}, ceiling {o:.4f})", flush=True)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
