#!/usr/bin/env python3
"""Fit the calibrated additive fusion (roadmap ②, Phase A of the 80% plan).

Input: target/fusion_training.jsonl — one line per query from the harness
feature export, each carrying candidate-level rows:
    features.candidates = [{vec, bm25, graph, is_gold}, ...]
plus gold_final_rank (the CURRENT default fusion's outcome for this query),
which gives the baseline for the offline re-ranking simulation.

Model (9 parameters):
    score(c) = w_v*sigmoid(a_v + b_v*vec) + w_b*sigmoid(a_b + b_b*bm25)
             + w_g*sigmoid(a_g + b_g*graph)
Per-leg logistic calibration maps each leg's query-relative score to a
comparable [0,1] "evidence" scale (the failure of the raw min-max SUM was
exactly the lack of this), then a weighted SUM combines evidence additively
(the ACT-R form). Loss: pairwise logistic over (gold, non-gold) pairs within
each query — a ranking objective, not pointwise classification.

Output: fitted coefficients (Rust-pasteable) + holdout metrics:
  - pairwise accuracy (gold above non-gold)
  - simulated recall@10 re-ranking the exported candidate pools, vs the
    baseline recall@10 implied by gold_final_rank (the pre-registration
    number for the SHODH_FUSION_CALSUM A/B).

Split: 70/30 by stable hash of case_id (no leakage across queries).
"""

import json
import math
import sys
import hashlib
from pathlib import Path

import numpy as np

PATH = Path(sys.argv[1] if len(sys.argv) > 1 else "target/fusion_training.jsonl")
K = 10  # recall@K for the simulation


def query_phi(feats: dict) -> np.ndarray:
    """Per-query conditioning features for the stage-2 leg weights — the same
    family the fitted trust gate proved discriminative (bm peakedness,
    vec peakedness, leg agreement, pool shapes)."""
    return np.array(
        [
            1.0,
            math.log(max(feats.get("bm_peak", 1.0), 1e-3)),
            math.log(max(feats.get("vec_peak", 1.0), 1e-3)),
            feats.get("agreement_top10", 0.0),
            math.log1p(feats.get("n_hybrid", 0)),
            math.log1p(feats.get("n_graph", 0)),
        ],
        dtype=np.float64,
    )


N_PHI = 6


def load(path: Path):
    queries = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        feats = row.get("features", {})
        cands = feats.get("candidates", [])
        golds = [c for c in cands if c["is_gold"]]
        negs = [c for c in cands if not c["is_gold"]]
        if not golds or not negs:
            continue  # nothing to learn/rank for this query
        queries.append(
            {
                "case_id": row["case_id"],
                "category": row.get("category", "?"),
                "gold_final_rank": row.get("gold_final_rank"),
                "X": np.array(
                    [[c["vec"], c["bm25"], c["graph"]] for c in cands], dtype=np.float64
                ),
                "y": np.array([1.0 if c["is_gold"] else 0.0 for c in cands]),
                "phi": query_phi(feats),
            }
        )
    return queries


def split(queries):
    train, hold = [], []
    for q in queries:
        h = int(hashlib.sha1(q["case_id"].encode()).hexdigest(), 16) % 100
        (train if h < 70 else hold).append(q)
    return train, hold


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def scores(params, X):
    a = params[0:3]  # calibration intercepts (vec, bm25, graph)
    b = params[3:6]  # calibration slopes
    w = params[6:9]  # leg weights (softplus to keep positive)
    wpos = np.log1p(np.exp(w))
    cal = sigmoid(a[None, :] + b[None, :] * X)  # (n,3)
    return cal @ wpos


def query_loss_grad(params, q):
    """Pairwise logistic loss + gradient for one query (gold vs sampled negs)."""
    X, y = q["X"], q["y"]
    s = scores(params, X)
    gold_idx = np.where(y == 1.0)[0]
    neg_idx = np.where(y == 0.0)[0]
    # Cap negatives per gold for cost; hardest (highest-scored) negatives matter.
    if len(neg_idx) > 40:
        neg_idx = neg_idx[np.argsort(-s[neg_idx])[:40]]
    a = params[0:3]
    b = params[3:6]
    w = params[6:9]
    wpos = np.log1p(np.exp(w))
    dw_dwraw = sigmoid(w)  # d softplus
    cal = sigmoid(a[None, :] + b[None, :] * X)
    dcal = cal * (1 - cal)  # (n,3)

    loss = 0.0
    grad = np.zeros_like(params)
    npairs = 0
    for g in gold_idx:
        diff = s[g] - s[neg_idx]
        p = sigmoid(diff)
        loss += float(np.sum(np.log1p(np.exp(-np.clip(diff, -30, 30)))))
        coef = -(1 - p)  # dL/ddiff
        # ds/da_j = wpos_j * dcal_j ; ds/db_j = wpos_j * dcal_j * x_j ; ds/dw_j = cal_j * dsoftplus
        for j in range(3):
            ds_g_a = wpos[j] * dcal[g, j]
            ds_n_a = wpos[j] * dcal[neg_idx, j]
            grad[j] += float(np.sum(coef * (ds_g_a - ds_n_a)))
            ds_g_b = wpos[j] * dcal[g, j] * X[g, j]
            ds_n_b = wpos[j] * dcal[neg_idx, j] * X[neg_idx, j]
            grad[3 + j] += float(np.sum(coef * (ds_g_b - ds_n_b)))
            ds_g_w = cal[g, j] * dw_dwraw[j]
            ds_n_w = cal[neg_idx, j] * dw_dwraw[j]
            grad[6 + j] += float(np.sum(coef * (ds_g_w - ds_n_w)))
        npairs += len(neg_idx)
    return loss, grad, npairs


def fit(train, iters=300, lr=0.05, seed=7):
    rng = np.random.default_rng(seed)
    # Init: identity-ish calibration, equal weights.
    params = np.array([-2.0, -2.0, -2.0, 4.0, 4.0, 4.0, 0.5, 0.5, 0.5])
    for it in range(iters):
        total, gradsum, pairs = 0.0, np.zeros_like(params), 0
        for q in train:
            l, g, n = query_loss_grad(params, q)
            total += l
            gradsum += g
            pairs += n
        params -= lr * gradsum / max(pairs, 1)
        if it % 50 == 0 or it == iters - 1:
            print(f"  iter {it:3d} pairwise-loss/pair = {total / max(pairs,1):.4f}")
    return params


def evaluate(params, queries, label):
    pair_correct = pair_total = 0
    hit_fitted = hit_baseline = n = 0
    by_cat = {}
    for q in queries:
        s = scores(params, q["X"])
        gold_idx = np.where(q["y"] == 1.0)[0]
        neg_idx = np.where(q["y"] == 0.0)[0]
        for g in gold_idx:
            pair_correct += int(np.sum(s[g] > s[neg_idx]))
            pair_total += len(neg_idx)
        order = np.argsort(-s)
        topk = set(order[:K].tolist())
        fitted_hit = any(int(g) in topk for g in gold_idx)
        base_rank = q["gold_final_rank"]
        base_hit = base_rank is not None and base_rank < K
        hit_fitted += int(fitted_hit)
        hit_baseline += int(base_hit)
        n += 1
        c = q["category"]
        agg = by_cat.setdefault(c, [0, 0, 0])
        agg[0] += int(fitted_hit)
        agg[1] += int(base_hit)
        agg[2] += 1
    print(f"\n[{label}] queries={n}")
    print(f"  pairwise accuracy: {pair_correct / max(pair_total,1):.4f}")
    print(
        f"  simulated recall@{K}: fitted={hit_fitted / max(n,1):.4f} "
        f"baseline(FLAT)={hit_baseline / max(n,1):.4f} "
        f"delta={ (hit_fitted - hit_baseline) / max(n,1):+.4f}"
    )
    for c, (f, bse, m) in sorted(by_cat.items()):
        print(f"    {c:14s} fitted={f/m:.4f} baseline={bse/m:.4f} (n={m})")


# ---------------------------------------------------------------------------
# Stage 2: query-conditioned leg weights.
#   score(c|q) = sum_l softplus(u_l . phi(q)) * sigmoid(a_l + b_l * s_l(c))
# Params: a(3) + b(3) + U(3*N_PHI). The global model is the special case
# u_l = (w_l, 0, ..., 0) — strictly more expressive, regularized toward it.
# ---------------------------------------------------------------------------


def s2_unpack(params):
    a = params[0:3]
    b = params[3:6]
    U = params[6:].reshape(3, N_PHI)
    return a, b, U


def s2_scores(params, X, phi):
    a, b, U = s2_unpack(params)
    z = U @ phi  # (3,)
    w = np.log1p(np.exp(np.clip(z, -30, 30)))
    cal = sigmoid(a[None, :] + b[None, :] * X)
    return cal @ w


def s2_query_loss_grad(params, q, l2=0.01):
    X, y, phi = q["X"], q["y"], q["phi"]
    a, b, U = s2_unpack(params)
    z = U @ phi
    w = np.log1p(np.exp(np.clip(z, -30, 30)))
    dw_dz = sigmoid(z)
    cal = sigmoid(a[None, :] + b[None, :] * X)
    dcal = cal * (1 - cal)
    s = cal @ w

    gold_idx = np.where(y == 1.0)[0]
    neg_idx = np.where(y == 0.0)[0]
    if len(neg_idx) > 40:
        neg_idx = neg_idx[np.argsort(-s[neg_idx])[:40]]

    loss = 0.0
    grad = np.zeros_like(params)
    npairs = 0
    for g in gold_idx:
        diff = s[g] - s[neg_idx]
        p = sigmoid(diff)
        loss += float(np.sum(np.log1p(np.exp(-np.clip(diff, -30, 30)))))
        coef = -(1 - p)
        for j in range(3):
            d_a = w[j] * (dcal[g, j] - dcal[neg_idx, j])
            grad[j] += float(np.sum(coef * d_a))
            d_b = w[j] * (dcal[g, j] * X[g, j] - dcal[neg_idx, j] * X[neg_idx, j])
            grad[3 + j] += float(np.sum(coef * d_b))
            d_z = (cal[g, j] - cal[neg_idx, j]) * dw_dz[j]
            gz = float(np.sum(coef * d_z))
            grad[6 + j * N_PHI : 6 + (j + 1) * N_PHI] += gz * phi
        npairs += len(neg_idx)
    # L2 on the non-intercept weight features pulls toward the global model.
    for j in range(3):
        sl = slice(6 + j * N_PHI + 1, 6 + (j + 1) * N_PHI)
        loss += l2 * float(np.sum(params[sl] ** 2))
        grad[sl] += 2 * l2 * params[sl]
    return loss, grad, npairs


def s2_fit(train, iters=400, lr=0.05):
    params = np.zeros(6 + 3 * N_PHI)
    params[0:3] = -2.0
    params[3:6] = 4.0
    params[6::N_PHI] = 0.5  # intercept feature of each leg's weight
    for it in range(iters):
        total, gradsum, pairs = 0.0, np.zeros_like(params), 0
        for q in train:
            l, g, n = s2_query_loss_grad(params, q)
            total += l
            gradsum += g
            pairs += n
        params -= lr * gradsum / max(pairs, 1)
        if it % 100 == 0 or it == iters - 1:
            print(f"  s2 iter {it:3d} pairwise-loss/pair = {total / max(pairs,1):.4f}")
    return params


def evaluate_generic(score_fn, queries, label):
    pair_correct = pair_total = 0
    hit_fitted = hit_baseline = hit_oracle = n = 0
    by_cat = {}
    for q in queries:
        s = score_fn(q)
        gold_idx = np.where(q["y"] == 1.0)[0]
        neg_idx = np.where(q["y"] == 0.0)[0]
        for g in gold_idx:
            pair_correct += int(np.sum(s[g] > s[neg_idx]))
            pair_total += len(neg_idx)
        order = np.argsort(-s)
        topk = set(order[:K].tolist())
        fitted_hit = any(int(g) in topk for g in gold_idx)
        base_rank = q["gold_final_rank"]
        base_hit = base_rank is not None and base_rank < K
        # Oracle: gold reachable in top-K by ANY single leg's own ordering.
        oracle_hit = False
        for col in range(3):
            leg_order = np.argsort(-q["X"][:, col])
            if any(int(g) in set(leg_order[:K].tolist()) for g in gold_idx):
                oracle_hit = True
                break
        hit_fitted += int(fitted_hit)
        hit_baseline += int(base_hit)
        hit_oracle += int(oracle_hit)
        n += 1
        agg = by_cat.setdefault(q["category"], [0, 0, 0])
        agg[0] += int(fitted_hit)
        agg[1] += int(base_hit)
        agg[2] += 1
    print(f"\n[{label}] queries={n}")
    print(f"  pairwise accuracy: {pair_correct / max(pair_total,1):.4f}")
    print(
        f"  simulated recall@{K}: fitted={hit_fitted / max(n,1):.4f} "
        f"baseline(FLAT)={hit_baseline / max(n,1):.4f} "
        f"delta={(hit_fitted - hit_baseline) / max(n,1):+.4f} "
        f"| single-leg oracle={hit_oracle / max(n,1):.4f}"
    )
    for c, (f, bse, m) in sorted(by_cat.items()):
        print(f"    {c:14s} fitted={f/m:.4f} baseline={bse/m:.4f} (n={m})")


def main():
    queries = load(PATH)
    print(f"loaded {len(queries)} usable queries from {PATH}")
    if not queries:
        sys.exit("no usable queries — was the export armed and candidates populated?")
    train, hold = split(queries)
    print(f"train={len(train)} holdout={len(hold)}")

    print("\n=== stage 1: global per-leg weights (control) ===")
    p1 = fit(train)
    evaluate_generic(lambda q: scores(p1, q["X"]), hold, "s1 holdout")

    print("\n=== stage 2: query-conditioned leg weights ===")
    p2 = s2_fit(train)
    evaluate_generic(lambda q: s2_scores(p2, q["X"], q["phi"]), train, "s2 train")
    evaluate_generic(lambda q: s2_scores(p2, q["X"], q["phi"]), hold, "s2 holdout")

    a, b, U = s2_unpack(p2)
    print("\nRust constants (stage 2, paste into the CALSUM mode):")
    print(f"  // fitted {PATH.name}: calibration (intercept, slope) per leg")
    for name, i in (("VEC", 0), ("BM25", 1), ("GRAPH", 2)):
        print(f"  const CALSUM_CAL_{name}: (f32, f32) = ({a[i]:.4f}, {b[i]:.4f});")
    print("  // weight features: [1, ln bm_peak, ln vec_peak, agreement, ln1p n_hybrid, ln1p n_graph]")
    for name, i in (("VEC", 0), ("BM25", 1), ("GRAPH", 2)):
        row = ", ".join(f"{v:.4f}" for v in U[i])
        print(f"  const CALSUM_W_{name}: [f32; {N_PHI}] = [{row}];")


if __name__ == "__main__":
    main()
