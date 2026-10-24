#!/usr/bin/env python3
"""Fit the trust gate (src/memory/trust_gate.rs) from the harness feature export.

Input: the JSONL that `recall-eval` writes when SHODH_FUSION_FEATURE_EXPORT is
set, one line per query in full mode:

    {"case_id", "category", "query_len", "gold_final_rank",
     "features": {... "gold_vec_rank", "gold_bm_rank", ...},
     "gate": {"features": [f0..f9], "eval": {...}}}

`gate.features` is the exact array the Rust gate consumed, in
trust_gate::FEATURE_NAMES order, so the fit sees what the gate will see.

Label: 1 when the vector leg ranked this query's gold above BM25, 0 when BM25
ranked it above the vector leg. A query where neither leg found gold, or both
found it at the same rank, says nothing about which leg to trust and is left out.

Holdout: every case whose id starts with --holdout-prefix (default conv-42,
the conversation the locomo-gate suite is drawn from) is kept out of the fit
and reported on separately, so the gate's PR check stays a held-out measurement.

Output: the Rust `FIT` table and `FIT_BIAS` to paste into trust_gate.rs, plus
train and holdout AUC, accuracy, and the spread of t on each side.

Standard library only.
"""
import argparse
import json
import math
import statistics
import sys

NAMES = [
    "log_bm_peak",
    "log_vec_peak",
    "agreement_top10",
    "max_bm",
    "max_vec",
    "n_vec_strong",
    "vec_mass_10_of_30",
    "bm_mass_10_of_30",
    "graph_max_activation",
    "query_len",
]


def label(features: dict):
    v = features.get("gold_vec_rank")
    b = features.get("gold_bm_rank")
    if v is None and b is None:
        return None
    if v is None:
        return 0.0
    if b is None:
        return 1.0
    if v == b:
        return None
    return 1.0 if v < b else 0.0


def load(path, holdout_prefix):
    train_x, train_y, hold_x, hold_y = [], [], [], []
    skipped = 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            gate = row.get("gate")
            if not gate or "features" not in gate:
                skipped += 1
                continue
            y = label(row["features"])
            if y is None:
                skipped += 1
                continue
            x = [float(v) for v in gate["features"]]
            if len(x) != len(NAMES):
                sys.exit(f"{row.get('case_id')}: expected {len(NAMES)} features, got {len(x)}")
            if row["case_id"].startswith(holdout_prefix):
                hold_x.append(x)
                hold_y.append(y)
            else:
                train_x.append(x)
                train_y.append(y)
    return train_x, train_y, hold_x, hold_y, skipped


def sigmoid(z):
    z = max(-30.0, min(30.0, z))
    return 1.0 / (1.0 + math.exp(-z))


def standardise(x, mu, sd, clamp):
    return [max(-clamp, min(clamp, (xi - m) / s)) for xi, m, s in zip(x, mu, sd)]


def fit(xs, ys, l2, iters, lr, clamp):
    """Logistic regression on standardised, clamped features; full-batch gradient descent with L2."""
    d = len(NAMES)
    n = len(ys)
    mu = [sum(x[j] for x in xs) / n for j in range(d)]
    sd = [math.sqrt(sum((x[j] - mu[j]) ** 2 for x in xs) / n) for j in range(d)]
    sd = [s if s > 1e-9 else 1.0 for s in sd]
    zs = [standardise(x, mu, sd, clamp) for x in xs]
    w = [0.0] * d
    b = 0.0
    for _ in range(iters):
        gw = [0.0] * d
        gb = 0.0
        for z, y in zip(zs, ys):
            g = sigmoid(sum(wj * zj for wj, zj in zip(w, z)) + b) - y
            gb += g
            for j in range(d):
                gw[j] += g * z[j]
        for j in range(d):
            w[j] -= lr * (gw[j] / n + l2 * w[j])
        b -= lr * gb / n
    return mu, sd, w, b


def predict(mu, sd, w, b, xs, clamp):
    return [sigmoid(sum(wj * zj for wj, zj in zip(w, standardise(x, mu, sd, clamp))) + b) for x in xs]


def auc(p, y):
    pos = [pi for pi, yi in zip(p, y) if yi == 1.0]
    neg = [pi for pi, yi in zip(p, y) if yi == 0.0]
    if not pos or not neg:
        return float("nan")
    wins = 0.0
    for a in pos:
        for c in neg:
            wins += 1.0 if a > c else 0.5 if a == c else 0.0
    return wins / (len(pos) * len(neg))


def describe(name, p, y):
    if not y:
        print(f"[{name}] n=0")
        return
    acc = sum(1 for pi, yi in zip(p, y) if (pi >= 0.5) == (yi == 1.0)) / len(y)
    print(
        f"[{name}] n={len(y)} pos={int(sum(y))} auc={auc(p, y):.3f} acc={acc:.3f} "
        f"t min={min(p):.3f} median={statistics.median(p):.3f} max={max(p):.3f}"
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("export", help="SHODH_FUSION_FEATURE_EXPORT JSONL")
    ap.add_argument("--holdout-prefix", default="conv-42")
    ap.add_argument("--l2", type=float, default=0.01)
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--clamp", type=float, default=4.0, help="must equal trust_gate::FIT_RANGE_SD")
    ap.add_argument("--run", default="", help="run id or note recorded beside the table")
    args = ap.parse_args()

    tx, ty, hx, hy, skipped = load(args.export, args.holdout_prefix)
    if not ty:
        sys.exit("no labelled training queries in the export")
    print(f"train queries={len(ty)} holdout({args.holdout_prefix}) queries={len(hy)} unlabelled/skipped={skipped}")
    mu, sd, w, b = fit(tx, ty, args.l2, args.iters, args.lr, args.clamp)
    describe("train", predict(mu, sd, w, b, tx, args.clamp), ty)
    if hy:
        describe("holdout", predict(mu, sd, w, b, hx, args.clamp), hy)

    print()
    print("// Paste into src/memory/trust_gate.rs. Order matches FEATURE_NAMES.")
    print(f"// Fit: {args.run or 'scripts/fit_trust_gate.py'}; train n={len(ty)}, holdout {args.holdout_prefix} n={len(hy)}, l2={args.l2}")
    print("pub const FIT: [(f32, f32, f32); FEATURE_COUNT] = [")
    for name, m, s_, wi in zip(NAMES, mu, sd, w):
        print(f"    ({m:.6g}, {s_:.6g}, {wi:.6g}), // {name}")
    print("];")
    print(f"pub const FIT_BIAS: f32 = {b:.6g};")


if __name__ == "__main__":
    main()
