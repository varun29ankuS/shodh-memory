#!/usr/bin/env python3
"""Reader-composition study: answer LongMemEval questions from shodh retrievals.

The memory system is LLM-free; this study composes it with a SMALL LOCAL reader
model (llama.cpp SERVER, CPU, temperature 0, fixed seed) to measure the
answer-half on top of our published retrieval-half. Design constraints:

- The reader sees EXACTLY what the memory system retrieved (top-k turn texts
  from `recall-eval ... SHODH_DUMP_CONTEXT=...`), nothing else. Gold answers
  are never in the dump; they are joined here from the pinned upstream dataset
  by question id, only for scoring.
- The model is served by llama-server and loaded ONCE (per-invocation reloads
  of a 2GB model made per-question latency minutes, not seconds).
- Scoring is reported two ways, both clearly labeled:
    * substring-EM: normalized gold answer contained in the model answer.
      Deterministic, conservative (paraphrases score 0).
    * local-judge: the same local model judges answer/gold equivalence.
      NOT comparable to GPT-4-judged numbers in the literature; reported as a
      supplementary signal with that caveat attached.

Usage:
  reader_composition.py answer --dump ctx.jsonl --server http://127.0.0.1:8080 \
      --out answers.jsonl
  reader_composition.py score --answers answers.jsonl --dataset lme.json \
      --server http://127.0.0.1:8080 --out scores.json
"""

import argparse
import json
import pathlib
import re
import sys
import time
import urllib.request

TURN_CHAR_CAP = 800  # bound per-turn prompt contribution


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def ask(server: str, prompt: str, n_predict: int) -> str:
    body = json.dumps({
        "prompt": prompt,
        "n_predict": n_predict,
        "temperature": 0,
        "seed": 29,
        "cache_prompt": False,
    }).encode("utf-8")
    req = urllib.request.Request(
        server.rstrip("/") + "/completion",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                return json.loads(resp.read())["content"].strip()
        except Exception as e:  # transient server hiccup: brief retry
            if attempt == 2:
                raise
            print(f"  retry after: {e}", file=sys.stderr)
            time.sleep(5)
    raise RuntimeError("unreachable")


def wait_ready(server: str, timeout_s: int = 300) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(server.rstrip("/") + "/health", timeout=5) as r:
                if r.status == 200:
                    return
        except Exception:
            pass
        time.sleep(3)
    raise RuntimeError("llama-server never became healthy")


def cmd_answer(args: argparse.Namespace) -> None:
    wait_ready(args.server)
    out_path = pathlib.Path(args.out)
    done = set()
    if out_path.exists():  # resumable
        for line in out_path.open(encoding="utf-8"):
            done.add(json.loads(line)["question_id"])
    with out_path.open("a", encoding="utf-8") as sink:
        for i, line in enumerate(open(args.dump, encoding="utf-8")):
            rec = json.loads(line)
            if rec["question_id"] in done:
                continue
            context = "\n".join(
                f"[{r['rank']}] {r['content'][:TURN_CHAR_CAP]}"
                for r in rec["retrieved"]
            )
            prompt = (
                "You are answering a question about a user's conversation history. "
                "Use ONLY the retrieved memory excerpts below. If the answer is not "
                "in them, answer with your best guess from them anyway.\n\n"
                f"Retrieved memories:\n{context}\n\n"
                f"Question: {rec['question']}\n"
                "Answer (one short sentence, no explanation):"
            )
            answer = ask(args.server, prompt, 96)
            sink.write(json.dumps({
                "question_id": rec["question_id"],
                "question": rec["question"],
                "category": rec["category"],
                "gold_in_topk": rec["gold_in_topk"],
                "answer": answer,
            }) + "\n")
            sink.flush()
            if (i + 1) % 10 == 0:
                print(f"  answered {i + 1}", file=sys.stderr)


def cmd_score(args: argparse.Namespace) -> None:
    data = json.load(open(args.dataset, encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("questions", list(data.values()))
    gold = {q["question_id"]: str(q.get("answer", "")) for q in data}

    if args.server:
        wait_ready(args.server)
    rows = [json.loads(l) for l in open(args.answers, encoding="utf-8")]
    em_hits, judge_hits, per_cat = 0, 0, {}
    for r in rows:
        g = gold.get(r["question_id"], "")
        em = bool(g) and normalize(g) in normalize(r["answer"])
        judge = None
        if args.server:
            verdict = ask(
                args.server,
                "Judge whether the candidate answer conveys the same fact as the "
                f"gold answer.\nQuestion: {r.get('question', '')}\n"
                f"Gold answer: {g}\nCandidate answer: {r['answer']}\n"
                "Reply with exactly one word, yes or no:",
                4,
            )
            judge = verdict.strip().lower().startswith("yes")
        em_hits += em
        judge_hits += bool(judge)
        c = per_cat.setdefault(r["category"], {"n": 0, "em": 0, "judge": 0})
        c["n"] += 1
        c["em"] += em
        c["judge"] += bool(judge)
        r["em"] = em
        r["judge"] = judge

    n = len(rows)
    report = {
        "questions": n,
        "substring_em": em_hits / n if n else 0.0,
        "local_judge": judge_hits / n if n else 0.0,
        "judge_caveat": "local small-model judge; NOT comparable to GPT-4-judged numbers",
        "by_category": {
            c: {"n": v["n"], "em": v["em"] / v["n"], "judge": v["judge"] / v["n"]}
            for c, v in sorted(per_cat.items())
        },
        "rows": rows,
    }
    pathlib.Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("answer")
    a.add_argument("--dump", required=True)
    a.add_argument("--server", required=True)
    a.add_argument("--out", required=True)
    a.set_defaults(fn=cmd_answer)
    s = sub.add_parser("score")
    s.add_argument("--answers", required=True)
    s.add_argument("--dataset", required=True)
    s.add_argument("--server", default=None)
    s.add_argument("--out", required=True)
    s.set_defaults(fn=cmd_score)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
