"""Benchmark the agent and the retriever.

Agent metrics (needs GROQ_API_KEY): answer accuracy vs SQL-computed gold answers, tool-routing
F1 (tools used vs tools a correct solution needs), tool-call reliability (share of tool calls that
executed without error), and end-to-end latency p50/p95.
Retriever metrics (offline): recall@1/@5 and MRR on noisy product queries.
Run: python evals/run_evals.py [--retrieval-only]
"""
import json
import random
import re
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "evals"))

import db  # noqa: E402
import tools  # noqa: E402
from config import AGENT_MODEL, ARTIFACTS  # noqa: E402
from questions import Q  # noqa: E402

NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def gold_value(item: dict):
    if "gold_sql" in item:
        return db.run_readonly(item["gold_sql"])["rows"][0][0]
    if item["gold_fn"] == "pr_cut10":
        code = db.run_readonly("SELECT stock_code FROM elasticity WHERE reliable ORDER BY forecast_units_8w DESC LIMIT 1")["rows"][0][0]
        return tools.simulate_price_change(code, [-0.1])["scenarios"][0]["revenue_8w"]
    if item["gold_fn"] == "pr_84879_up5":
        return tools.simulate_price_change("84879", [0.05])["scenarios"][0]["revenue_8w"]
    raise ValueError(item["gold_fn"])


DASHES = str.maketrans({"‐": "-", "‑": "-", "‒": "-", "–": "-", "—": "-", "−": "-"})


def is_correct(answer: str, gold, tol: float) -> bool:
    if gold is None:
        return False
    answer = answer.translate(DASHES)  # models often write non-breaking hyphens / unicode minus signs
    if isinstance(gold, str):
        return gold.lower() in answer.lower()
    nums = [float(n.replace(",", "")) for n in NUM.findall(answer) if n.replace(",", "").replace(".", "").lstrip("-")]
    g = float(gold)
    return any(abs(n - g) <= max(abs(g) * tol, 0.011) for n in nums)


def f1(pred: set, gold: set) -> float:
    if not pred and not gold:
        return 1.0
    tp = len(pred & gold)
    p = tp / len(pred) if pred else 0.0
    r = tp / len(gold) if gold else 0.0
    return 2 * p * r / (p + r) if p + r else 0.0


def pct(xs: list[float], q: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))]


def eval_retrieval(n: int = 300, seed: int = 7) -> dict:
    """Noisy queries: drop one word, shuffle the rest, lowercase, add a typo."""
    rng = random.Random(seed)
    rows = db.run_readonly("SELECT stock_code, description FROM products WHERE length(description) > 12 "
                           "ORDER BY revenue DESC LIMIT 1000", max_rows=1000)["rows"]
    sample = rng.sample(rows, min(n, len(rows)))
    hits1 = hits5 = 0
    rr = []
    for code, desc in sample:
        words = desc.lower().split()
        if len(words) > 2:
            words.pop(rng.randrange(len(words)))
        rng.shuffle(words)
        q = " ".join(words)
        if len(q) > 5:
            i = rng.randrange(1, len(q) - 1)
            q = q[:i] + q[i + 1:]  # delete one character
        res = [r["stock_code"] for r in tools.search_products(q, k=5)["results"]]
        # Several stock codes share a description; count a hit if the description matches.
        descs = [tools._index().products.loc[c, "description"] for c in res]
        ranks = [i for i, d in enumerate(descs) if d == desc]
        hits1 += bool(ranks and ranks[0] == 0)
        hits5 += bool(ranks)
        rr.append(1 / (ranks[0] + 1) if ranks else 0.0)
    return {"queries": len(sample), "recall_at_1": round(hits1 / len(sample), 3),
            "recall_at_5": round(hits5 / len(sample), 3), "mrr": round(sum(rr) / len(rr), 3)}


def eval_agent() -> dict:
    import agent
    results = []
    for item in Q:
        gold = gold_value(item)
        r = agent.run(item["q"])
        used = {s.tool for s in r.steps}
        results.append({"id": item["id"], "question": item["q"], "gold": gold, "answer": r.answer,
                        "correct": is_correct(r.answer, gold, item.get("tol", 0.01)),
                        "tools_expected": sorted(item["tools"]), "tools_used": sorted(used),
                        "routing_f1": round(f1(used, item["tools"]), 3),
                        "tool_calls": len(r.steps), "tool_calls_ok": sum(s.ok for s in r.steps),
                        "seconds": r.seconds, "llm_calls": r.llm_calls, "error": r.error})
        print(f"{item['id']:6} {'OK ' if results[-1]['correct'] else 'BAD'} {r.seconds:5.1f}s "
              f"tools={sorted(used)} gold={gold}", flush=True)
        time.sleep(1.5)  # stay inside free-tier rate limits
    calls = sum(x["tool_calls"] for x in results)
    lat = [x["seconds"] for x in results]
    by_cat: dict[str, list[bool]] = {}
    for x in results:
        by_cat.setdefault(re.sub(r"\d+", "", x["id"]), []).append(x["correct"])
    return {
        "model": AGENT_MODEL, "questions": len(results),
        "answer_accuracy": round(sum(x["correct"] for x in results) / len(results), 3),
        "accuracy_by_category": {k: round(sum(v) / len(v), 3) for k, v in by_cat.items()},
        "routing_f1": round(statistics.mean(x["routing_f1"] for x in results), 3),
        "tool_call_reliability": round(sum(x["tool_calls_ok"] for x in results) / calls, 3) if calls else None,
        "latency_p50_s": round(pct(lat, 0.5), 2), "latency_p95_s": round(pct(lat, 0.95), 2),
        "details": results,
    }


def main() -> None:
    ARTIFACTS.mkdir(exist_ok=True)
    report = {"retrieval": eval_retrieval()}
    print("retrieval:", report["retrieval"], flush=True)
    if "--retrieval-only" not in sys.argv:
        report["agent"] = eval_agent()
        summary = {k: v for k, v in report["agent"].items() if k != "details"}
        print(json.dumps(summary, indent=2))
    (ARTIFACTS / "eval_report.json").write_text(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
