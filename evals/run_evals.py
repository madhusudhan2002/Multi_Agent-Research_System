#!/usr/bin/env python3
"""
run_evals.py — batch-runs the eval query set through the pipeline and
reports topic coverage, latency, and cost. This is what separates a
resume-worthy agent project from a toy: you can point to real, measured
numbers instead of "it seems to work."

Usage:
    python evals/run_evals.py
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.orchestrator import Orchestrator  # noqa: E402

EVAL_FILE = os.path.join(os.path.dirname(__file__), "eval_queries.json")


def coverage_score(report_text: str, expected_topics: list[str]) -> float:
    """Naive but transparent: fraction of expected topic keywords that
    appear (case-insensitively) somewhere in the final report."""
    text = report_text.lower()
    hits = sum(1 for topic in expected_topics if topic.lower() in text)
    return hits / len(expected_topics) if expected_topics else 1.0


def main():
    with open(EVAL_FILE) as f:
        eval_set = json.load(f)

    orchestrator = Orchestrator()
    results = []

    print(f"Running {len(eval_set)} eval queries...\n")
    for case in eval_set:
        t0 = time.time()
        result = orchestrator.run(case["query"], verbose=False)
        elapsed = time.time() - t0

        score = coverage_score(result.final_report_markdown, case["expected_topics"])
        results.append({
            "id": case["id"],
            "query": case["query"],
            "topic_coverage": round(score, 2),
            "latency_sec": round(elapsed, 2),
            "cost_usd": round(result.estimated_cost_usd, 4),
            "critic_confidence": result.critic_reports[-1].confidence,
        })
        print(f"[{case['id']}] coverage={score:.2f} latency={elapsed:.2f}s "
              f"cost=${result.estimated_cost_usd:.4f} confidence={result.critic_reports[-1].confidence}")

    avg_coverage = sum(r["topic_coverage"] for r in results) / len(results)
    avg_latency = sum(r["latency_sec"] for r in results) / len(results)
    total_cost = sum(r["cost_usd"] for r in results)

    print("\n" + "=" * 50)
    print("EVAL SUMMARY")
    print("=" * 50)
    print(f"Avg topic coverage : {avg_coverage:.2%}")
    print(f"Avg latency        : {avg_latency:.2f}s")
    print(f"Total cost         : ${total_cost:.4f}")

    out_path = os.path.join(os.path.dirname(__file__), "eval_results.json")
    with open(out_path, "w") as f:
        json.dump({
            "results": results,
            "avg_coverage": avg_coverage,
            "avg_latency_sec": avg_latency,
            "total_cost_usd": total_cost,
        }, f, indent=2)
    print(f"\nFull results saved to {out_path}")


if __name__ == "__main__":
    main()
