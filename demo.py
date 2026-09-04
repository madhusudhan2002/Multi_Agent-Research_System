#!/usr/bin/env python3
"""
demo.py — run the full multi-agent research pipeline end-to-end.

Usage:
    python demo.py
    python demo.py --query "What is the current state of solid-state batteries?"
    python demo.py --query "..." --quiet     # suppress step-by-step trace

Runs with ZERO API keys configured (uses mock LLM + mock search so you can
see the full pipeline structure and output format immediately). Add
ANTHROPIC_API_KEY (and optionally TAVILY_API_KEY) to a .env file for real,
live-researched reports.
"""
import argparse
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.orchestrator import Orchestrator  # noqa: E402
from src.config import settings  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Run the multi-agent research pipeline.")
    parser.add_argument(
        "--query",
        default="What is the current state of autonomous AI agents in 2026, "
        "and what are the biggest risks to adopting them in production?",
        help="Research question to investigate.",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress step-by-step agent trace.")
    parser.add_argument(
        "--out", default="output/report.md", help="Where to save the final Markdown report."
    )
    args = parser.parse_args()

    print("=" * 70)
    print("MULTI-AGENT RESEARCH & REPORT GENERATOR")
    print("=" * 70)
    print(f"Query   : {args.query}")
    print(f"LLM mode: {'LIVE (Anthropic API)' if settings.LIVE_LLM else 'MOCK (no API key set)'}")
    print(f"Search  : {'LIVE (Tavily)' if settings.LIVE_SEARCH else 'MOCK (no API key set)'}")
    print("=" * 70)

    orchestrator = Orchestrator()
    result = orchestrator.run(args.query, verbose=not args.quiet)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(f"<!-- Generated {datetime.now(datetime.now().astimezone().tzinfo).isoformat()} -->\n\n")
        f.write(result.final_report_markdown)

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print(f"Sub-tasks researched : {len(result.subtasks)}")
    print(f"Critic review rounds : {len(result.critic_reports)}")
    print(f"Final confidence     : {result.critic_reports[-1].confidence}")
    print(f"Token usage          : {result.total_input_tokens:,} in / {result.total_output_tokens:,} out")
    print(f"Estimated cost       : ${result.estimated_cost_usd:.4f}")
    print(f"Report saved to      : {args.out}")
    print("=" * 70)
    print("\n--- FINAL REPORT PREVIEW ---\n")
    print(result.final_report_markdown[:1500])
    if len(result.final_report_markdown) > 1500:
        print(f"\n...({len(result.final_report_markdown) - 1500} more characters — see {args.out})")


if __name__ == "__main__":
    main()
