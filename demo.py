#!/usr/bin/env python3
"""
demo.py — run the full multi-agent research pipeline end-to-end.

Usage:
    python demo.py
    python demo.py --query "What is the current state of solid-state batteries?"
    python demo.py --query "..." --quiet

The project uses:
    - Groq for the LLM
    - DuckDuckGo for web search
    - Mock LLM/search only as fallbacks

Configure GROQ_API_KEY in your .env file for live LLM responses.

The search tool uses DuckDuckGo and does not require a Tavily API key.
"""

import argparse
import os
import sys
from datetime import datetime

sys.path.insert(
    0,
    os.path.dirname(os.path.abspath(__file__))
)

from src.orchestrator import Orchestrator
from src.config import settings


def main():

    parser = argparse.ArgumentParser(
        description="Run the multi-agent research pipeline."
    )

    parser.add_argument(
        "--query",
        default=(
            "What is the current state of autonomous AI agents in 2026, "
            "and what are the biggest risks to adopting them in production?"
        ),
        help="Research question to investigate.",
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress step-by-step agent trace."
    )

    parser.add_argument(
        "--out",
        default="output/report.md",
        help="Where to save the final Markdown report."
    )

    args = parser.parse_args()

    # ---------------------------------------------------------
    # HEADER
    # ---------------------------------------------------------

    print("=" * 70)
    print("MULTI-AGENT RESEARCH & REPORT GENERATOR")
    print("=" * 70)

    print(f"Query   : {args.query}")

    # Groq LLM status
    print(
        "LLM mode: "
        f"{'LIVE (Groq API)' if settings.LIVE_LLM else 'MOCK (no API key set)'}"
    )

    # Current search implementation
    print("Search  : LIVE (DuckDuckGo)")

    print("=" * 70)

    # ---------------------------------------------------------
    # RUN PIPELINE
    # ---------------------------------------------------------

    orchestrator = Orchestrator()

    result = orchestrator.run(
        args.query,
        verbose=not args.quiet
    )

    # ---------------------------------------------------------
    # SAVE REPORT
    # ---------------------------------------------------------

    output_directory = os.path.dirname(args.out)

    if output_directory:
        os.makedirs(
            output_directory,
            exist_ok=True
        )

    with open(
        args.out,
        "w",
        encoding="utf-8"
    ) as f:

        timestamp = datetime.now(
            datetime.now().astimezone().tzinfo
        ).isoformat()

        f.write(
            f"<!-- Generated {timestamp} -->\n\n"
        )

        f.write(
            result.final_report_markdown
        )

    # ---------------------------------------------------------
    # PIPELINE SUMMARY
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)

    print(
        f"Sub-tasks researched : "
        f"{len(result.subtasks)}"
    )

    print(
        f"Critic review rounds : "
        f"{len(result.critic_reports)}"
    )

    if result.critic_reports:
        print(
            f"Final confidence     : "
            f"{result.critic_reports[-1].confidence}"
        )
    else:
        print(
            "Final confidence     : N/A"
        )

    print(
        f"Token usage          : "
        f"{result.total_input_tokens:,} in / "
        f"{result.total_output_tokens:,} out"
    )

    print(
        f"Estimated cost       : "
        f"${result.estimated_cost_usd:.4f}"
    )

    print(
        f"Report saved to      : "
        f"{args.out}"
    )

    print("=" * 70)

    # ---------------------------------------------------------
    # REPORT PREVIEW
    # ---------------------------------------------------------

    print(
        "\n--- FINAL REPORT PREVIEW ---\n"
    )

    preview_length = 1500

    print(
        result.final_report_markdown[
            :preview_length
        ]
    )

    if len(result.final_report_markdown) > preview_length:

        remaining = (
            len(result.final_report_markdown)
            - preview_length
        )

        print(
            f"\n...({remaining} more characters "
            f"— see {args.out})"
        )


if __name__ == "__main__":
    main()