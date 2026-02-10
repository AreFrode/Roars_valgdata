#!/usr/bin/env python3
"""Small CLI wrapper to run basic analysis using the new package."""

import argparse
import json
from pathlib import Path
from valgdata import load_csv, summarize


def main():
    default = Path.cwd() / "Valgresultat 2025👍 (Svar) - Skjemasvar 1.csv"
    p = argparse.ArgumentParser(
        description="Run basic analysis on the CSV included in the repo"
    )
    p.add_argument("-i", "--input", default=str(default), help="Path to CSV input")
    p.add_argument("-o", "--out", help="Optional output JSON file")
    args = p.parse_args()

    df = load_csv(args.input)
    summary = summarize(df)

    if args.out:
        Path(args.out).write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    else:
        print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
