#!/usr/bin/env python3
"""Run an offline detector replay against a CSV of completed 5m candles."""
import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.replay import SUPPORTED_STRATEGIES, load_strategy_universes, replay_dataframe


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="CSV with symbol,timestamp,open,high,low,close,volume")
    parser.add_argument("--output", required=True, help="Output CSV path for raw detector events")
    parser.add_argument("--universe-dir", default="config", help="Directory containing live symbol-universe files")
    parser.add_argument(
        "--strategies", default="all",
        help="Comma-separated strategy names, or 'all' (default)",
    )
    parser.add_argument(
        "--thresholds-json", help="Optional JSON file with threshold overrides",
    )
    parser.add_argument(
        "--include-rejections", action="store_true",
        help="Include one row for each rejected detector evaluation (can be large)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    selected = SUPPORTED_STRATEGIES if args.strategies.strip().lower() == "all" else tuple(
        name.strip().upper() for name in args.strategies.split(",") if name.strip()
    )
    data = pd.read_csv(input_path)
    overrides = json.loads(Path(args.thresholds_json).read_text(encoding="utf-8")) if args.thresholds_json else None
    universes = load_strategy_universes(args.universe_dir)
    events, summary = replay_dataframe(
        data, strategies=selected, include_rejections=args.include_rejections,
        threshold_overrides=overrides, strategy_symbols=universes,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    events.to_csv(output_path, index=False)
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({
        "events_csv": str(output_path),
        "summary_json": str(summary_path),
        "summary": summary,
    }, indent=2))


if __name__ == "__main__":
    main()
