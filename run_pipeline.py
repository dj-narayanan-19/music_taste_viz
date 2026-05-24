#!/usr/bin/env python3
"""
End-to-end pipeline runner.

Usage:
  python run_pipeline.py <lastfm_username> [options]

Steps (in order):
  1. pipeline/fetch_lastfm.py       — pull top tracks + play counts
  2. pipeline/fetch_spotify.py      — resolve Spotify IDs + fetch Soundcharts features
  3. pipeline/build_dataset.py      — inner-join, clean → data/processed/dataset.csv
  4. processing/build_viz_data.py   — UMAP + HDBSCAN → data/processed/viz_data.json
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent

STEPS = [
    {
        "label": "Step 1/5  fetch_lastfm",
        "script": ROOT / "pipeline" / "fetch_lastfm.py",
        "extra_args": lambda args: [args.username, "--min-plays", str(args.min_plays)],
        "skip_if": lambda args: False,
    },
    {
        "label": "Step 2/5  fetch_scrobble_timestamps",
        "script": ROOT / "pipeline" / "fetch_scrobble_timestamps.py",
        "extra_args": lambda args: [args.username],
        "skip_if": lambda args: args.skip_timestamps,
    },
    {
        "label": "Step 3/5  fetch_spotify",
        "script": ROOT / "pipeline" / "fetch_spotify.py",
        "extra_args": lambda args: (
            ["--features-only"] if args.features_only else []
        ) + ["--limit", str(args.limit)],
        "skip_if": lambda args: False,
    },
    {
        "label": "Step 4/5  build_dataset",
        "script": ROOT / "pipeline" / "build_dataset.py",
        "extra_args": lambda args: [],
        "skip_if": lambda args: False,
    },
    {
        "label": "Step 5/5  build_viz_data",
        "script": ROOT / "processing" / "build_viz_data.py",
        "extra_args": lambda args: [],
        "skip_if": lambda args: False,
    },
]

SEPARATOR = "─" * 60


def run_step(label: str, script: Path, extra_args: list[str]) -> None:
    print(f"\n{SEPARATOR}")
    print(f"  {label}")
    print(SEPARATOR)
    cmd = [sys.executable, str(script)] + extra_args
    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode != 0:
        print(f"\n  ERROR: {label} exited with code {result.returncode}. Aborting.")
        sys.exit(result.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the full music-taste-viz pipeline for a Last.fm user"
    )
    parser.add_argument("username", help="Last.fm username")
    parser.add_argument(
        "--min-plays", type=int, default=10, metavar="N",
        help="Minimum play count to include a track (default: 10)",
    )
    parser.add_argument(
        "--features-only", action="store_true",
        help="Skip Spotify ID search in fetch_spotify; use cached IDs only",
    )
    parser.add_argument(
        "--limit", type=int, default=1000, metavar="N",
        help="Max tracks to process for Soundcharts features per run (default: 1000)",
    )
    parser.add_argument(
        "--skip-timestamps", action="store_true",
        help="Skip the scrobble timestamp fetch (step 2) — era coloring will be unavailable",
    )
    args = parser.parse_args()

    print(f"\nMusic taste viz pipeline — user: {args.username}")

    for step in STEPS:
        if step["skip_if"](args):
            print(f"\n  Skipping: {step['label']}")
            continue
        run_step(
            label=step["label"],
            script=step["script"],
            extra_args=step["extra_args"](args),
        )

    print(f"\n{SEPARATOR}")
    print("  All steps complete.")
    print(f"  Visualization data → data/processed/viz_data.json")
    print(f"  Open web/index.html (via a local server) to view.")
    print(SEPARATOR + "\n")


if __name__ == "__main__":
    main()
