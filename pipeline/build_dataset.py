#!/usr/bin/env python3
"""
Join track_counts.csv (play counts from Last.fm) with audio_features.csv
(audio features from Soundcharts) into a single clean dataset.

Input:  data/raw/track_counts.csv  + data/raw/audio_features.csv
Output: data/processed/dataset.csv
Columns: artist, title, play_count, spotify_id, acousticness, danceability,
         energy, instrumentalness, key, liveness, loudness, mode,
         speechiness, tempo, timeSignature, valence
"""

import csv
import logging
import argparse
from pathlib import Path

DATA_RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
DATA_PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"

AUDIO_FEATURE_COLS = [
    "acousticness", "danceability", "energy", "instrumentalness",
    "key", "liveness", "loudness", "mode",
    "speechiness", "tempo", "timeSignature", "valence",
]


def load_csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_timestamps(path: Path) -> dict:
    """Load track_timestamps.csv into a dict keyed by (artist_lower, title_lower)."""
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    index = {(r["artist_lower"], r["title_lower"]): r for r in rows}
    logging.info(f"Loaded {len(index):,} track timestamp entries from {path.name}")
    return index


def join_datasets(
    track_counts: list[dict],
    audio_features: list[dict],
    timestamps: dict,
) -> tuple[list[dict], int]:
    """
    Inner-join: iterate audio_features (the smaller set) and look up play
    counts from track_counts. Tracks without audio features are excluded.
    Timestamps are left-joined — missing entries leave year fields empty.
    Returns (joined rows, number of audio_features rows skipped due to missing
    feature values or no matching play-count entry).
    """
    counts_index = {
        (r["artist"].lower(), r["title"].lower()): r
        for r in track_counts
    }

    joined = []
    skipped = 0
    for feat in audio_features:
        key = (feat["artist"].lower(), feat["title"].lower())
        counts = counts_index.get(key)
        if counts is None:
            logging.warning(f"No play-count entry for {feat['artist']} — {feat['title']}; skipping")
            skipped += 1
            continue
        feature_vals = {}
        skip = False
        for col in AUDIO_FEATURE_COLS:
            raw = feat.get(col)
            if raw is None or raw == "":
                logging.warning(f"Missing feature '{col}' for {feat['artist']} — {feat['title']}; skipping track")
                skip = True
                break
            feature_vals[col] = float(raw)
        if skip:
            skipped += 1
            continue
        ts = timestamps.get(key, {})
        joined.append({
            "artist":     feat["artist"],
            "title":      feat["title"],
            "play_count": int(counts["play_count"]),
            "spotify_id": feat["spotify_id"],
            **feature_vals,
            "first_year": ts.get("first_year", ""),
            "last_year":  ts.get("last_year", ""),
            "peak_year":  ts.get("peak_year", ""),
        })

    return joined, skipped


def save_dataset(rows: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        ["artist", "title", "play_count", "spotify_id"]
        + AUDIO_FEATURE_COLS
        + ["first_year", "last_year", "peak_year"]
    )
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Join Last.fm and Soundcharts data into clean dataset")
    parser.add_argument("--counts", type=Path, default=DATA_RAW_DIR / "track_counts.csv")
    parser.add_argument("--features", type=Path, default=DATA_RAW_DIR / "audio_features.csv")
    parser.add_argument("--output", type=Path, default=DATA_PROCESSED_DIR / "dataset.csv")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S")

    track_counts = load_csv(args.counts)
    audio_features = load_csv(args.features)
    logging.info(f"Loaded {len(track_counts):,} tracks (Last.fm) and {len(audio_features):,} tracks (Soundcharts)")

    timestamps = load_timestamps(DATA_RAW_DIR / "track_timestamps.csv")
    joined, skipped = join_datasets(track_counts, audio_features, timestamps)
    logging.info(f"Joined: {len(joined):,} tracks, {skipped:,} skipped (incomplete features)")

    save_dataset(joined, args.output)
    logging.info(f"Saved → {args.output}")


if __name__ == "__main__":
    main()
