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


def join_datasets(
    track_counts: list[dict],
    audio_features: list[dict],
) -> tuple[list[dict], int]:
    """
    Inner-join: iterate audio_features (the smaller set) and look up play
    counts from track_counts. Tracks without audio features are excluded.
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
        joined.append({
            "artist": feat["artist"],
            "title": feat["title"],
            "play_count": int(counts["play_count"]),
            "spotify_id": feat["spotify_id"],
            **feature_vals,
        })

    return joined, skipped


def save_dataset(rows: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["artist", "title", "play_count", "spotify_id"] + AUDIO_FEATURE_COLS
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

    joined, skipped = join_datasets(track_counts, audio_features)
    logging.info(f"Joined: {len(joined):,} tracks, {skipped:,} skipped (incomplete features)")

    save_dataset(joined, args.output)
    logging.info(f"Saved → {args.output}")


if __name__ == "__main__":
    main()
