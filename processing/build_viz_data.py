#!/usr/bin/env python3
"""
Transform the enriched dataset into visualization-ready JSON.

Steps:
  1. Load data/processed/dataset.csv
  2. Normalize audio features with StandardScaler
  3. Run UMAP → 2D coordinates (x, y)
  4. Run HDBSCAN on the original normalized 12D space → cluster labels
  5. Compute log-scaled bubble size from play count
  6. Output data/processed/viz_data.json

The JSON is consumed directly by the web app (no backend required).
"""

import csv
import json
import math
import logging
import argparse
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler
import umap
import hdbscan

DATA_PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"

AUDIO_FEATURE_COLS = [
    "acousticness", "danceability", "energy", "instrumentalness",
    "key", "liveness", "loudness", "mode",
    "speechiness", "tempo", "timeSignature", "valence",
]

# UMAP defaults — tuned for ~1k–5k points
UMAP_N_NEIGHBORS = 15       # local neighborhood size; lower = more local structure
UMAP_MIN_DIST = 0.1         # minimum spread of points in 2D
UMAP_RANDOM_STATE = 42

# HDBSCAN defaults
HDBSCAN_MIN_CLUSTER_SIZE = 15   # minimum tracks to form a cluster
HDBSCAN_MIN_SAMPLES = 5         # controls noise sensitivity


def load_dataset(path: Path) -> tuple[list[dict], np.ndarray]:
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"Dataset is empty: {path}")
    feature_matrix = np.array(
        [[float(r[col]) for col in AUDIO_FEATURE_COLS] for r in rows]
    )
    return rows, feature_matrix


def normalize(matrix: np.ndarray) -> np.ndarray:
    # StandardScaler: zero mean, unit variance per feature.
    # Better than MinMax here because loudness (-60–0 dB) and tempo (BPM)
    # are on very different scales from the 0–1 features.
    return StandardScaler().fit_transform(matrix)


def run_umap(
    normalized: np.ndarray,
    n_neighbors: int = UMAP_N_NEIGHBORS,
    min_dist: float = UMAP_MIN_DIST,
) -> np.ndarray:
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=UMAP_RANDOM_STATE,
        metric="euclidean",
    )
    logging.info(
        f"Running UMAP (n_neighbors={n_neighbors}, min_dist={min_dist}) "
        f"on {normalized.shape[0]:,} tracks..."
    )
    coords = reducer.fit_transform(normalized)
    logging.info("UMAP done.")
    return coords


def run_hdbscan(
    normalized: np.ndarray,
    min_cluster_size: int = HDBSCAN_MIN_CLUSTER_SIZE,
) -> np.ndarray:
    # Cluster in original 12D normalized space, NOT in 2D UMAP space.
    # 2D coords are for display only; similarity lives in the full feature space.
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=HDBSCAN_MIN_SAMPLES,
        metric="euclidean",
    )
    logging.info(
        f"Running HDBSCAN (min_cluster_size={min_cluster_size}) "
        f"on {normalized.shape[0]:,} tracks..."
    )
    labels = clusterer.fit_predict(normalized)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int(np.sum(labels == -1))
    logging.info(f"HDBSCAN done: {n_clusters} clusters, {n_noise:,} noise points (label -1)")
    return labels


def build_viz_records(
    rows: list[dict],
    coords: np.ndarray,
    cluster_labels: np.ndarray,
) -> list[dict]:
    records = []
    for i, row in enumerate(rows):
        play_count = int(row["play_count"])
        records.append({
            "artist": row["artist"],
            "title": row["title"],
            "play_count": play_count,
            "spotify_id": row["spotify_id"],
            "x": round(float(coords[i, 0]), 4),
            "y": round(float(coords[i, 1]), 4),
            "cluster": int(cluster_labels[i]),   # -1 = noise/unclustered
            "log_size": round(math.log1p(play_count), 4),
        })
    return records


def save_viz_json(records: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False)
    logging.info(f"Saved {len(records):,} records → {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build visualization data from enriched dataset")
    parser.add_argument("--input", type=Path, default=DATA_PROCESSED_DIR / "dataset.csv")
    parser.add_argument("--output", type=Path, default=DATA_PROCESSED_DIR / "viz_data.json")
    parser.add_argument(
        "--umap-neighbors", type=int, default=UMAP_N_NEIGHBORS, metavar="N",
        help=f"UMAP n_neighbors (default: {UMAP_N_NEIGHBORS})",
    )
    parser.add_argument(
        "--umap-min-dist", type=float, default=UMAP_MIN_DIST, metavar="F",
        help=f"UMAP min_dist (default: {UMAP_MIN_DIST})",
    )
    parser.add_argument(
        "--hdbscan-min-cluster", type=int, default=HDBSCAN_MIN_CLUSTER_SIZE, metavar="N",
        help=f"HDBSCAN min_cluster_size (default: {HDBSCAN_MIN_CLUSTER_SIZE})",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    rows, feature_matrix = load_dataset(args.input)
    logging.info(f"Loaded {len(rows):,} tracks from {args.input}")

    normalized = normalize(feature_matrix)
    coords = run_umap(normalized, n_neighbors=args.umap_neighbors, min_dist=args.umap_min_dist)
    cluster_labels = run_hdbscan(normalized, min_cluster_size=args.hdbscan_min_cluster)

    records = build_viz_records(rows, coords, cluster_labels)
    save_viz_json(records, args.output)

    logging.info("Processing complete. Run the web app to view the visualization.")


if __name__ == "__main__":
    main()
