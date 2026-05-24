#!/usr/bin/env python3
"""
Transform the enriched dataset into visualization-ready JSON.

Steps:
  1. Load data/processed/dataset.csv
  2. Build 10-feature matrix:
       - Drop `loudness` (r=+0.75 with energy; collinear)
       - Drop `timeSignature` (92% in 4/4; outliers at 0/1 dominate UMAP)
       - Drop `mode` (binary; dominated x-axis after scaling)
       - Replace `key` (linear 0–11) with key_sin/key_cos (circular encoding)
  3. Normalize with StandardScaler
  4. Run UMAP → 2D coordinates (x, y)
  5. Run HDBSCAN on the normalized 10D space → track cluster labels
  6. Build artist profiles (unweighted mean per artist, 3+ tracks only)
     and cluster artists → artist_cluster label per track
  7. Compute log-scaled bubble size from play count
  8. Output data/processed/viz_data.json

The JSON is consumed directly by the web app (no backend required).
"""

import csv
import json
import math
import logging
import argparse
from collections import defaultdict, Counter
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import umap
import hdbscan

DATA_PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"

# Continuous perceptual features:
#   loudness   dropped — r=+0.75 with energy (collinear)
#   timeSignature dropped — 92% in 4/4; outliers at 0/1 dominate UMAP
#   mode       dropped — binary feature dominated x-axis after scaling
#   key        handled separately via sin/cos circular encoding
AUDIO_FEATURE_COLS = [
    "acousticness", "danceability", "energy", "instrumentalness",
    "liveness", "speechiness", "tempo", "valence",
]

UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.1
UMAP_RANDOM_STATE = 42

TSNE_PERPLEXITY = 30
TSNE_N_ITER = 1000
TSNE_RANDOM_STATE = 42

DISPLAY_FEATURES = [
    "acousticness", "danceability", "energy", "instrumentalness",
    "liveness", "speechiness", "valence", "tempo",
]

HDBSCAN_MIN_CLUSTER_SIZE = 10
HDBSCAN_MIN_SAMPLES = 5

ARTIST_MIN_TRACKS = 3   # artists below this don't anchor clusters
ARTIST_N_CLUSTERS = 6   # KMeans k — produces genre-feel groupings


def load_dataset(path: Path) -> tuple[list[dict], np.ndarray]:
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"Dataset is empty: {path}")

    feature_matrix = []
    for r in rows:
        feats = [float(r[col]) for col in AUDIO_FEATURE_COLS]
        key = float(r["key"])
        feats.append(math.sin(2 * math.pi * key / 12))
        feats.append(math.cos(2 * math.pi * key / 12))
        feature_matrix.append(feats)

    return rows, np.array(feature_matrix)


def normalize(matrix: np.ndarray) -> np.ndarray:
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


def run_tsne(
    normalized: np.ndarray,
    perplexity: int = TSNE_PERPLEXITY,
    n_iter: int = TSNE_N_ITER,
) -> np.ndarray:
    reducer = TSNE(
        n_components=2,
        perplexity=perplexity,
        max_iter=n_iter,
        random_state=TSNE_RANDOM_STATE,
        metric="euclidean",
        init="pca",
    )
    logging.info(
        f"Running t-SNE (perplexity={perplexity}, n_iter={n_iter}) "
        f"on {normalized.shape[0]:,} tracks..."
    )
    coords = reducer.fit_transform(normalized)
    logging.info("t-SNE done.")
    return coords


def run_hdbscan(
    normalized: np.ndarray,
    min_cluster_size: int = HDBSCAN_MIN_CLUSTER_SIZE,
) -> np.ndarray:
    # Cluster in original 10D normalized space, not in 2D UMAP space.
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
    logging.info(f"HDBSCAN done: {n_clusters} clusters, {n_noise:,} noise points")
    return labels


def build_artist_profiles(
    rows: list[dict],
    normalized: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, list[int]]]:
    """Unweighted mean feature vector per artist in normalized space."""
    by_artist: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        by_artist[row["artist"]].append(i)

    profiles = {
        artist: normalized[idxs].mean(axis=0)
        for artist, idxs in by_artist.items()
    }
    return profiles, dict(by_artist)


def cluster_artists(
    profiles: dict[str, np.ndarray],
    by_artist: dict[str, list[int]],
    min_tracks: int = ARTIST_MIN_TRACKS,
    n_clusters: int = ARTIST_N_CLUSTERS,
) -> dict[str, int]:
    """
    KMeans on mean artist profiles.
    Only artists with >= min_tracks anchor the clustering.
    Artists with fewer tracks are assigned to the nearest centroid afterward.
    KMeans is used over HDBSCAN here because the artist feature space has
    two broad lobes (hip-hop vs indie) without tight sub-cluster density —
    HDBSCAN collapses it to 2 groups; KMeans enforces the requested k.
    """
    all_artists = list(profiles.keys())
    qualified   = [a for a in all_artists if len(by_artist[a]) >= min_tracks]
    unqualified = [a for a in all_artists if len(by_artist[a]) < min_tracks]

    q_matrix = np.array([profiles[a] for a in qualified])
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
    q_labels = km.fit_predict(q_matrix)

    logging.info(
        f"Artist KMeans (k={n_clusters}): fit on {len(qualified)} qualified artists "
        f"(≥{min_tracks} tracks), assigning {len(unqualified)} tail artists to nearest centroid"
    )

    artist_cluster: dict[str, int] = {}
    for artist, label in zip(qualified, q_labels):
        artist_cluster[artist] = int(label)

    # Assign tail artists (< min_tracks) to nearest KMeans centroid
    centroids = km.cluster_centers_
    for artist in unqualified:
        vec = profiles[artist]
        nearest = int(np.argmin([np.linalg.norm(vec - c) for c in centroids]))
        artist_cluster[artist] = nearest

    # Log top 5 artists per cluster by track count
    for cid in range(n_clusters):
        members = sorted(
            [(a, len(by_artist[a])) for a, c in artist_cluster.items() if c == cid],
            key=lambda x: -x[1],
        )
        top5 = ", ".join(f"{a}({n})" for a, n in members[:5])
        tail = "…" if len(members) > 5 else ""
        logging.info(f"  Cluster {cid} ({len(members)} artists): {top5}{tail}")

    return artist_cluster


def build_viz_records(
    rows: list[dict],
    coords: np.ndarray,
    cluster_labels: np.ndarray,
    artist_cluster: dict[str, int],
) -> list[dict]:
    records = []
    for i, row in enumerate(rows):
        play_count = int(row["play_count"])
        rec = {
            "artist":         row["artist"],
            "title":          row["title"],
            "play_count":     play_count,
            "spotify_id":     row["spotify_id"],
            "x":              round(float(coords[i, 0]), 4),
            "y":              round(float(coords[i, 1]), 4),
            "cluster":        int(cluster_labels[i]),
            "artist_cluster": artist_cluster.get(row["artist"], 0),
            "log_size":       round(math.log1p(play_count), 4),
        }
        for f in DISPLAY_FEATURES:
            if f in row:
                rec[f] = round(float(row[f]), 4)
        # Year fields — pass through as int if present, omit if not
        for f in ("first_year", "last_year", "peak_year"):
            val = row.get(f)
            if val is not None and val != "":
                rec[f] = int(val)
        records.append(rec)
    return records


def save_viz_json(records: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False)
    logging.info(f"Saved {len(records):,} records → {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build visualization data from enriched dataset")
    parser.add_argument("--input", type=Path, default=DATA_PROCESSED_DIR / "dataset.csv")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--method", choices=["umap", "tsne"], default="umap",
                        help="Dimensionality reduction method (default: umap)")
    parser.add_argument("--umap-neighbors", type=int, default=UMAP_N_NEIGHBORS, metavar="N")
    parser.add_argument("--umap-min-dist", type=float, default=UMAP_MIN_DIST, metavar="F")
    parser.add_argument("--tsne-perplexity", type=int, default=TSNE_PERPLEXITY, metavar="N")
    parser.add_argument("--tsne-n-iter", type=int, default=TSNE_N_ITER, metavar="N")
    parser.add_argument("--hdbscan-min-cluster", type=int, default=HDBSCAN_MIN_CLUSTER_SIZE, metavar="N")
    parser.add_argument("--artist-clusters", type=int, default=ARTIST_N_CLUSTERS, metavar="K",
                        help=f"Number of KMeans artist clusters (default: {ARTIST_N_CLUSTERS})")
    args = parser.parse_args()
    if args.output is None:
        args.output = DATA_PROCESSED_DIR / (
            "viz_data_tsne.json" if args.method == "tsne" else "viz_data.json"
        )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    rows, feature_matrix = load_dataset(args.input)
    logging.info(f"Loaded {len(rows):,} tracks from {args.input}")

    normalized = normalize(feature_matrix)
    if args.method == "tsne":
        coords = run_tsne(normalized, perplexity=args.tsne_perplexity, n_iter=args.tsne_n_iter)
    else:
        coords = run_umap(normalized, n_neighbors=args.umap_neighbors, min_dist=args.umap_min_dist)
    cluster_labels = run_hdbscan(normalized, min_cluster_size=args.hdbscan_min_cluster)

    logging.info("Building artist profiles and clustering...")
    profiles, by_artist = build_artist_profiles(rows, normalized)
    artist_cluster = cluster_artists(profiles, by_artist, n_clusters=args.artist_clusters)

    records = build_viz_records(rows, coords, cluster_labels, artist_cluster)
    save_viz_json(records, args.output)

    logging.info("Processing complete.")


if __name__ == "__main__":
    main()
