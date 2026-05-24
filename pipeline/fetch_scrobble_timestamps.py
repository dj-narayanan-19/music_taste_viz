#!/usr/bin/env python3
"""
Fetch every scrobble timestamp from Last.fm and compute per-track temporal stats.

First run pages through the full user history — ~1,285 pages / ~5 min for 257k
scrobbles. Results are cached to data/raw/scrobble_cache.csv. Re-runs only fetch
scrobbles newer than the most recent cached timestamp (usually a few seconds).

Outputs:
  data/raw/scrobble_cache.csv    — raw (artist, title, timestamp) for every scrobble
  data/raw/track_timestamps.csv — per-track: first_year, last_year, peak_year
"""

import os
import csv
import time
import logging
import argparse
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime, timezone

import requests
from dotenv import load_dotenv

load_dotenv()

LASTFM_API_BASE     = "https://ws.audioscrobbler.com/2.0/"
PAGE_LIMIT          = 200
DATA_RAW_DIR        = Path(__file__).parent.parent / "data" / "raw"
SCROBBLE_CACHE_PATH = DATA_RAW_DIR / "scrobble_cache.csv"
TIMESTAMPS_OUT_PATH = DATA_RAW_DIR / "track_timestamps.csv"


def load_cache(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def save_cache(scrobbles: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["artist", "title", "timestamp"])
        writer.writeheader()
        writer.writerows(scrobbles)


def fetch_scrobbles(username: str, api_key: str, from_ts: int = 0) -> list[dict]:
    """
    Page through user.getRecentTracks and return scrobbles newer than from_ts.
    Each item: {"artist": str, "title": str, "timestamp": int}.
    """
    results = []
    page = 1
    total_pages = None

    while total_pages is None or page <= total_pages:
        params = {
            "method":   "user.getRecentTracks",
            "user":     username,
            "api_key":  api_key,
            "format":   "json",
            "limit":    PAGE_LIMIT,
            "page":     page,
            "extended": 0,
        }
        if from_ts:
            params["from"] = from_ts + 1  # exclusive lower bound

        for attempt in range(3):
            try:
                resp = requests.get(LASTFM_API_BASE, params=params, timeout=30)
                if resp.status_code < 500:
                    break
            except requests.RequestException:
                pass
            time.sleep(2 ** attempt)
        resp.raise_for_status()
        data = resp.json()

        if "error" in data:
            raise RuntimeError(f"Last.fm API error {data['error']}: {data['message']}")

        rt   = data["recenttracks"]
        attr = rt["@attr"]

        if total_pages is None:
            total_pages = int(attr["totalPages"])
            total       = int(attr["total"])
            label = f"since {datetime.fromtimestamp(from_ts, tz=timezone.utc).strftime('%Y-%m-%d')}" if from_ts else "full history"
            logging.info(f"Fetching {total:,} scrobble(s) across {total_pages} page(s) ({label})")

        tracks = rt["track"]
        if isinstance(tracks, dict):
            tracks = [tracks]

        for t in tracks:
            # Skip "now playing" entries — they have no date field
            if "@attr" in t and t["@attr"].get("nowplaying") == "true":
                continue
            if "date" not in t:
                continue
            results.append({
                "artist":    t["artist"]["#text"],
                "title":     t["name"],
                "timestamp": int(t["date"]["uts"]),
            })

        if page % 100 == 0 or page == total_pages:
            logging.info(f"  Page {page}/{total_pages} — {len(results):,} scrobbles fetched")

        page += 1
        time.sleep(0.25)

    return results


def compute_track_timestamps(scrobbles: list[dict]) -> list[dict]:
    """Group scrobbles by (artist, title) and derive first_year, last_year, peak_year."""
    by_track: dict[tuple, list[int]] = defaultdict(list)
    for s in scrobbles:
        key = (s["artist"].lower(), s["title"].lower())
        by_track[key].append(int(s["timestamp"]))

    rows = []
    for (artist_lower, title_lower), timestamps in by_track.items():
        years     = [datetime.fromtimestamp(ts, tz=timezone.utc).year for ts in timestamps]
        peak_year = Counter(years).most_common(1)[0][0]
        rows.append({
            "artist_lower": artist_lower,
            "title_lower":  title_lower,
            "first_year":   min(years),
            "last_year":    max(years),
            "peak_year":    peak_year,
        })
    return rows


def save_track_timestamps(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["artist_lower", "title_lower", "first_year", "last_year", "peak_year"],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch scrobble timestamps from Last.fm and compute per-track year stats"
    )
    parser.add_argument("username", help="Last.fm username")
    parser.add_argument(
        "--full", action="store_true",
        help="Ignore existing cache and re-fetch the complete history",
    )
    args = parser.parse_args()

    api_key = os.environ.get("LASTFM_API_KEY")
    if not api_key:
        parser.error("LASTFM_API_KEY not set in environment / .env file")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    cached  = [] if args.full else load_cache(SCROBBLE_CACHE_PATH)
    from_ts = max((int(s["timestamp"]) for s in cached), default=0) if cached else 0

    if cached:
        ts_str = datetime.fromtimestamp(from_ts, tz=timezone.utc).strftime("%Y-%m-%d")
        logging.info(f"Loaded {len(cached):,} cached scrobbles — fetching new ones since {ts_str}...")
    else:
        logging.info("No cache found. Fetching full history (~5 min for 257k scrobbles)...")

    new_scrobbles = fetch_scrobbles(args.username, api_key, from_ts=from_ts)

    all_scrobbles = cached + new_scrobbles
    logging.info(f"Total: {len(all_scrobbles):,} scrobbles ({len(new_scrobbles):,} new)")

    save_cache(all_scrobbles, SCROBBLE_CACHE_PATH)
    logging.info(f"Cache saved → {SCROBBLE_CACHE_PATH}")

    rows = compute_track_timestamps(all_scrobbles)
    save_track_timestamps(rows, TIMESTAMPS_OUT_PATH)
    logging.info(f"Saved {len(rows):,} track timestamp records → {TIMESTAMPS_OUT_PATH}")


if __name__ == "__main__":
    main()
