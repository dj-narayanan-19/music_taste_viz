#!/usr/bin/env python3
"""
Fetch all-time top tracks with play counts from Last.fm.

Uses user.getTopTracks(period=overall) rather than iterating raw scrobbles —
results come pre-aggregated and sorted by play count descending, so we can
stop as soon as we hit the min_play_count threshold instead of fetching the
entire scrobble history.

Output: data/raw/track_counts.csv
Columns: artist, title, play_count, mbid
"""

import os
import csv
import time
import argparse
import logging
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

LASTFM_API_BASE = "https://ws.audioscrobbler.com/2.0/"
PAGE_LIMIT = 1000  # max per-page limit allowed by Last.fm
DEFAULT_MIN_PLAYS = 5
DATA_RAW_DIR = Path(__file__).parent.parent / "data" / "raw"


def fetch_top_tracks(
    username: str,
    api_key: str,
    min_play_count: int = DEFAULT_MIN_PLAYS,
) -> list[dict]:
    """
    Page through user.getTopTracks until play count drops below threshold.
    Returns list of dicts with keys: artist, title, play_count, mbid.
    """
    results = []
    page = 1
    total_pages = None

    while total_pages is None or page <= total_pages:
        params = {
            "method": "user.getTopTracks",
            "user": username,
            "api_key": api_key,
            "format": "json",
            "limit": PAGE_LIMIT,
            "page": page,
            "period": "overall",
        }

        # Last.fm occasionally returns transient 500s; retry up to 3 times
        for attempt in range(3):
            resp = requests.get(LASTFM_API_BASE, params=params, timeout=30)
            if resp.status_code < 500:
                break
            wait = 2 ** attempt
            logging.warning(f"Last.fm returned {resp.status_code} on page {page}, retrying in {wait}s...")
            time.sleep(wait)
        resp.raise_for_status()
        data = resp.json()

        if "error" in data:
            raise RuntimeError(f"Last.fm API error {data['error']}: {data['message']}")

        attr = data["toptracks"]["@attr"]
        if total_pages is None:
            total_pages = int(attr["totalPages"])
            total_tracks = int(attr["total"])
            logging.info(
                f"User '{username}' has {total_tracks:,} unique tracks across "
                f"{total_pages} pages. Fetching above {min_play_count}-play threshold..."
            )

        tracks = data["toptracks"]["track"]
        if not tracks:
            break

        # Ensure tracks is always a list (single result comes back as a dict)
        if isinstance(tracks, dict):
            tracks = [tracks]

        for track in tracks:
            play_count = int(track["playcount"])
            if play_count < min_play_count:
                logging.info(
                    f"Hit threshold ({min_play_count} plays) at page {page}/{total_pages}. "
                    f"Collected {len(results):,} tracks."
                )
                return results
            results.append({
                "artist": track["artist"]["name"],
                "title": track["name"],
                "play_count": play_count,
                "mbid": track.get("mbid") or "",
            })

        logging.info(f"Page {page}/{total_pages} — {len(results):,} tracks collected")
        page += 1
        time.sleep(0.25)  # ~4 req/s, well within Last.fm's rate limit

    return results


def save_csv(tracks: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["artist", "title", "play_count", "mbid"])
        writer.writeheader()
        writer.writerows(tracks)
    logging.info(f"Saved {len(tracks):,} tracks → {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch Last.fm top tracks with play counts")
    parser.add_argument("username", help="Last.fm username")
    parser.add_argument(
        "--min-plays",
        type=int,
        default=DEFAULT_MIN_PLAYS,
        metavar="N",
        help=f"Only include tracks with at least N plays (default: {DEFAULT_MIN_PLAYS})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DATA_RAW_DIR / "track_counts.csv",
        metavar="PATH",
        help="Output CSV path (default: data/raw/track_counts.csv)",
    )
    args = parser.parse_args()

    api_key = os.environ.get("LASTFM_API_KEY")
    if not api_key:
        parser.error(
            "LASTFM_API_KEY is not set. "
            "Add it to a .env file or export it as an environment variable.\n"
            "Get a free key at: https://www.last.fm/api/account/create"
        )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    tracks = fetch_top_tracks(args.username, api_key, min_play_count=args.min_plays)
    save_csv(tracks, args.output)


if __name__ == "__main__":
    main()
