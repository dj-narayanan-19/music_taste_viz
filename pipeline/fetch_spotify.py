#!/usr/bin/env python3
"""
Enrich top tracks from data/raw/track_counts.csv with audio features via Soundcharts.

Two-phase approach:
  1. Search Spotify to resolve each (artist, title) → Spotify track ID. Results
     are cached in data/raw/spotify_id_cache.json so re-runs skip already-resolved
     tracks. (Spotify track search is still available; only audio features were
     deprecated in late 2024.)
  2. Fetch audio features from Soundcharts using the Spotify ID as a lookup key.
     Soundcharts returns the same 12-dimension feature set as the original Spotify
     audio-features endpoint.

Only the top --limit tracks by play count are processed (default 1000) to stay
within the Soundcharts free tier (1,000 requests).

Output: data/raw/audio_features.csv
Columns: artist, title, play_count, spotify_id, + 12 audio feature columns

Required env vars (in .env):
  SOUNDCHARTS_APP_ID   — "soundcharts" for sandbox, your app ID for production
  SOUNDCHARTS_API_KEY  — "soundcharts" for sandbox, your API key for production
  SPOTIFY_CLIENT_ID    — only needed for Phase 1 (resolving new Spotify IDs)
  SPOTIFY_CLIENT_SECRET

Soundcharts sandbox credentials are pre-filled in .env.example and work for
testing but are restricted to 2 songs. Register a free production token at
https://developers.soundcharts.com to get real coverage.
"""

import os
import csv
import json
import time
import argparse
import logging
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

DATA_RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
DEFAULT_INPUT  = DATA_RAW_DIR / "track_counts.csv"
DEFAULT_OUTPUT = DATA_RAW_DIR / "audio_features.csv"
ID_CACHE_PATH  = DATA_RAW_DIR / "spotify_id_cache.json"
NOT_FOUND_PATH = DATA_RAW_DIR / "spotify_not_found.txt"

DEFAULT_LIMIT = 1000

AUDIO_FEATURE_COLS = [
    "acousticness", "danceability", "energy", "instrumentalness",
    "key", "liveness", "loudness", "mode",
    "speechiness", "tempo", "timeSignature", "valence",
]

SOUNDCHARTS_BASE  = "https://customer.api.soundcharts.com"
SOUNDCHARTS_CACHE_PATH = DATA_RAW_DIR / "soundcharts_cache.json"


# ── Helpers ────────────────────────────────────────────────────────────────────

def cache_key(artist: str, title: str) -> str:
    return f"{artist.lower()}||{title.lower()}"


def load_cache(path: Path) -> dict:
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_cache(cache: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


# ── Phase 1: resolve Spotify track IDs ────────────────────────────────────────

def make_spotify_client():
    try:
        import spotipy
        from spotipy.oauth2 import SpotifyClientCredentials
    except ImportError:
        return None

    client_id = os.environ.get("SPOTIFY_CLIENT_ID")
    client_secret = os.environ.get("SPOTIFY_CLIENT_SECRET")
    if not client_id or not client_secret:
        return None

    auth = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
    return spotipy.Spotify(auth_manager=auth, retries=5)


def search_track_id(sp, artist: str, title: str) -> str | None:
    safe_title  = title.replace('"', "")
    safe_artist = artist.replace('"', "")
    query = f'track:"{safe_title}" artist:"{safe_artist}"'
    try:
        results = sp.search(q=query, type="track", limit=1)
        items = results["tracks"]["items"]
        if items:
            return items[0]["id"]
    except Exception as exc:
        logging.warning(f"Spotify search failed [{artist} — {title}]: {exc}")
    return None


def resolve_track_ids(sp, tracks: list[dict], cache: dict) -> list[str]:
    """
    Searches Spotify for uncached tracks, updates cache in place.
    Returns list of 'artist — title' strings with no Spotify match.
    """
    uncached = [t for t in tracks if cache_key(t["artist"], t["title"]) not in cache]
    logging.info(
        f"{len(tracks) - len(uncached):,} tracks already in cache, "
        f"{len(uncached):,} need searching"
    )

    not_found = []
    for i, track in enumerate(uncached):
        key = cache_key(track["artist"], track["title"])
        track_id = search_track_id(sp, track["artist"], track["title"])
        cache[key] = track_id  # None stored so we don't retry on re-run
        if track_id is None:
            not_found.append(f"{track['artist']} — {track['title']}")

        if (i + 1) % 100 == 0:
            logging.info(f"  Searched {i + 1:,}/{len(uncached):,}...")
            save_cache(cache, ID_CACHE_PATH)

        time.sleep(0.05)

    save_cache(cache, ID_CACHE_PATH)
    return not_found


# ── Phase 2: audio features via Soundcharts ───────────────────────────────────

def make_soundcharts_session() -> requests.Session:
    app_id  = os.environ.get("SOUNDCHARTS_APP_ID")
    api_key = os.environ.get("SOUNDCHARTS_API_KEY")
    if not app_id or not api_key:
        raise EnvironmentError(
            "SOUNDCHARTS_APP_ID and SOUNDCHARTS_API_KEY must be set in .env.\n"
            "  Sandbox (2-song test): set both to 'soundcharts'\n"
            "  Production (free, 1k calls): register at https://developers.soundcharts.com"
        )
    session = requests.Session()
    session.headers.update({"x-app-id": app_id, "x-api-key": api_key})
    return session


def fetch_soundcharts_features(
    session: requests.Session,
    track_ids: list[str],
) -> dict[str, dict]:
    """
    Calls Soundcharts /api/v2.25/song/by-platform/spotify/{id} for each track.

    Loads a local cache (soundcharts_cache.json) first and skips IDs already
    fetched — so interrupted runs and re-runs never re-spend quota calls.
    Deduplicates track_ids before calling so the same Spotify ID is never
    fetched twice in one run.

    Returns dict mapping spotify_id → {feature: value} (cache + new results).
    Stops and saves on quota exhaustion.
    """
    sc_cache: dict[str, dict] = load_cache(SOUNDCHARTS_CACHE_PATH)

    # Deduplicate while preserving order; skip IDs already in cache
    seen: set[str] = set()
    to_fetch: list[str] = []
    for tid in track_ids:
        if tid not in seen:
            seen.add(tid)
            if tid not in sc_cache:
                to_fetch.append(tid)

    cache_hits = len(seen) - len(to_fetch)
    logging.info(
        f"  {cache_hits:,} track(s) already in Soundcharts cache — "
        f"{len(to_fetch):,} new call(s) needed"
    )

    for i, track_id in enumerate(to_fetch):
        url = f"{SOUNDCHARTS_BASE}/api/v2.25/song/by-platform/spotify/{track_id}"
        try:
            resp = session.get(url, timeout=10)
        except requests.RequestException as exc:
            logging.warning(f"  Network error for {track_id}: {exc}")
            time.sleep(1)
            continue

        if resp.status_code in (401, 403):
            try:
                data = resp.json()
                errs = data.get("errors", [{}])
                msg  = errs[0].get("message", "")
            except Exception:
                msg = resp.text[:200]
            if "quota" in msg.lower() or "limit" in msg.lower():
                logging.error(
                    f"\n  Soundcharts free quota exhausted after {i} new call(s).\n"
                    f"  {len(sc_cache):,} tracks in cache — saving partial results."
                )
            else:
                logging.error(
                    f"Soundcharts auth error ({resp.status_code}): {msg}\n"
                    "Check SOUNDCHARTS_APP_ID / SOUNDCHARTS_API_KEY in .env"
                )
            break

        if resp.status_code == 404:
            logging.debug(f"  Not in Soundcharts: {track_id}")
            # Cache the miss so re-runs don't retry a track that doesn't exist
            sc_cache[track_id] = {}
        else:
            data  = resp.json()
            audio = data.get("object", {}).get("audio")
            sc_cache[track_id] = {col: audio.get(col) for col in AUDIO_FEATURE_COLS} if audio else {}

        # Checkpoint every 50 fetches so a crash doesn't lose progress
        if (i + 1) % 50 == 0:
            save_cache(sc_cache, SOUNDCHARTS_CACHE_PATH)
            logging.info(f"  Soundcharts: {i + 1:,}/{len(to_fetch):,} fetched (cache saved)...")

        time.sleep(0.05)  # ~20 req/s; Soundcharts allows 5,000/min

    save_cache(sc_cache, SOUNDCHARTS_CACHE_PATH)

    # Return only entries that have actual features (not misses)
    return {tid: feats for tid, feats in sc_cache.items() if feats}


# ── Output ─────────────────────────────────────────────────────────────────────

def save_features_csv(
    tracks: list[dict],
    cache: dict,
    features_map: dict,
    output_path: Path,
) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["artist", "title", "play_count", "spotify_id"] + AUDIO_FEATURE_COLS
    written = 0
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for track in tracks:
            key = cache_key(track["artist"], track["title"])
            track_id = cache.get(key)
            if not track_id or track_id not in features_map:
                continue
            writer.writerow({
                "artist":     track["artist"],
                "title":      track["title"],
                "play_count": track["play_count"],
                "spotify_id": track_id,
                **features_map[track_id],
            })
            written += 1
    return written


def save_not_found(not_found: list[str], path: Path) -> None:
    if not not_found:
        return
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"Tracks not found on Spotify ({len(not_found)}):\n\n")
        f.write("\n".join(not_found))
    logging.info(f"{len(not_found):,} unmatched tracks logged → {path}")


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Enrich top Last.fm tracks with audio features via Soundcharts"
    )
    parser.add_argument(
        "--input", type=Path, default=DEFAULT_INPUT, metavar="PATH",
        help="track_counts.csv from fetch_lastfm.py (default: data/raw/track_counts.csv)",
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT, metavar="PATH",
        help="Output path (default: data/raw/audio_features.csv)",
    )
    parser.add_argument(
        "--limit", type=int, default=DEFAULT_LIMIT, metavar="N",
        help=f"Process only the top N tracks by play count (default: {DEFAULT_LIMIT}). "
             "Keeps Soundcharts call count within the free-tier 1,000-request quota.",
    )
    parser.add_argument(
        "--features-only", action="store_true",
        help="Skip Phase 1 (Spotify ID search); use only what's already in the cache.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    with open(args.input, encoding="utf-8") as f:
        all_tracks = list(csv.DictReader(f))

    # Sort by play count descending and cap at --limit
    all_tracks.sort(key=lambda t: int(t["play_count"]), reverse=True)
    tracks = all_tracks[: args.limit]
    if not tracks:
        logging.error("No tracks found in input file — exiting.")
        return
    logging.info(
        f"Loaded {len(all_tracks):,} tracks; working on top {len(tracks):,} by play count "
        f"(min play count in set: {tracks[-1]['play_count']})"
    )

    cache = load_cache(ID_CACHE_PATH)

    # ── Phase 1: resolve Spotify track IDs ──────────────────────────────────
    if args.features_only:
        logging.info("Phase 1 skipped (--features-only): using existing ID cache")
        not_found = []
    else:
        sp = make_spotify_client()
        if sp is None:
            logging.warning(
                "Phase 1 skipped: spotipy not installed or Spotify credentials missing. "
                "Falling back to existing ID cache (%d entries).",
                sum(1 for v in cache.values() if v),
            )
            not_found = []
        else:
            logging.info("Phase 1: resolving Spotify track IDs...")
            not_found = resolve_track_ids(sp, tracks, cache)
            save_not_found(not_found, NOT_FOUND_PATH)

    # ── Phase 2: fetch audio features from Soundcharts ──────────────────────
    track_ids = [
        cache[cache_key(t["artist"], t["title"])]
        for t in tracks
        if cache_key(t["artist"], t["title"]) in cache
        and cache[cache_key(t["artist"], t["title"])] is not None
    ]
    logging.info(
        f"Phase 2: fetching Soundcharts audio features for {len(track_ids):,} tracks..."
    )

    session = make_soundcharts_session()
    features_map = fetch_soundcharts_features(session, track_ids)

    written = save_features_csv(tracks, cache, features_map, args.output)
    logging.info(
        f"Done. {written:,} enriched tracks saved → {args.output}"
        + (f" ({len(not_found):,} had no Spotify match)" if not_found else "")
    )


if __name__ == "__main__":
    main()
