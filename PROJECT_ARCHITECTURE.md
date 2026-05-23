# Music Taste Visualizer — Project Architecture

## Overview

A static, data-driven web app that maps a user's listening history into an interactive 2D "music cloud." Songs are positioned by sonic similarity using audio features and UMAP dimensionality reduction. Bubble size reflects how often a song has been played (log scale). The end result is a shareable snapshot that requires no live API calls when viewed.

---

## Data Sources

### Last.fm (Listening History)
- **Purpose**: Retrieve top tracks and play counts for a user
- **Why Last.fm**: Spotify's API only returns the last 50 played tracks. Last.fm is purpose-built for logging full play history going back years.
- **Auth**: API key only (no OAuth required for read operations)
- **Approach**: Uses `user.getTopTracks(period=overall)` — results are pre-aggregated and sorted by play count descending, so we can stop early once we hit the minimum play count threshold instead of fetching the entire scrobble history.
- **Key data pulled**: artist, track title, play count, MusicBrainz ID (mbid)
- **Threshold**: Only tracks with 5+ plays are included (configurable via `--min-plays`)

### Soundcharts API (Audio Features)
- **Purpose**: Enrich each unique track with 12-dimensional audio feature vectors
- **Why Soundcharts**: Spotify deprecated their public audio features endpoint in late 2024. Soundcharts provides an equivalent feature set, looked up via Spotify track ID.
- **Auth**: App ID + API key via request headers (`x-app-id`, `x-api-key`)
- **Free tier**: 1,000 requests — sufficient for the top 1,000 tracks by play count
- **Lookup**: `GET /api/v2.25/song/by-platform/spotify/{spotifyId}`
- **Audio features used**:
  - `acousticness` — confidence the track is acoustic (0–1)
  - `danceability` — rhythmic suitability for dancing (0–1)
  - `energy` — intensity and activity (0–1)
  - `instrumentalness` — predicts absence of vocals (0–1)
  - `key` — musical key (0–11, standard pitch class notation)
  - `liveness` — presence of live audience (0–1)
  - `loudness` — overall loudness in dB (typically –60 to 0)
  - `mode` — major (1) or minor (0)
  - `speechiness` — presence of spoken words (0–1)
  - `tempo` — estimated BPM
  - `timeSignature` — beats per bar (3–7)
  - `valence` — musical positivity/mood (0–1)

### Spotify Web API (Track ID Resolution Only)
- **Purpose**: Resolve (artist, title) pairs to Spotify track IDs, which are used as lookup keys for Soundcharts
- **Auth**: OAuth 2.0 client credentials (developer app required)
- **Note**: Track search is still freely available; only the audio features endpoint was deprecated. This step is optional — resolved IDs are cached locally and reused across runs.

---

## Pipeline Architecture

### Running the full pipeline

```bash
python run_pipeline.py <lastfm_username>
```

This runs all four steps in order and aborts on the first failure. Options:

| Flag | Default | Description |
|---|---|---|
| `--min-plays N` | `5` | Minimum play count to include a track |
| `--features-only` | off | Skip Spotify ID search; use only cached IDs |

Individual steps can also be run directly (see below).

---

### Phase 1 — Data Pipeline (`pipeline/`)

#### Step 1: `fetch_lastfm.py` — Last.fm top tracks
1. Authenticate with Last.fm API (API key from `.env`)
2. Page through `user.getTopTracks(period=overall, limit=1000)` in descending play count order
3. Stop early once play count drops below the minimum threshold
4. Retries up to 3× with exponential backoff on transient 5xx responses
5. Output: `data/raw/track_counts.csv` — `(artist, title, play_count, mbid)`

```bash
python pipeline/fetch_lastfm.py <username> [--min-plays N] [--output PATH]
```

#### Step 2: `fetch_spotify.py` — Audio feature enrichment
1. Load `track_counts.csv`, sort by play count descending, take top 1,000
2. **Phase 1 — Spotify ID resolution**: For each track not already in `spotify_id_cache.json`, search Spotify by artist + title to get a track ID. Results are cached so re-runs are free.
3. **Phase 2 — Soundcharts audio features**: For each track with a Spotify ID not already in `soundcharts_cache.json`, call Soundcharts to retrieve the 12 audio features. Results are cached and checkpointed every 50 fetches — interrupted runs resume without re-spending quota.
4. Output: `data/raw/audio_features.csv` — `(artist, title, play_count, spotify_id, feature_1..12)`

```bash
python pipeline/fetch_spotify.py [--input PATH] [--output PATH] [--limit N] [--features-only]
```

#### Step 3: `build_dataset.py` — Join and clean
1. Load `audio_features.csv` (the enriched top-1,000 set) as the primary dataset
2. Look up play counts from `track_counts.csv` by (artist, title)
3. Skip any tracks with null or missing feature values (logged as warnings)
4. Output: `data/processed/dataset.csv` — final clean joined dataset

```bash
python pipeline/build_dataset.py [--counts PATH] [--features PATH] [--output PATH]
```

#### Caching strategy
All API results are persisted locally to avoid re-spending quota on re-runs:

| Cache file | Keyed by | Contents |
|---|---|---|
| `data/raw/spotify_id_cache.json` | `"artist\|\|title"` | Spotify track ID (or `null` if not found) |
| `data/raw/soundcharts_cache.json` | Spotify track ID | 12 audio feature values (or `{}` for 404s) |

---

### Phase 2 — Data Processing (`processing/`)

All processing is handled by a single script: `build_viz_data.py`.

1. Load `data/processed/dataset.csv`
2. Normalize all 12 audio feature columns with `StandardScaler` (zero mean, unit variance per feature — preferred over MinMax because loudness in dB and tempo in BPM are on very different scales from the 0–1 features)
3. Run UMAP on the normalized 12-dimensional vectors → 2D coordinates `(x, y)`
4. Run HDBSCAN clustering on the same normalized 12D space (not on the 2D UMAP output — clustering in the full feature space is more accurate)
5. Compute bubble size: `log_size = log(play_count + 1)`
6. Output: `data/processed/viz_data.json` — one record per track with `(artist, title, play_count, spotify_id, x, y, cluster, log_size)`

```bash
python processing/build_viz_data.py [--input PATH] [--output PATH] \
  [--umap-neighbors N] [--umap-min-dist F] [--hdbscan-min-cluster N]
```

| Parameter | Default | Effect |
|---|---|---|
| `--umap-neighbors` | `15` | Local neighborhood size; lower = more local structure |
| `--umap-min-dist` | `0.1` | Minimum spread of points in 2D |
| `--hdbscan-min-cluster` | `15` | Minimum tracks to form a cluster |

---

### Phase 3 — Interactive Web App (`web/`)

Static HTML/JS — no backend required when viewing.

- **`web/index.html`**: Dark-themed layout with header and filter controls
- **`web/app.js`**: Loads `data/processed/viz_data.json`, renders a Plotly.js scatter plot

Features:
- Hover tooltip: track title, artist, play count, cluster
- Bubble size = `log(play_count + 1)`, scaled to pixel range 4–24
- Bubble color = cluster assignment (qualitative palette; cluster `-1` / noise rendered in grey)
- Cluster dropdown filter (dynamically populated from data)
- Min-plays range slider to hide low-count tracks
- Graceful error state if viz data hasn't been generated yet

To view locally:

```bash
# From the project root:
python -m http.server 8080
# Then open: http://localhost:8080/web/index.html
```

---

## Static vs. Live

This project is **static by design**. The pipeline is run locally to generate a snapshot. The resulting web app reads pre-generated data and makes no live API calls. This makes it:
- Easy to share (just a static site)
- Fast to load
- Suitable for hosting on GitHub Pages

The pipeline can be re-run periodically (e.g. monthly) to regenerate the snapshot with updated listening data.

---

## Tech Stack

| Layer | Tool |
|---|---|
| Language | Python |
| Last.fm API | `requests` (raw HTTP) |
| Spotify API | `spotipy` (track ID resolution only, optional) |
| Audio features | Soundcharts API (`requests`) |
| Data storage | CSV (flat files in `data/raw/` and `data/processed/`) |
| Dimensionality reduction | `umap-learn` |
| Clustering | `hdbscan` |
| Normalization | `scikit-learn` (`StandardScaler`) |
| Visualization | Plotly.js (CDN) |
| Hosting | GitHub Pages |

---

## Project Structure

```
music_taste_viz/
├── PROJECT_ARCHITECTURE.md
├── .env                              # API keys (not committed)
├── requirements.txt
├── run_pipeline.py                   # end-to-end runner: fetch → enrich → join → viz
├── pipeline/
│   ├── fetch_lastfm.py               # pull top tracks and play counts from Last.fm
│   ├── fetch_spotify.py              # resolve Spotify IDs + fetch Soundcharts audio features
│   └── build_dataset.py              # inner-join, clean, output final dataset
├── processing/
│   └── build_viz_data.py             # normalize → UMAP → HDBSCAN → viz_data.json
├── data/
│   ├── raw/
│   │   ├── track_counts.csv          # Last.fm output
│   │   ├── spotify_id_cache.json     # (artist, title) → Spotify ID
│   │   ├── soundcharts_cache.json    # Spotify ID → audio features
│   │   ├── spotify_not_found.txt     # tracks with no Spotify match (informational)
│   │   └── audio_features.csv        # enriched top-1000 tracks
│   └── processed/
│       ├── dataset.csv               # joined + cleaned dataset (pipeline output)
│       └── viz_data.json             # UMAP coords + clusters (processing output, loaded by web app)
└── web/
    ├── index.html                    # app shell + filter controls
    └── app.js                        # Plotly.js scatter plot, data loading, filtering
```

---

## Key Decisions & Rationale

- **Last.fm over Spotify for history**: Spotify API caps recent tracks at 50; Last.fm logs everything.
- **`user.getTopTracks` over raw scrobbles**: Pre-aggregated and sorted, so we can stop early at the play count threshold instead of paginating the full scrobble history.
- **Soundcharts over Spotify for audio features**: Spotify deprecated the `/audio-features` endpoint for new developer apps in late 2024. Soundcharts provides an identical 12-feature set via Spotify ID lookup.
- **Top 1,000 tracks only**: Fits within the Soundcharts free tier (1,000 requests). Tracks below ~78 plays are less meaningful for a "music taste" snapshot anyway.
- **Dual caching layer**: Spotify ID cache + Soundcharts feature cache mean the pipeline can be interrupted and resumed without re-spending any API quota.
- **Inner-join in `build_dataset.py`**: Dataset is built from the enriched tracks outward, so only tracks with complete audio features are included. No silent large-scale drops.
- **All 12 features for UMAP/HDBSCAN**: The full feature set — including `key`, `mode`, and `timeSignature` — is used for both dimensionality reduction and clustering. `StandardScaler` normalizes across the mixed scales (dB, BPM, 0–1 ratios) before any distance computation.
- **Single processing script**: `build_viz_data.py` handles normalize → UMAP → HDBSCAN in one pass. Splitting into separate scripts would add orchestration complexity with no benefit, since all three steps share the same normalized matrix.
- **UMAP over t-SNE**: Better preservation of global structure; faster; more suitable for sharing clusters.
- **Clustering in original 12D space**: UMAP coordinates are for display only; cluster assignments are computed in the full feature space for accuracy.
- **Log scale for play count**: Prevents songs with very high play counts from visually dominating.
- **Plotly.js over D3**: Faster to build correct interactions (hover, zoom, pan); good defaults for scatter plots. D3 would add significant boilerplate for the same result.
- **Static output**: Simplifies sharing and hosting; no server required.
- **Retry logic in `fetch_lastfm.py`**: Last.fm intermittently returns 500s on specific pages (observed in practice). Up to 3 retries with exponential backoff (1s, 2s, 4s) recover from transient failures without aborting the run.
