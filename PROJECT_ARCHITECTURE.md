# Music Taste Visualizer — Project Architecture

## Overview

A static, data-driven web app that maps a user's listening history into an interactive 2D "music cloud." Songs are positioned by sonic similarity using Spotify's audio features and UMAP dimensionality reduction. Bubble size reflects how often a song has been played (log scale). The end result is a shareable snapshot that requires no live API calls when viewed.

---

## Data Sources

### Last.fm (Listening History)
- **Purpose**: Retrieve all scrobbled tracks and play counts for a user
- **Why Last.fm**: Spotify's API only returns the last 50 played tracks. Last.fm is purpose-built for logging full play history going back years.
- **Auth**: API key only (no OAuth required for read operations)
- **Key data pulled**: track name, artist, play count
- **Threshold**: Only include tracks above a minimum play count (e.g. 5+ plays) to reduce noise

### Spotify Web API (Audio Features)
- **Purpose**: Enrich each unique track with 9-dimensional audio feature vectors
- **Why Spotify**: Best-in-class per-track numeric audio metadata
- **Auth**: OAuth 2.0 (developer app required)
- **Audio features used**:
  - `energy` — intensity and activity (0–1)
  - `danceability` — rhythmic suitability for dancing (0–1)
  - `valence` — musical positivity/mood (0–1)
  - `acousticness` — confidence the track is acoustic (0–1)
  - `instrumentalness` — predicts absence of vocals (0–1)
  - `speechiness` — presence of spoken words (0–1)
  - `liveness` — presence of live audience (0–1)
  - `loudness` — overall loudness in dB (typically –60 to 0)
  - `tempo` — estimated BPM

---

## Pipeline Architecture

### Phase 1 — Data Pipeline
1. Authenticate with Last.fm API (API key)
2. Fetch all scrobbles for target user, paginating through full history
3. Aggregate to unique tracks with total play count
4. Filter to tracks above the minimum play threshold
5. Authenticate with Spotify API (OAuth 2.0)
6. For each unique track, look up Spotify track ID and fetch audio features
7. Store enriched dataset locally — SQLite or flat JSON/CSV for simplicity
8. Output: a clean table of `(track_id, artist, title, play_count, feature_1..9)`

### Phase 2 — Data Processing & Visualization
1. Normalize audio feature vectors (min-max or standard scaling)
2. Run UMAP on the normalized 9-dimensional vectors → 2D coordinates `(x, y)`
3. Optionally run HDBSCAN clustering on the *original* high-dimensional space for automatic genre grouping (used for color coding, not position)
4. Compute bubble size using log scale: `size = log(play_count + 1)`
5. Output: enriched table with `(x, y, cluster_id, log_size, metadata)`

### Phase 3 — Interactive Web App
1. Load the processed data (static JSON or CSV)
2. Render interactive scatter plot — candidates: Plotly.js or D3.js
   - Plotly: faster to build, good defaults
   - D3: more control over interactions and styling
3. Features:
   - Hover tooltip: track name, artist, play count, cluster
   - Bubble size = log(play count)
   - Bubble color = cluster/genre group
   - Optional: 30-second Spotify preview on click (via Spotify preview URLs)
   - Optional: filter by cluster, date range, or play count range
4. Static HTML/JS — no backend required when viewing

---

## Static vs. Live

This project is **static by design**. The pipeline is run locally (or via CI) to generate a snapshot. The resulting web app reads pre-generated data and makes no live API calls. This makes it:
- Easy to share (just a static site)
- Fast to load
- Suitable for hosting on GitHub Pages

The pipeline can be re-run periodically (e.g. monthly) to regenerate the snapshot with updated listening data.

---

## Tech Stack (Planned)

| Layer | Tool |
|---|---|
| Language | Python |
| Last.fm API | `pylast` or raw `requests` |
| Spotify API | `spotipy` |
| Data storage | SQLite or CSV/JSON |
| Dimensionality reduction | `umap-learn` |
| Clustering | `hdbscan` |
| Normalization | `scikit-learn` |
| Visualization | Plotly.js or D3.js |
| Hosting | GitHub Pages |

---

## Project Structure (Planned)

```
music_taste_viz/
├── PROJECT_ARCHITECTURE.md   ← this file
├── pipeline/
│   ├── fetch_lastfm.py       # pull scrobbles and play counts
│   ├── fetch_spotify.py      # enrich tracks with audio features
│   └── build_dataset.py      # combine, clean, store
├── processing/
│   ├── normalize.py          # scale audio features
│   ├── umap_reduce.py        # run UMAP, output 2D coords
│   └── cluster.py            # HDBSCAN clustering
├── data/
│   ├── raw/                  # raw API responses
│   └── processed/            # final enriched dataset for viz
├── web/
│   ├── index.html
│   ├── app.js
│   └── data.json             # pre-generated snapshot
└── README.md
```

---

## Key Decisions & Rationale

- **Last.fm over Spotify for history**: Spotify API caps recent tracks at 50; Last.fm logs everything.
- **Spotify for audio features**: Best structured numeric metadata for ML/viz.
- **UMAP over t-SNE**: Better preservation of global structure; faster; more suitable for sharing clusters.
- **Log scale for play count**: Prevents songs with very high play counts from visually dominating.
- **Static output**: Simplifies sharing and hosting; no server required.
- **Nearest-neighbor computed in original space**: 2D UMAP coordinates are for display only; similarity is measured in the full 9D feature space.
