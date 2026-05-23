# Music Taste Visualizer

A static web app that turns your Last.fm listening history into an interactive "music cloud" — every song you've played, plotted by sound.

## What it does

Songs are positioned in 2D by how they **sound** (tempo, energy, mood, danceability, etc.), not by genre or artist. Songs that sound similar end up near each other. Bigger bubbles = songs you've played more.

You can color the map two ways:
- **By artist** — each of your top 50 artists gets a color
- **By genre** — songs grouped into 6 broad sonic clusters (think indie rock vs hip-hop vs lo-fi)

Click any chip to highlight just that artist or cluster.

## How it works

1. **Fetch** — pulls your top tracks + play counts from Last.fm
2. **Enrich** — looks up audio features (energy, tempo, mood, etc.) for each track via Soundcharts
3. **Reduce** — runs UMAP to compress 10 audio dimensions down to 2D coordinates
4. **Cluster** — groups similar-sounding artists together with KMeans
5. **Visualize** — renders everything as a Plotly.js scatter plot in a static HTML page

The pipeline runs automatically every day via GitHub Actions and updates the data in the repo.

## Run it yourself

**Prerequisites:** Python 3.12+, API keys for Last.fm, Spotify (client credentials), and Soundcharts. Put them in a `.env` file:

```
LASTFM_API_KEY=...
SPOTIFY_CLIENT_ID=...
SPOTIFY_CLIENT_SECRET=...
SOUNDCHARTS_APP_ID=...
SOUNDCHARTS_API_KEY=...
```

**Install dependencies:**
```bash
pip install -r requirements.txt
```

**Run the pipeline:**
```bash
python run_pipeline.py <your_lastfm_username>
```

This runs all four steps and writes `data/processed/viz_data.json`.

**View the result:**
```bash
python -m http.server 8080
# Open http://localhost:8080/web/index.html
```

## Options

| Flag | Default | Description |
|---|---|---|
| `--min-plays N` | `10` | Only include tracks played at least N times |
| `--limit N` | `1000` | Max tracks to enrich per run |
| `--features-only` | off | Skip Spotify ID search; use cached IDs only |

## Project structure

```
music_taste_viz/
├── run_pipeline.py          # runs all steps in order
├── pipeline/
│   ├── fetch_lastfm.py      # step 1: get top tracks + play counts
│   ├── fetch_spotify.py     # step 2: resolve Spotify IDs + fetch audio features
│   └── build_dataset.py     # step 3: join and clean
├── processing/
│   └── build_viz_data.py    # step 4: UMAP + clustering → viz_data.json
├── web/
│   ├── index.html           # app shell
│   └── app.js               # Plotly scatter, color modes, chip filters
└── data/
    ├── raw/                 # pipeline intermediate files + caches
    └── processed/           # dataset.csv + viz_data.json (web app reads this)
```

## Why these tools?

- **Last.fm** instead of Spotify for play history — Spotify only keeps your last 50 plays; Last.fm logs everything
- **Soundcharts** instead of Spotify for audio features — Spotify deprecated their audio features API for new apps in late 2024
- **UMAP** instead of t-SNE — better at preserving global structure (similar artists cluster together, not just similar songs)
- **Static site** — nothing to host, easy to share, works on GitHub Pages
