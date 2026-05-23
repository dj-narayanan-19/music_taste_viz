const DATA_PATH = "../data/processed/viz_data.json";

const TOP_N_ARTISTS = 50;

// Fixed axis bounds — never recalculate on filter so the view stays stable
const X_RANGE = [-2.2, 7.9];
const Y_RANGE = [1.7, 10.0];

const BASE_LAYOUT = {
  paper_bgcolor: "#f7f5f0",
  plot_bgcolor:  "#f7f5f0",
  font: { color: "#1a1a2e", size: 11, family: "Georgia, serif" },
  xaxis: { range: X_RANGE, showgrid: true, gridcolor: "#e8e5e0",
           zeroline: false, showticklabels: false, fixedrange: true },
  yaxis: { range: Y_RANGE, showgrid: true, gridcolor: "#e8e5e0",
           zeroline: false, showticklabels: false, fixedrange: true },
  hovermode: "closest",
  showlegend: false,
  margin: { t: 12, r: 12, b: 12, l: 12 },
};

const CONFIG = { responsive: true, displayModeBar: false, scrollZoom: false };

// ── Palettes ────────────────────────────────────────────────────────────────

// 50 perceptually-spaced colors via golden angle, tuned for cream background
function makeArtistPalette(n) {
  return Array.from({ length: n }, (_, i) => {
    const h = (i * 137.508) % 360;
    const s = 55 + (i % 3) * 8;   // 55 / 63 / 71
    const l = 34 + (i % 2) * 9;   // 34 or 43
    return `hsl(${h.toFixed(1)},${s}%,${l}%)`;
  });
}

// 6 hand-picked genre colors — distinct, readable on cream
const GENRE_PALETTE = [
  "#2471a3",  // 0 — blue      indie rock/alt
  "#c0392b",  // 1 — red       conscious/lyrical hip-hop
  "#27ae60",  // 2 — green     indie folk/lo-fi
  "#7d3c98",  // 3 — purple    indie soul/alt-R&B
  "#d35400",  // 4 — orange    West Coast / boom-bap
  "#148f77",  // 5 — teal      indie pop / alt-pop
];

// Muted color for non-selected tracks
const OTHER_COLOR   = "#9898a8";
const OTHER_OPACITY = 0.40;

const ARTIST_PALETTE = makeArtistPalette(TOP_N_ARTISTS);

// ── State ───────────────────────────────────────────────────────────────────

let allRecords     = [];
let colorMode      = "artist";   // "artist" | "genre"
let selectedGroup  = null;       // artist name (string) or cluster id (number) | null
let artistColorMap = {};         // artist → hsl color
let clusterLabels  = {};         // cluster id → display label
let initialized    = false;

// ── Helpers ─────────────────────────────────────────────────────────────────

function scaleSizes(logSizes) {
  const mn = Math.min(...logSizes), mx = Math.max(...logSizes), r = mx - mn || 1;
  return logSizes.map(s => 4 + ((s - mn) / r) * 20);
}

function hoverText(r) {
  return `<b>${r.title}</b><br>${r.artist}<br>${r.play_count.toLocaleString()} plays`;
}

// Top-N artists by track count
function rankArtists(records, n) {
  const counts = {};
  for (const r of records) counts[r.artist] = (counts[r.artist] || 0) + 1;
  return Object.entries(counts).sort((a, b) => b[1] - a[1]).slice(0, n).map(([a]) => a);
}

// Genre cluster display label: top-3 artists in that cluster by track count
function buildClusterLabels(records) {
  const clusterCounts = {};
  for (const r of records) {
    const cid = r.artist_cluster;
    if (!clusterCounts[cid]) clusterCounts[cid] = {};
    clusterCounts[cid][r.artist] = (clusterCounts[cid][r.artist] || 0) + 1;
  }
  const labels = {};
  for (const [cid, artistCounts] of Object.entries(clusterCounts)) {
    labels[cid] = Object.entries(artistCounts)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 3)
      .map(([a]) => a)
      .join(" · ");
  }
  return labels;
}

// ── Trace building ──────────────────────────────────────────────────────────

function buildArtistTraces(records) {
  const knownArtists = new Set(Object.keys(artistColorMap));
  const isFiltered   = selectedGroup !== null;

  const colored = {}, other = [];
  for (const r of records) {
    const isKnown    = knownArtists.has(r.artist);
    const isSelected = r.artist === selectedGroup;
    if (isFiltered ? isSelected : isKnown) {
      (colored[r.artist] = colored[r.artist] || []).push(r);
    } else {
      other.push(r);
    }
  }

  const traces = [];
  if (other.length) {
    traces.push({
      type: "scatter", mode: "markers",
      x: other.map(r => r.x), y: other.map(r => r.y),
      text: other.map(hoverText), hovertemplate: "%{text}<extra></extra>",
      marker: { color: OTHER_COLOR, size: scaleSizes(other.map(r => r.log_size)),
                opacity: OTHER_OPACITY, line: { width: 0.8, color: "#c0bdb8" } },
    });
  }
  for (const [artist, subset] of Object.entries(colored)) {
    traces.push({
      type: "scatter", mode: "markers", name: artist,
      x: subset.map(r => r.x), y: subset.map(r => r.y),
      text: subset.map(hoverText), hovertemplate: "%{text}<extra></extra>",
      marker: { color: artistColorMap[artist],
                size: scaleSizes(subset.map(r => r.log_size)),
                opacity: 0.88, line: { width: 1, color: "rgba(0,0,0,0.2)" } },
    });
  }
  return traces;
}

function buildGenreTraces(records) {
  const clusterIds = [...new Set(records.map(r => r.artist_cluster))].sort((a, b) => a - b);
  const isFiltered  = selectedGroup !== null;
  const traces = [];

  // "Other" bucket first (unselected clusters when filtered)
  if (isFiltered) {
    const other = records.filter(r => r.artist_cluster !== selectedGroup);
    if (other.length) {
      traces.push({
        type: "scatter", mode: "markers",
        x: other.map(r => r.x), y: other.map(r => r.y),
        text: other.map(hoverText), hovertemplate: "%{text}<extra></extra>",
        marker: { color: OTHER_COLOR, size: scaleSizes(other.map(r => r.log_size)),
                  opacity: OTHER_OPACITY, line: { width: 0.8, color: "#c0bdb8" } },
      });
    }
  }

  for (const cid of clusterIds) {
    if (isFiltered && cid !== selectedGroup) continue;
    const subset = records.filter(r => r.artist_cluster === cid);
    if (!subset.length) continue;
    const color = GENRE_PALETTE[cid % GENRE_PALETTE.length];
    traces.push({
      type: "scatter", mode: "markers",
      name: clusterLabels[cid] || `cluster ${cid}`,
      x: subset.map(r => r.x), y: subset.map(r => r.y),
      text: subset.map(hoverText), hovertemplate: "%{text}<extra></extra>",
      marker: { color, size: scaleSizes(subset.map(r => r.log_size)),
                opacity: 0.88, line: { width: 1, color: "rgba(0,0,0,0.2)" } },
    });
  }
  return traces;
}

// ── Render ───────────────────────────────────────────────────────────────────

function render() {
  const traces = colorMode === "artist"
    ? buildArtistTraces(allRecords)
    : buildGenreTraces(allRecords);

  const visibleCount = selectedGroup === null
    ? allRecords.length
    : colorMode === "artist"
      ? allRecords.filter(r => r.artist === selectedGroup).length
      : allRecords.filter(r => r.artist_cluster === selectedGroup).length;

  document.getElementById("track-count").textContent =
    `${visibleCount.toLocaleString()} of ${allRecords.length.toLocaleString()} tracks`;

  if (!initialized) {
    Plotly.newPlot("plot", traces, BASE_LAYOUT, CONFIG);
    initialized = true;
  } else {
    Plotly.react("plot", traces, BASE_LAYOUT, CONFIG);
  }
}

// ── Chip strip ───────────────────────────────────────────────────────────────

function buildChips() {
  const strip = document.getElementById("chip-strip");
  strip.innerHTML = "";
  selectedGroup = null;

  if (colorMode === "artist") {
    const topArtists = rankArtists(allRecords, TOP_N_ARTISTS);
    for (const artist of topArtists) {
      const color = artistColorMap[artist];
      const chip  = document.createElement("div");
      chip.className   = "chip";
      chip.dataset.key = artist;
      chip.style.color = color;
      chip.innerHTML   =
        `<div class="chip-dot" style="background:${color}"></div>${artist}`;
      chip.addEventListener("click", () => handleChipClick(chip, artist));
      strip.appendChild(chip);
    }
  } else {
    const clusterIds = [...new Set(allRecords.map(r => r.artist_cluster))].sort((a, b) => a - b);
    for (const cid of clusterIds) {
      const color = GENRE_PALETTE[cid % GENRE_PALETTE.length];
      const label = clusterLabels[cid] || `cluster ${cid}`;
      const chip  = document.createElement("div");
      chip.className   = "chip";
      chip.dataset.key = cid;
      chip.style.color = color;
      chip.innerHTML   =
        `<div class="chip-dot" style="background:${color}"></div>${label}`;
      chip.addEventListener("click", () => handleChipClick(chip, cid));
      strip.appendChild(chip);
    }
  }
}

function handleChipClick(chip, key) {
  const strip = document.getElementById("chip-strip");
  if (selectedGroup === key) {
    selectedGroup = null;
    strip.querySelectorAll(".chip").forEach(c => c.classList.remove("selected"));
  } else {
    selectedGroup = key;
    strip.querySelectorAll(".chip").forEach(c => c.classList.remove("selected"));
    chip.classList.add("selected");
  }
  render();
}

// ── Mode toggle ───────────────────────────────────────────────────────────────

document.querySelectorAll(".mode-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    if (btn.dataset.mode === colorMode) return;
    colorMode = btn.dataset.mode;
    document.querySelectorAll(".mode-btn").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    buildChips();
    render();
  });
});

// ── Bootstrap ────────────────────────────────────────────────────────────────

fetch(DATA_PATH)
  .then(res => {
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return res.json();
  })
  .then(data => {
    allRecords = data;

    const topArtists = rankArtists(allRecords, TOP_N_ARTISTS);
    topArtists.forEach((artist, i) => {
      artistColorMap[artist] = ARTIST_PALETTE[i];
    });
    clusterLabels = buildClusterLabels(allRecords);

    document.getElementById("loading").style.display = "none";
    buildChips();
    render();
  })
  .catch(err => {
    document.getElementById("loading").style.display = "none";
    document.getElementById("plot").innerHTML =
      `<div style="padding:40px;font-family:sans-serif;color:#c00;font-size:0.85rem">
        Failed to load data: ${err.message}<br><br>
        Run: python processing/build_viz_data.py
      </div>`;
  });
