const DATA_PATH = "../data/processed/viz_data.json";

const TOP_N_ARTISTS = 50;
const X_RANGE = [-2.2, 7.9];
const Y_RANGE = [1.7, 10.0];
const OTHER_COLOR   = "#b0aead";
const OTHER_OPACITY = 0.30;

// One place to change the group-mode label
const NEIGHBORHOOD_LABEL = "Neighborhood";

const GENRE_PALETTE = [
  "#2471a3", "#c0392b", "#27ae60",
  "#7d3c98", "#d35400", "#148f77",
];

function makeArtistPalette(n) {
  return Array.from({ length: n }, (_, i) => {
    const h = (i * 137.508) % 360;
    const s = 55 + (i % 3) * 8;
    const l = 34 + (i % 2) * 9;
    return `hsl(${h.toFixed(1)},${s}%,${l}%)`;
  });
}

const ARTIST_PALETTE = makeArtistPalette(TOP_N_ARTISTS);

// ── State ───────────────────────────────────────────────────────────────────

let allRecords    = [];
let colorMode     = "artist";   // "artist" | "genre"
let selectedGroup = null;
let artistColorMap = {};
let clusterLabels  = {};        // cid → [primaryArtist, second, third]
let globalMinSize, globalMaxSize;
let inited = false;

// ── Size scaling (global — consistent across artist/genre switch) ────────────

function scaleSize(s) {
  return 4 + ((s - globalMinSize) / (globalMaxSize - globalMinSize || 1)) * 20;
}

// ── Helpers ─────────────────────────────────────────────────────────────────

function rankArtists(records, n) {
  const c = {};
  for (const r of records) c[r.artist] = (c[r.artist] || 0) + 1;
  return Object.entries(c).sort((a, b) => b[1] - a[1]).slice(0, n).map(([a]) => a);
}

// Returns { cid: [top1, top2, top3] }
function buildClusterLabels(records) {
  const cc = {};
  for (const r of records) {
    if (!cc[r.artist_cluster]) cc[r.artist_cluster] = {};
    cc[r.artist_cluster][r.artist] = (cc[r.artist_cluster][r.artist] || 0) + 1;
  }
  const labels = {};
  for (const [cid, ac] of Object.entries(cc)) {
    labels[cid] = Object.entries(ac)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 3)
      .map(([a]) => a);
  }
  return labels;
}

// ── Trace building ──────────────────────────────────────────────────────────

function makeTrace(recs, color, opacity, isOther) {
  return {
    type: "scatter", mode: "markers",
    x: recs.map(r => r.x),
    y: recs.map(r => r.y),
    customdata: recs.map(r => ({
      title: r.title, artist: r.artist,
      play_count: r.play_count,
      color: isOther ? OTHER_COLOR : color,
    })),
    hoverinfo: "none",
    marker: {
      color,
      size: recs.map(r => scaleSize(r.log_size)),
      opacity,
      line: { width: isOther ? 0.5 : 1, color: isOther ? "#ccc" : "rgba(0,0,0,0.18)" },
    },
  };
}

function buildTraces(filtered) {
  const isFiltered = selectedGroup !== null;

  if (colorMode === "artist") {
    const colored = {}, other = [];
    for (const r of filtered) {
      const isKnown = !!artistColorMap[r.artist];
      const isSel   = r.artist === selectedGroup;
      if (isFiltered ? isSel : isKnown) (colored[r.artist] = colored[r.artist] || []).push(r);
      else other.push(r);
    }
    const traces = [];
    if (other.length) traces.push(makeTrace(other, OTHER_COLOR, OTHER_OPACITY, true));
    for (const [a, sub] of Object.entries(colored))
      traces.push(makeTrace(sub, artistColorMap[a], 0.88, false));
    return traces;
  }

  // genre / neighborhood mode
  const cids = [...new Set(filtered.map(r => r.artist_cluster))].sort((a, b) => a - b);
  const traces = [];
  if (isFiltered) {
    const other = filtered.filter(r => r.artist_cluster !== selectedGroup);
    if (other.length) traces.push(makeTrace(other, OTHER_COLOR, OTHER_OPACITY, true));
  }
  for (const cid of cids) {
    if (isFiltered && cid !== selectedGroup) continue;
    const sub = filtered.filter(r => r.artist_cluster === cid);
    if (!sub.length) continue;
    traces.push(makeTrace(sub, GENRE_PALETTE[cid % GENRE_PALETTE.length], 0.88, false));
  }
  return traces;
}

// ── Render ───────────────────────────────────────────────────────────────────

function getFiltered() {
  return allRecords.filter(r => r.play_count >= +document.getElementById("plays-slider").value);
}

const BASE_LAYOUT = {
  paper_bgcolor: "#f7f5f0", plot_bgcolor: "#f7f5f0",
  font: { color: "#1a1a2e", size: 11 },
  xaxis: { range: X_RANGE, showgrid: true, gridcolor: "#e8e5e0",
           zeroline: false, showticklabels: false, fixedrange: true },
  yaxis: { range: Y_RANGE, showgrid: true, gridcolor: "#e8e5e0",
           zeroline: false, showticklabels: false, fixedrange: true },
  hovermode: "closest", showlegend: false,
  margin: { t: 10, r: 10, b: 10, l: 10 },
};

function render() {
  const filtered = getFiltered();
  const count = selectedGroup === null ? filtered.length
    : colorMode === "artist"
      ? filtered.filter(r => r.artist === selectedGroup).length
      : filtered.filter(r => r.artist_cluster === selectedGroup).length;

  document.getElementById("track-count").textContent =
    `${count.toLocaleString()} / ${allRecords.length.toLocaleString()} tracks`;

  if (!inited) {
    Plotly.newPlot("plot", buildTraces(filtered), BASE_LAYOUT, { responsive: true, displayModeBar: false });
    inited = true;
    initHover();
  } else {
    Plotly.react("plot", buildTraces(filtered), BASE_LAYOUT);
  }
}

// ── Fixed hover panel ────────────────────────────────────────────────────────

function initHover() {
  const panel = document.getElementById("hover-panel");
  document.getElementById("plot").on("plotly_hover", function(data) {
    const r = data.points[0].customdata;
    document.getElementById("hp-dot").style.background = r.color;
    document.getElementById("hp-title").textContent    = r.title;
    document.getElementById("hp-artist").textContent   = r.artist;
    document.getElementById("hp-plays").textContent    = r.play_count.toLocaleString() + " plays";
    panel.classList.add("has-data");
  });
  document.getElementById("plot").on("plotly_unhover", function() {
    panel.classList.remove("has-data");
  });
}

// ── Sidebar ───────────────────────────────────────────────────────────────────

function makeArtistRow(name, color, count) {
  const row = document.createElement("div");
  row.className  = "grow";
  row.style.color = color;
  row.innerHTML = `<div class="gdot" style="background:${color}"></div>
    <span class="gname">${name}</span>
    <span class="gcnt">${count}</span>`;
  return row;
}

function makeNeighborhoodRow(artists, color, count) {
  const row = document.createElement("div");
  row.className   = "grow gcluster";
  row.style.color = color;
  const [primary, ...rest] = artists;
  row.innerHTML = `
    <div class="gdot" style="background:${color}"></div>
    <div class="gcluster-names">
      <span class="gname">${primary}</span>
      ${rest.map(a => `<span class="gsec">${a}</span>`).join("")}
    </div>
    <span class="gcnt">${count}</span>`;
  return row;
}

function buildSidebar() {
  const list = document.getElementById("group-list");
  list.innerHTML = "";
  selectedGroup  = null;
  const counts   = {};

  if (colorMode === "artist") {
    document.getElementById("group-label").textContent = "Artists";
    for (const r of allRecords) counts[r.artist] = (counts[r.artist] || 0) + 1;
    for (const a of rankArtists(allRecords, TOP_N_ARTISTS)) {
      const row = makeArtistRow(a, artistColorMap[a], counts[a]);
      row.addEventListener("click", () => selectGroup(row, a));
      list.appendChild(row);
    }
  } else {
    document.getElementById("group-label").textContent = `Artist ${NEIGHBORHOOD_LABEL}s`;
    for (const r of allRecords) counts[r.artist_cluster] = (counts[r.artist_cluster] || 0) + 1;
    const cids = [...new Set(allRecords.map(r => r.artist_cluster))].sort((a, b) => a - b);
    for (const cid of cids) {
      const artists = clusterLabels[cid] || [`Cluster ${cid}`];
      const color   = GENRE_PALETTE[cid % GENRE_PALETTE.length];
      const row     = makeNeighborhoodRow(artists, color, counts[cid]);
      row.addEventListener("click", () => selectGroup(row, cid));
      list.appendChild(row);
    }
  }
}

function selectGroup(row, key) {
  if (selectedGroup === key) {
    selectedGroup = null;
    document.querySelectorAll(".grow").forEach(r => r.classList.remove("active"));
  } else {
    selectedGroup = key;
    document.querySelectorAll(".grow").forEach(r => r.classList.remove("active"));
    row.classList.add("active");
  }
  render();
}

// ── Mode toggle ───────────────────────────────────────────────────────────────

document.querySelectorAll(".mode-btn").forEach(btn => btn.addEventListener("click", () => {
  if (btn.dataset.mode === colorMode) return;
  colorMode = btn.dataset.mode;
  document.querySelectorAll(".mode-btn").forEach(b => b.classList.remove("active"));
  btn.classList.add("active");
  buildSidebar();
  render();
}));

// Min-plays filter
document.getElementById("plays-slider").addEventListener("input", function() {
  document.getElementById("plays-val").textContent = this.value;
  render();
});

// ── Bootstrap ────────────────────────────────────────────────────────────────

fetch(DATA_PATH)
  .then(res => { if (!res.ok) throw new Error(`HTTP ${res.status}`); return res.json(); })
  .then(data => {
    allRecords = data;

    const sizes = allRecords.map(r => r.log_size);
    globalMinSize = Math.min(...sizes);
    globalMaxSize = Math.max(...sizes);

    const topArtists = rankArtists(allRecords, TOP_N_ARTISTS);
    topArtists.forEach((a, i) => artistColorMap[a] = ARTIST_PALETTE[i]);
    clusterLabels = buildClusterLabels(allRecords);

    // Set button label and slider max from data
    document.getElementById("genre-btn").textContent = NEIGHBORHOOD_LABEL;
    document.getElementById("plays-slider").max = Math.max(...allRecords.map(r => r.play_count));

    document.getElementById("loading").style.display = "none";
    buildSidebar();
    render();
  })
  .catch(err => {
    document.getElementById("loading").innerHTML =
      `<div style="text-align:center;color:#c00;font-size:0.8rem">
        Failed to load data: ${err.message}<br><br>
        Run: python processing/build_viz_data.py
      </div>`;
  });
