const DATA_PATH = "../data/processed/viz_data.json";

const TOP_N_ARTISTS      = 50;
const OTHER_COLOR        = "#4a4a5a";
const OTHER_OPACITY      = 0.60;
const N_NEIGHBORS        = 15;
const NEIGHBORHOOD_LABEL = "Neighborhood";

const ALL_FEATURE_KEYS   = ["acousticness","danceability","energy","instrumentalness",
                            "liveness","speechiness","tempo","valence",
                            "loudness"];
const ALL_FEATURE_LABELS = ["Acousticness","Danceability","Energy","Instrumentalness",
                            "Liveness","Speechiness","Tempo","Valence",
                            "Loudness"];
const DEFAULT_FEATURE_KEYS = ["acousticness","danceability","energy","instrumentalness",
                              "liveness","speechiness","tempo","valence"];

const GENRE_PALETTE = [
  "#00eaff", "#ff003c", "#00ff88", "#bf00ff", "#ff9900", "#00ffcc",
  "#ff007f", "#c8ff00", "#00eaff", "#ff003c", "#00ff88", "#bf00ff",
];

// Sunset colorscale: deep night → violet → red → orange → cream
const TOPO_COLORSCALE = [
  [0,    "#1a0533"],
  [0.25, "#5c1a6e"],
  [0.5,  "#c0394b"],
  [0.75, "#f28a30"],
  [1,    "#fde8c8"],
];

function makeArtistPalette(n) {
  return Array.from({ length: n }, (_, i) => {
    const h = (i * 137.508) % 360;
    const s = 95 + (i % 2) * 5;
    const l = 52 + (i % 3) * 6;
    return `hsl(${h.toFixed(1)},${s}%,${l}%)`;
  });
}

const ARTIST_PALETTE = makeArtistPalette(TOP_N_ARTISTS);

// ── State ─────────────────────────────────────────────────────────────────────

let allRecords     = [];
let colorMode      = "genre";
let selectedGroup  = null;
let artistColorMap = {};
let clusterLabels  = {};
let clusterProfiles = {};
let globalMinSize, globalMaxSize;
let xRange, yRange;
let tempoMin = 60, tempoMax = 200;
let hasFeatureData = false;
let currentK = 8;
let inited = false;
let terrainFeature  = null;
let contourCache    = {};
let featureStats   = {}; // { [key]: { mean, std } } — matches pipeline StandardScaler
let availableFeatures = new Set();
let selectedFeatures  = new Set(DEFAULT_FEATURE_KEYS);
let mapFeatureKeys    = [...DEFAULT_FEATURE_KEYS]; // features the current map was built with
let loudnessMin = -60, loudnessMax = 0;
let umapMod = null; // lazy-loaded umap-js

// Focus mode: sonic neighbors / path
let clickedIndices = [];       // [] | [i] | [i, j]
let focusSet       = new Set();

// ── Feature key helpers ───────────────────────────────────────────────────────

function featureKeys() { return ALL_FEATURE_KEYS.filter(k => selectedFeatures.has(k)); }
function featureLabelFor(k) { return ALL_FEATURE_LABELS[ALL_FEATURE_KEYS.indexOf(k)]; }

// ── Size scaling ──────────────────────────────────────────────────────────────

function scaleSize(s) {
  return 5 + ((s - globalMinSize) / (globalMaxSize - globalMinSize || 1)) * 20;
}

// ── Color helpers ─────────────────────────────────────────────────────────────

function getRecordColor(r) {
  if (colorMode === "artist") return artistColorMap[r.artist] || OTHER_COLOR;
  return GENRE_PALETTE[r.artist_cluster % GENRE_PALETTE.length];
}

// ── Data helpers ──────────────────────────────────────────────────────────────

function rankArtists(records, n) {
  const c = {};
  for (const r of records) c[r.artist] = (c[r.artist] || 0) + 1;
  return Object.entries(c).sort((a, b) => b[1] - a[1]).slice(0, n).map(([a]) => a);
}

function buildClusterLabels(records) {
  const cc = {};
  for (const r of records) {
    if (!cc[r.artist_cluster]) cc[r.artist_cluster] = {};
    cc[r.artist_cluster][r.artist] = (cc[r.artist_cluster][r.artist] || 0) + 1;
  }
  const labels = {};
  for (const [cid, ac] of Object.entries(cc)) {
    labels[cid] = Object.entries(ac).sort((a, b) => b[1] - a[1]).slice(0, 3).map(([a]) => a);
  }
  return labels;
}

function buildClusterProfiles(records) {
  if (!hasFeatureData) return {};
  const fkeys = featureKeys();
  const byCluster = {};
  for (const r of records) {
    const cid = r.artist_cluster;
    if (!byCluster[cid]) byCluster[cid] = [];
    byCluster[cid].push(r);
  }
  const globalMeans = {};
  for (const f of fkeys) {
    globalMeans[f] = records.reduce((s, r) => s + (r[f] || 0), 0) / records.length;
  }
  const profiles = {};
  for (const [cid, recs] of Object.entries(byCluster)) {
    profiles[cid] = { _globalMeans: globalMeans };
    for (const f of fkeys) {
      profiles[cid][f] = recs.reduce((s, r) => s + (r[f] || 0), 0) / recs.length;
    }
  }
  return profiles;
}

function computeRange(vals, pad = 0.02) {
  const mn = Math.min(...vals), mx = Math.max(...vals);
  const d = (mx - mn) * pad;
  return [mn - d, mx + d];
}

function applyData(data) {
  allRecords = data.map((r, i) => ({ ...r, _idx: i }));
  const sizes = allRecords.map(r => r.log_size);
  globalMinSize = Math.min(...sizes);
  globalMaxSize = Math.max(...sizes);
  xRange = computeRange(allRecords.map(r => r.x));
  yRange = computeRange(allRecords.map(r => r.y));
  hasFeatureData = allRecords[0]?.acousticness !== undefined;
  if (hasFeatureData) {
    const tempos = allRecords.map(r => r.tempo);
    tempoMin = Math.min(...tempos);
    tempoMax = Math.max(...tempos);
    if (allRecords[0]?.loudness !== undefined) {
      const ls = allRecords.map(r => r.loudness);
      loudnessMin = Math.min(...ls);
      loudnessMax = Math.max(...ls);
    }
  }
  availableFeatures = new Set(ALL_FEATURE_KEYS.filter(k => allRecords[0]?.[k] !== undefined));
  selectedFeatures  = new Set([...DEFAULT_FEATURE_KEYS].filter(k => availableFeatures.has(k)));
  mapFeatureKeys    = [...selectedFeatures]; // initial map was built with these features
}

// ── In-browser KMeans ─────────────────────────────────────────────────────────

function seededRng(seed) {
  let s = seed >>> 0;
  return () => { s = (Math.imul(1664525, s) + 1013904223) >>> 0; return s / 4294967296; };
}

function kmeans(points, k, seed = 42, maxIter = 300) {
  const rng = seededRng(seed);
  const n = points.length, dims = points[0].length;

  // k-means++ init
  const centroids = [points[Math.floor(rng() * n)].slice()];
  while (centroids.length < k) {
    const dists = points.map(p =>
      Math.min(...centroids.map(c => c.reduce((s, v, i) => s + (v - p[i]) ** 2, 0)))
    );
    const sum = dists.reduce((a, b) => a + b, 0);
    let r = rng() * sum;
    const chosen = dists.findIndex((d) => (r -= d) <= 0);
    centroids.push(points[chosen < 0 ? n - 1 : chosen].slice());
  }

  let labels = new Array(n).fill(0);
  for (let iter = 0; iter < maxIter; iter++) {
    const next = points.map(p => {
      let best = 0, bestD = Infinity;
      for (let c = 0; c < k; c++) {
        const d = centroids[c].reduce((s, v, i) => s + (v - p[i]) ** 2, 0);
        if (d < bestD) { bestD = d; best = c; }
      }
      return best;
    });
    if (next.every((l, i) => l === labels[i])) break;
    labels = next;
    for (let c = 0; c < k; c++) {
      const members = points.filter((_, i) => labels[i] === c);
      if (!members.length) continue;
      for (let j = 0; j < dims; j++)
        centroids[c][j] = members.reduce((s, p) => s + p[j], 0) / members.length;
    }
  }
  return { labels, centroids };
}

function recomputeArtistClusters(k) {
  const fkeys = featureKeys();
  if (!fkeys.length) return;
  const means = fkeys.map(f => allRecords.reduce((s, r) => s + (r[f] || 0), 0) / allRecords.length);
  const stds  = fkeys.map((f, i) => {
    const v = allRecords.reduce((s, r) => s + ((r[f] || 0) - means[i]) ** 2, 0) / allRecords.length;
    return Math.sqrt(v) || 1;
  });
  const normVec = r => fkeys.map((f, i) => ((r[f] || 0) - means[i]) / stds[i]);

  // Group records by artist
  const byArtist = {};
  for (const r of allRecords) (byArtist[r.artist] = byArtist[r.artist] || []).push(r);

  const qualified   = Object.keys(byArtist).filter(a => byArtist[a].length >= 3);
  const unqualified = Object.keys(byArtist).filter(a => byArtist[a].length <  3);

  // Mean normalized feature vector per qualified artist
  const profiles = qualified.map(a => {
    const vecs = byArtist[a].map(normVec);
    return fkeys.map((_, i) => vecs.reduce((s, v) => s + v[i], 0) / vecs.length);
  });

  const { labels, centroids } = kmeans(profiles, k);

  const artistCluster = {};
  qualified.forEach((a, i) => { artistCluster[a] = labels[i]; });

  // Assign tail artists (< 3 tracks) to nearest centroid
  for (const a of unqualified) {
    const vec = normVec(byArtist[a][0]);
    let best = 0, bestD = Infinity;
    for (let c = 0; c < k; c++) {
      const d = centroids[c].reduce((s, v, i) => s + (v - vec[i]) ** 2, 0);
      if (d < bestD) { bestD = d; best = c; }
    }
    artistCluster[a] = best;
  }

  for (const r of allRecords) r.artist_cluster = artistCluster[r.artist] ?? 0;
  clusterLabels   = buildClusterLabels(allRecords);
  clusterProfiles = buildClusterProfiles(allRecords);
}

// ── Sonic neighbors ───────────────────────────────────────────────────────────

function getSonicNeighbors(idx, n) {
  const r = allRecords[idx];
  return allRecords
    .map((s, i) => ({ i, d: Math.hypot(s.x - r.x, s.y - r.y) }))
    .sort((a, b) => a.d - b.d)
    .slice(1, n + 1)
    .map(x => x.i);
}

// ── Feature z-score stats (mirrors pipeline StandardScaler) ──────────────────

const TERRAIN_SIGMA = 3; // ±3σ maps to [0, 1]; clipped beyond that

function computeFeatureStats() {
  featureStats = {};
  for (const key of ALL_FEATURE_KEYS) {
    const vals = allRecords.map(r => r[key] ?? 0);
    const mean = vals.reduce((a, b) => a + b, 0) / vals.length;
    const std  = Math.sqrt(vals.reduce((s, v) => s + (v - mean) ** 2, 0) / vals.length) || 1;
    featureStats[key] = { mean, std };
  }
}

function normalizeForTerrain(key, val) {
  const { mean, std } = featureStats[key] || { mean: 0, std: 1 };
  const z = (val - mean) / std;
  return Math.max(0, Math.min(1, (z + TERRAIN_SIGMA) / (2 * TERRAIN_SIGMA)));
}

// ── Terrain contour ───────────────────────────────────────────────────────────

function buildContourData(featureKey) {
  const GRID = 50;
  const xStep = (xRange[1] - xRange[0]) / GRID;
  const yStep = (yRange[1] - yRange[0]) / GRID;
  const xGrid = Array.from({ length: GRID }, (_, i) => xRange[0] + (i + 0.5) * xStep);
  const yGrid = Array.from({ length: GRID }, (_, j) => yRange[0] + (j + 0.5) * yStep);
  // Inverse-distance-weighting (p=2) over all songs per grid cell
  const raw = yGrid.map(gy =>
    xGrid.map(gx => {
      let wSum = 0, vSum = 0;
      for (const r of allRecords) {
        const d2 = (r.x - gx) ** 2 + (r.y - gy) ** 2;
        const w  = d2 < 1e-10 ? 1e10 : 1 / d2;
        wSum += w;
        vSum += w * normalizeForTerrain(featureKey, r[featureKey] ?? 0);
      }
      return vSum / wSum;
    })
  );

  // IDW averaging compresses values toward the mean — stretch the interpolated
  // grid's actual range to [0,1] so the full colorscale is always used.
  const flat = raw.flat();
  const gMin = Math.min(...flat);
  const gMax = Math.max(...flat);
  const span = gMax - gMin || 1;
  const zMatrix = raw.map(row => row.map(v => (v - gMin) / span));

  return { xGrid, yGrid, zMatrix };
}

function makeContourTrace(featureKey) {
  if (!hasFeatureData || !featureKey) return null;
  if (!contourCache[featureKey]) contourCache[featureKey] = buildContourData(featureKey);
  const { xGrid, yGrid, zMatrix } = contourCache[featureKey];
  return {
    type: "contour",
    x: xGrid, y: yGrid, z: zMatrix,
    colorscale: TOPO_COLORSCALE,
    showscale: false,
    hoverinfo: "skip",
    opacity: 0.72,
    zmin: 0, zmax: 1,
    contours: { coloring: "heatmap", showlines: true, size: 0.1 },
    line: { width: 1.6, color: "rgba(0,0,0,0.38)" },
  };
}

// ── Trace building ─────────────────────────────────────────────────────────────

function makeCustomdata(r) {
  return {
    title: r.title, artist: r.artist, play_count: r.play_count,
    color: getRecordColor(r),
    idx: r._idx,
    feats: hasFeatureData ? featureKeys().map(k => r[k]) : null,
  };
}

function makeTrace(recs, color, opacity, isOther) {
  return {
    type: "scatter", mode: "markers",
    x: recs.map(r => r.x),
    y: recs.map(r => r.y),
    customdata: recs.map(r => makeCustomdata(r)),
    hoverinfo: "none",
    marker: {
      color,
      size: recs.map(r => scaleSize(r.log_size)),
      opacity,
      line: { width: isOther ? 0.5 : 1, color: isOther ? "#ccc" : "rgba(0,0,0,0.18)" },
    },
  };
}

function makePinnedTrace(r) {
  return {
    type: "scatter", mode: "markers",
    x: [r.x], y: [r.y],
    customdata: [makeCustomdata(r)],
    hoverinfo: "none",
    marker: {
      color: getRecordColor(r),
      size: [scaleSize(r.log_size) + 5],
      opacity: 1,
      line: { width: 2.5, color: "#1a1a2e" },
    },
  };
}

function buildTraces(filtered) {
  const traces = [];

  // Terrain layer always renders first (bottom of stack)
  const contour = makeContourTrace(terrainFeature);
  if (contour) traces.push(contour);

  const isFocus = clickedIndices.length > 0;

  if (isFocus) {
    const clickedSet = new Set(clickedIndices);
    const dim         = filtered.filter(r => !focusSet.has(r._idx) && !clickedSet.has(r._idx));
    const highlighted = filtered.filter(r =>  focusSet.has(r._idx) && !clickedSet.has(r._idx));
    const pinned      = clickedIndices.map(i => allRecords[i]);

    if (dim.length) traces.push(makeTrace(dim, OTHER_COLOR, 0.22, true));

    // Render highlighted in their natural group colors
    const byColor = {};
    for (const r of highlighted) {
      const c = getRecordColor(r);
      (byColor[c] = byColor[c] || []).push(r);
    }
    for (const [color, recs] of Object.entries(byColor)) {
      traces.push(makeTrace(recs, color, 0.9, false));
    }
    for (const r of pinned) traces.push(makePinnedTrace(r));
    return traces;
  }

  // Normal mode
  const isFiltered = selectedGroup !== null;

  if (colorMode === "artist") {
    const colored = {}, other = [];
    for (const r of filtered) {
      const isKnown = !!artistColorMap[r.artist];
      const isSel   = r.artist === selectedGroup;
      if (isFiltered ? isSel : isKnown) (colored[r.artist] = colored[r.artist] || []).push(r);
      else other.push(r);
    }
    if (other.length) traces.push(makeTrace(other, OTHER_COLOR, OTHER_OPACITY, true));
    for (const [a, sub] of Object.entries(colored))
      traces.push(makeTrace(sub, artistColorMap[a], 0.88, false));
    return traces;
  }

  // Neighborhood mode
  const cids = [...new Set(filtered.map(r => r.artist_cluster))].sort((a, b) => a - b);
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

// ── Layout & render ───────────────────────────────────────────────────────────

function makeLayout() {
  return {
    paper_bgcolor: "#f7f5f0", plot_bgcolor: "#f7f5f0",
    font: { color: "#1a1a2e", size: 11 },
    xaxis: { range: xRange, showgrid: true, gridcolor: "#e8e5e0",
             zeroline: false, showticklabels: false, fixedrange: true },
    yaxis: { range: yRange, showgrid: true, gridcolor: "#e8e5e0",
             zeroline: false, showticklabels: false, fixedrange: true },
    dragmode: false,
    hovermode: "closest", showlegend: false,
    margin: { t: 10, r: 10, b: 10, l: 10 },
  };
}

function getFiltered() { return allRecords; }

function render() {
  const filtered = getFiltered();
  const count = selectedGroup === null ? filtered.length
    : colorMode === "artist"
      ? filtered.filter(r => r.artist === selectedGroup).length
      : filtered.filter(r => r.artist_cluster === selectedGroup).length;

  document.getElementById("track-count").textContent =
    `${allRecords.length.toLocaleString()} tracks`;

  if (!inited) {
    Plotly.newPlot("plot", buildTraces(filtered), makeLayout(),
      { responsive: true, displayModeBar: false });
    inited = true;
    initHover();
    initClick();
  } else {
    Plotly.react("plot", buildTraces(filtered), makeLayout());
  }
  updateNeighborhoodInfo();
}

// ── Hover panel ────────────────────────────────────────────────────────────────

function normalizeFeature(key, val) {
  if (val == null) return 0;
  if (key === "tempo")         return Math.max(0, Math.min(1, (val - tempoMin) / (tempoMax - tempoMin || 1)));
  if (key === "loudness")      return Math.max(0, Math.min(1, (val - loudnessMin) / (loudnessMax - loudnessMin || 1)));
  if (key === "timeSignature") return Math.max(0, Math.min(1, (val - 1) / 6));
  if (key === "key")           return val / 11;
  return Math.max(0, Math.min(1, val));
}

function renderFeatureBars(containerId, feats, globalMeans, accentColor) {
  const el = document.getElementById(containerId);
  if (!hasFeatureData || !feats) { el.innerHTML = ""; return; }
  el.style.setProperty("--bar-color", accentColor || "#999");
  const keys = featureKeys();
  el.innerHTML = keys.map((k, i) => {
    const pct = (normalizeFeature(k, feats[i]) * 100).toFixed(0);
    const avgPct = globalMeans
      ? (normalizeFeature(k, globalMeans[k]) * 100).toFixed(0) : null;
    return `<div class="feat-row">
      <span class="feat-lbl">${featureLabelFor(k)}</span>
      <div class="feat-bar-bg">
        <div class="feat-bar-fill" style="width:${pct}%"></div>
        ${avgPct !== null ? `<div class="feat-bar-avg" style="left:${avgPct}%"></div>` : ""}
      </div>
    </div>`;
  }).join("");
}

function updateClickHint() {
  const el = document.getElementById("hp-click-hint");
  if (!el) return;
  el.textContent = clickedIndices.length === 0
    ? "Click · sonic neighbors"
    : "Click again · clear";
}

function initHover() {
  const panel     = document.getElementById("hover-panel");
  const plotDiv   = document.getElementById("plot");
  const pulseRing = document.getElementById("pulse-ring");

  plotDiv.on("plotly_hover", function(data) {
    const r = data.points[0].customdata;
    document.getElementById("hp-dot").style.background = r.color;
    document.getElementById("hp-title").textContent    = r.title;
    document.getElementById("hp-artist").textContent   = r.artist;
    document.getElementById("hp-plays").textContent    = r.play_count.toLocaleString() + " plays";
    renderFeatureBars("hp-features", r.feats, null, r.color);
    panel.classList.add("has-data");
    updateClickHint();

    const la = plotDiv._fullLayout;
    const px = la.margin.l + la.xaxis.l2p(data.points[0].x);
    const py = la.margin.t + la.yaxis.l2p(data.points[0].y);
    pulseRing.style.left        = px + "px";
    pulseRing.style.top         = py + "px";
    pulseRing.style.borderColor = r.color;
    pulseRing.style.display     = "block";
  });

  plotDiv.on("plotly_unhover", function() {
    panel.classList.remove("has-data");
    pulseRing.style.display = "none";
  });
}

// ── Click / focus ─────────────────────────────────────────────────────────────

function clearFocus() {
  if (!clickedIndices.length) return;
  clickedIndices = [];
  focusSet = new Set();
  render();
  updateClickHint();
}

function initClick() {
  const plotDiv = document.getElementById("plot");
  let pointJustClicked = false;

  plotDiv.on("plotly_click", function(data) {
    if (window.matchMedia("(max-width: 640px)").matches) return;
    pointJustClicked = true;

    if (!data.points || !data.points.length) { clearFocus(); return; }

    const idx = data.points[0].customdata.idx;
    if (clickedIndices.length === 1 && clickedIndices[0] === idx) {
      clearFocus();
    } else {
      clickedIndices = [idx];
      focusSet = new Set(getSonicNeighbors(idx, N_NEIGHBORS));
      render();
      updateClickHint();
    }
  });

  // Fallback: native click fires before plotly_click, so defer the check.
  // If plotly_click didn't set the flag by then, the click was on empty space.
  plotDiv.addEventListener("click", () => {
    if (window.matchMedia("(max-width: 640px)").matches) return;
    setTimeout(() => {
      if (!pointJustClicked) clearFocus();
      pointJustClicked = false;
    }, 0);
  });
}


// ── Neighborhood info ─────────────────────────────────────────────────────────

function updateNeighborhoodInfo() {
  const panel = document.getElementById("neighborhood-info");
  if (!panel) return;

  if (!hasFeatureData || colorMode !== "genre" || selectedGroup === null) {
    panel.style.display = "none";
    return;
  }

  const profile = clusterProfiles[selectedGroup];
  if (!profile) { panel.style.display = "none"; return; }

  panel.style.display = "block";
  const color = GENRE_PALETTE[selectedGroup % GENRE_PALETTE.length];
  const feats = featureKeys().map(k => profile[k]);
  renderFeatureBars("ni-bars", feats, profile._globalMeans, color);
}

// ── Sidebar ───────────────────────────────────────────────────────────────────

function makeArtistRow(name, color, count) {
  const row = document.createElement("div");
  row.className   = "grow";
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

  document.getElementById("k-slider-wrap").style.display = colorMode === "genre" ? "block" : "none";

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
  updateNeighborhoodInfo();
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

// ── Mode toggle ────────────────────────────────────────────────────────────────

document.querySelectorAll(".mode-btn").forEach(btn => btn.addEventListener("click", () => {
  if (btn.dataset.mode === colorMode) return;
  colorMode = btn.dataset.mode;
  document.querySelectorAll(".mode-btn").forEach(b => b.classList.remove("active"));
  btn.classList.add("active");
  buildSidebar();
  render();
}));

// ── Neighborhood count slider ─────────────────────────────────────────────────

document.getElementById("k-slider").addEventListener("input", function() {
  currentK = +this.value;
  document.getElementById("k-val").textContent = currentK;
  recomputeArtistClusters(currentK);
  buildSidebar();
  render();
});

// ── Terrain legend ────────────────────────────────────────────────────────────

function updateTerrainLegend(featureKey) {
  const legend = document.getElementById("terrain-legend");
  if (!featureKey) { legend.classList.remove("visible"); return; }
  document.getElementById("terrain-legend-label").textContent =
    featureLabelFor(featureKey) || featureKey;
  const gradient = TOPO_COLORSCALE
    .map(([stop, color]) => `${color} ${(stop * 100).toFixed(0)}%`)
    .join(", ");
  document.getElementById("terrain-legend-bar").style.background =
    `linear-gradient(to right, ${gradient})`;
  legend.classList.add("visible");
}

// ── Terrain select ────────────────────────────────────────────────────────────

document.getElementById("terrain-select").addEventListener("change", function() {
  terrainFeature = this.value || null;
  updateTerrainLegend(terrainFeature);
  render();
});

// ── Info modal ────────────────────────────────────────────────────────────────

document.getElementById("info-btn").addEventListener("click", () => {
  document.getElementById("info-overlay").classList.add("open");
});
document.getElementById("info-close").addEventListener("click", () => {
  document.getElementById("info-overlay").classList.remove("open");
  localStorage.setItem("hasVisited", "1");
});
document.getElementById("info-overlay").addEventListener("click", function(e) {
  if (e.target === this) {
    this.classList.remove("open");
    localStorage.setItem("hasVisited", "1");
  }
});

if (!localStorage.getItem("hasVisited")) {
  document.getElementById("info-overlay").classList.add("open");
}

// ── Feature selector ─────────────────────────────────────────────────────────

function syncTerrainOptions() {
  const sel = document.getElementById("terrain-select");
  const prev = sel.value;
  sel.innerHTML = '<option value="">Off</option>' +
    featureKeys().map(k => `<option value="${k}">${featureLabelFor(k)}</option>`).join("");
  // Restore previous selection if still valid, else reset
  if (prev && featureKeys().includes(prev)) {
    sel.value = prev;
  } else if (prev) {
    sel.value = "";
    terrainFeature = null;
    updateTerrainLegend(null);
  }
}

function buildFeatureSelector() {
  const wrap = document.getElementById("feature-wrap");
  const grid = document.getElementById("feature-grid");
  if (!hasFeatureData || availableFeatures.size === 0) { wrap.style.display = "none"; return; }
  wrap.style.display = "block";

  grid.innerHTML = ALL_FEATURE_KEYS
    .filter(k => availableFeatures.has(k))
    .map(k => {
      const label   = featureLabelFor(k);
      const checked = selectedFeatures.has(k) ? "checked" : "";
      return `<label class="feat-chip">
        <input type="checkbox" value="${k}" ${checked}>
        <span>${label}</span>
      </label>`;
    }).join("");

  syncTerrainOptions();

  grid.querySelectorAll("input[type=checkbox]").forEach(cb => {
    cb.addEventListener("change", () => {
      if (cb.checked) {
        selectedFeatures.add(cb.value);
      } else {
        if (selectedFeatures.size <= 1) { cb.checked = true; return; }
        selectedFeatures.delete(cb.value);
      }
      onFeaturesChanged();
    });
  });
}

function onFeaturesChanged() {
  computeFeatureStats();
  contourCache = {};
  recomputeArtistClusters(currentK);
  clusterLabels   = buildClusterLabels(allRecords);
  clusterProfiles = buildClusterProfiles(allRecords);
  syncTerrainOptions();
  checkMapStaleness();
  render();
}

// ── Map recalculation ─────────────────────────────────────────────────────────

function checkMapStaleness() {
  const fkeys = featureKeys();
  const same  = fkeys.length === mapFeatureKeys.length &&
                fkeys.every((k, i) => k === mapFeatureKeys[i]);
  document.getElementById("recalc-wrap").style.display = same ? "none" : "block";
}

function standardScale(records, keys) {
  const n = records.length;
  const means = keys.map(k => records.reduce((s, r) => s + (r[k] ?? 0), 0) / n);
  const stds  = keys.map((k, i) => {
    const v = records.reduce((s, r) => s + ((r[k] ?? 0) - means[i]) ** 2, 0) / n;
    return Math.sqrt(v) || 1;
  });
  return records.map(r => keys.map((k, i) => ((r[k] ?? 0) - means[i]) / stds[i]));
}

async function recalculateMap() {
  const fkeys   = featureKeys();
  const btn     = document.getElementById("recalc-btn");
  const barWrap = document.getElementById("recalc-bar-wrap");
  const bar     = document.getElementById("recalc-bar");

  btn.disabled    = true;
  btn.textContent = "Recalculating…";
  barWrap.style.display = "block";
  bar.style.width = "0%";

  try {
    const normalized = standardScale(allRecords, fkeys);
    const nEpochs    = 100;

    if (!umapMod) umapMod = await import("https://esm.sh/umap-js@1.3.3");
    const { UMAP } = umapMod;

    const umap = new UMAP({
      nNeighbors: 10,
      minDist: 0.05,
      nComponents: 2,
      nEpochs,
      random: seededRng(42),
    });

    const embedding = await umap.fitAsync(normalized, async (epoch) => {
      bar.style.width = `${Math.round(((epoch + 1) / nEpochs) * 100)}%`;
      if (epoch % 5 === 0) await new Promise(r => setTimeout(r, 0));
    });

    for (let i = 0; i < allRecords.length; i++) {
      allRecords[i].x = embedding[i][0];
      allRecords[i].y = embedding[i][1];
    }
    xRange = computeRange(allRecords.map(r => r.x));
    yRange = computeRange(allRecords.map(r => r.y));
    contourCache = {};
    mapFeatureKeys = [...fkeys];

    clickedIndices = [];
    focusSet = new Set();
    document.getElementById("hover-panel").classList.remove("has-data");

    recomputeArtistClusters(currentK);
    clusterLabels   = buildClusterLabels(allRecords);
    clusterProfiles = buildClusterProfiles(allRecords);

    barWrap.style.display = "none";
    document.getElementById("recalc-wrap").style.display = "none";
    render();
  } catch (err) {
    btn.textContent = "Recalculate map";
    barWrap.style.display = "none";
    console.error("UMAP recalculation failed:", err);
  }

  btn.disabled    = false;
  btn.textContent = "Recalculate map";
}

document.getElementById("recalc-btn").addEventListener("click", recalculateMap);

// ── Bootstrap ──────────────────────────────────────────────────────────────────

fetch(DATA_PATH)
  .then(res => { if (!res.ok) throw new Error(`HTTP ${res.status}`); return res.json(); })
  .then(data => {
    applyData(data);
    computeFeatureStats();
    contourCache = {};

    const topArtists = rankArtists(allRecords, TOP_N_ARTISTS);
    topArtists.forEach((a, i) => artistColorMap[a] = ARTIST_PALETTE[i]);
    clusterLabels   = buildClusterLabels(allRecords);
    clusterProfiles = buildClusterProfiles(allRecords);

    document.getElementById("genre-btn").textContent = NEIGHBORHOOD_LABEL;

    document.getElementById("loading").style.display = "none";
    if (hasFeatureData) document.getElementById("terrain-wrap").style.display = "block";
    buildSidebar();
    buildFeatureSelector();
    if (hasFeatureData && availableFeatures.size > 0) {
      const keys = [...availableFeatures];
      terrainFeature = keys[Math.floor(Math.random() * keys.length)];
      const sel = document.getElementById("terrain-select");
      if (sel) sel.value = terrainFeature;
      updateTerrainLegend(terrainFeature);
    }
    render();
    // Warm up umap-js in the background so first Recalculate is fast
    if (hasFeatureData) import("https://esm.sh/umap-js@1.3.3").then(m => { umapMod = m; }).catch(() => {});
  })
  .catch(err => {
    document.getElementById("loading").innerHTML =
      `<div style="text-align:center;color:#c00;font-size:0.8rem">
        Failed to load data: ${err.message}<br><br>
        Run: python processing/build_viz_data.py
      </div>`;
  });
