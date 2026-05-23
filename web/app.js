const DATA_PATH = "../data/processed/viz_data.json";

// How many top artists (by track count) get a distinct color
const TOP_N_ARTISTS = 15;

const LAYOUT = {
  paper_bgcolor: "#0e0e12",
  plot_bgcolor: "#0e0e12",
  font: { color: "#c8c8e0", size: 12 },
  xaxis: {
    showgrid: true,
    gridcolor: "#1a1a26",
    zeroline: false,
    showticklabels: false,
    title: "",
  },
  yaxis: {
    showgrid: true,
    gridcolor: "#1a1a26",
    zeroline: false,
    showticklabels: false,
    title: "",
  },
  hovermode: "closest",
  margin: { t: 10, r: 10, b: 10, l: 10 },
  legend: {
    bgcolor: "#14141e",
    bordercolor: "#2a2a3e",
    borderwidth: 1,
    font: { size: 11, color: "#a0a0c0" },
  },
};

const CONFIG = {
  responsive: true,
  displayModeBar: true,
  modeBarButtonsToRemove: ["select2d", "lasso2d", "autoScale2d"],
  displaylogo: false,
};

// 15 distinct colors for top artists; "other" gets a muted grey
const ARTIST_PALETTE = [
  "#e05c5c", "#5cae8a", "#d9a03c", "#5b9ed6", "#cc7ad4",
  "#e8845a", "#5abccc", "#c45c8c", "#9dc45c", "#7c6fcd",
  "#c4b05c", "#5cc4a0", "#d45c7a", "#7ab55c", "#8c5ccd",
];
const OTHER_COLOR = "#333350";

// Scale log_size values into Plotly marker pixel sizes
function scaleSizes(logSizes) {
  const min = Math.min(...logSizes);
  const max = Math.max(...logSizes);
  const range = max - min || 1;
  return logSizes.map(s => 4 + ((s - min) / range) * 20);
}

function buildHoverText(record, artistLabel) {
  const cluster = record.cluster === -1 ? "noise" : `cluster ${record.cluster}`;
  return (
    `<b>${record.title}</b><br>` +
    `${record.artist}<br>` +
    `${record.play_count.toLocaleString()} plays<br>` +
    `${cluster}`
  );
}

// Rank artists by track count descending, return top-N set
function topArtistSet(records, n) {
  const counts = {};
  for (const r of records) counts[r.artist] = (counts[r.artist] || 0) + 1;
  return new Set(
    Object.entries(counts)
      .sort((a, b) => b[1] - a[1])
      .slice(0, n)
      .map(([artist]) => artist)
  );
}

function buildTraces(records, topArtists, artistColorMap) {
  // Group records by display label (artist name or "other")
  const groups = {};
  for (const r of records) {
    const label = topArtists.has(r.artist) ? r.artist : "other";
    if (!groups[label]) groups[label] = [];
    groups[label].push(r);
  }

  const traces = [];

  // Top artists first (sorted by track count descending for legend order)
  const sortedArtists = [...topArtists].sort(
    (a, b) =>
      (groups[b] || []).length - (groups[a] || []).length
  );

  for (const artist of sortedArtists) {
    const subset = groups[artist] || [];
    if (subset.length === 0) continue;
    traces.push({
      type: "scatter",
      mode: "markers",
      name: artist,
      x: subset.map(r => r.x),
      y: subset.map(r => r.y),
      text: subset.map(r => buildHoverText(r)),
      hovertemplate: "%{text}<extra></extra>",
      marker: {
        color: artistColorMap[artist],
        size: scaleSizes(subset.map(r => r.log_size)),
        opacity: 0.85,
        line: { width: 0.5, color: "rgba(255,255,255,0.1)" },
      },
    });
  }

  // "other" last so it renders beneath colored points
  const otherSubset = groups["other"] || [];
  if (otherSubset.length > 0) {
    traces.push({
      type: "scatter",
      mode: "markers",
      name: "other",
      x: otherSubset.map(r => r.x),
      y: otherSubset.map(r => r.y),
      text: otherSubset.map(r => buildHoverText(r)),
      hovertemplate: "%{text}<extra></extra>",
      marker: {
        color: OTHER_COLOR,
        size: scaleSizes(otherSubset.map(r => r.log_size)),
        opacity: 0.55,
        line: { width: 0 },
      },
    });
  }

  return traces;
}

function filterRecords(artistFilter, minPlays) {
  return allRecords.filter(r => {
    if (artistFilter !== "all" && r.artist !== artistFilter) return false;
    if (r.play_count < minPlays) return false;
    return true;
  });
}

function updateStats(count) {
  document.getElementById("stats").textContent = `${count.toLocaleString()} tracks`;
}

function populateArtistDropdown(topArtists) {
  const select = document.getElementById("artist-filter");
  [...topArtists]
    .sort((a, b) => a.localeCompare(b))
    .forEach(artist => {
      const opt = document.createElement("option");
      opt.value = artist;
      opt.textContent = artist;
      select.appendChild(opt);
    });
}

function setPlaysSliderMax(records) {
  const max = Math.max(...records.map(r => r.play_count));
  document.getElementById("plays-filter").max = Math.ceil(max / 10) * 10;
}

let allRecords = [];
let topArtists = new Set();
let artistColorMap = {};
let initialized = false;

function render() {
  const artistFilter = document.getElementById("artist-filter").value;
  const minPlays = parseInt(document.getElementById("plays-filter").value, 10);
  const filtered = filterRecords(artistFilter, minPlays);
  const traces = buildTraces(filtered, topArtists, artistColorMap);
  updateStats(filtered.length);

  if (!initialized) {
    Plotly.newPlot("plot", traces, LAYOUT, CONFIG);
    initialized = true;
  } else {
    Plotly.react("plot", traces, LAYOUT, CONFIG);
  }
}

document.getElementById("artist-filter").addEventListener("change", render);

document.getElementById("plays-filter").addEventListener("input", function () {
  document.getElementById("plays-value").textContent = this.value;
  render();
});

fetch(DATA_PATH)
  .then(res => {
    if (!res.ok) throw new Error(`HTTP ${res.status}: ${res.url}`);
    return res.json();
  })
  .then(data => {
    allRecords = data;
    topArtists = topArtistSet(allRecords, TOP_N_ARTISTS);

    // Assign a stable color to each top artist
    [...topArtists]
      .sort((a, b) => {
        const ca = allRecords.filter(r => r.artist === a).length;
        const cb = allRecords.filter(r => r.artist === b).length;
        return cb - ca;
      })
      .forEach((artist, i) => {
        artistColorMap[artist] = ARTIST_PALETTE[i % ARTIST_PALETTE.length];
      });

    document.getElementById("loading").style.display = "none";
    populateArtistDropdown(topArtists);
    setPlaysSliderMax(allRecords);
    render();
  })
  .catch(err => {
    document.getElementById("loading").style.display = "none";
    const el = document.getElementById("error");
    el.style.display = "flex";
    el.textContent =
      `Failed to load data: ${err.message}\n\n` +
      `Run the pipeline first:\n` +
      `  python pipeline/build_dataset.py\n` +
      `  python processing/build_viz_data.py`;
  });
