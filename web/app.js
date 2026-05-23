const DATA_PATH = "../data/processed/viz_data.json";

// Plotly layout shared between initial render and updates
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

// Qualitative palette for clusters; cluster -1 (noise) gets a neutral grey
const CLUSTER_COLORS = [
  "#7c6fcd", "#e05c5c", "#5cae8a", "#d9a03c", "#5b9ed6",
  "#cc7ad4", "#7ab55c", "#d4825b", "#5abccc", "#c45c8c",
  "#9dc45c", "#5c7acd", "#c4b05c", "#8c5ccd", "#5cc4a0",
];
const NOISE_COLOR = "#3a3a5a";

function clusterColor(id) {
  if (id === -1) return NOISE_COLOR;
  return CLUSTER_COLORS[id % CLUSTER_COLORS.length];
}

// Scale log_size values into Plotly marker pixel sizes
function scaleSizes(logSizes) {
  const min = Math.min(...logSizes);
  const max = Math.max(...logSizes);
  const range = max - min || 1;
  return logSizes.map(s => 4 + ((s - min) / range) * 20);
}

function buildHoverText(record) {
  const cluster = record.cluster === -1 ? "noise" : `cluster ${record.cluster}`;
  return (
    `<b>${record.title}</b><br>` +
    `${record.artist}<br>` +
    `${record.play_count.toLocaleString()} plays<br>` +
    `${cluster}`
  );
}

let allRecords = [];
let initialized = false;

function getUniqueClusterIds(records) {
  return [...new Set(records.map(r => r.cluster))].sort((a, b) => a - b);
}

function buildTraces(records) {
  const clusterIds = getUniqueClusterIds(records);
  return clusterIds.map(cid => {
    const subset = records.filter(r => r.cluster === cid);
    const label = cid === -1 ? "noise" : `cluster ${cid}`;
    return {
      type: "scatter",
      mode: "markers",
      name: label,
      x: subset.map(r => r.x),
      y: subset.map(r => r.y),
      text: subset.map(buildHoverText),
      hovertemplate: "%{text}<extra></extra>",
      marker: {
        color: clusterColor(cid),
        size: scaleSizes(subset.map(r => r.log_size)),
        opacity: 0.82,
        line: { width: 0.5, color: "rgba(255,255,255,0.08)" },
      },
      customdata: subset.map(r => r.play_count),
    };
  });
}

function filterRecords(clusterFilter, minPlays) {
  return allRecords.filter(r => {
    if (clusterFilter !== "all" && r.cluster !== parseInt(clusterFilter, 10)) return false;
    if (r.play_count < minPlays) return false;
    return true;
  });
}

function updateStats(count) {
  document.getElementById("stats").textContent = `${count.toLocaleString()} tracks`;
}

function populateClusterDropdown(records) {
  const select = document.getElementById("cluster-filter");
  const ids = getUniqueClusterIds(records);
  ids.forEach(cid => {
    const opt = document.createElement("option");
    opt.value = cid;
    opt.textContent = cid === -1 ? "noise (unclustered)" : `cluster ${cid}`;
    select.appendChild(opt);
  });
}

function setPlaysSliderMax(records) {
  const max = Math.max(...records.map(r => r.play_count));
  const slider = document.getElementById("plays-filter");
  // Snap to a readable ceiling
  slider.max = Math.ceil(max / 10) * 10;
}

function render() {
  const clusterFilter = document.getElementById("cluster-filter").value;
  const minPlays = parseInt(document.getElementById("plays-filter").value, 10);
  const filtered = filterRecords(clusterFilter, minPlays);
  const traces = buildTraces(filtered);
  updateStats(filtered.length);

  if (!initialized) {
    Plotly.newPlot("plot", traces, LAYOUT, CONFIG);
    initialized = true;
  } else {
    Plotly.react("plot", traces, LAYOUT, CONFIG);
  }
}

// Wire up controls
document.getElementById("cluster-filter").addEventListener("change", render);

document.getElementById("plays-filter").addEventListener("input", function () {
  document.getElementById("plays-value").textContent = this.value;
  render();
});

// Load data and bootstrap
fetch(DATA_PATH)
  .then(res => {
    if (!res.ok) throw new Error(`HTTP ${res.status}: ${res.url}`);
    return res.json();
  })
  .then(data => {
    allRecords = data;
    document.getElementById("loading").style.display = "none";
    populateClusterDropdown(allRecords);
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
