const sampleSelect = document.getElementById("sampleSelect");
const modelSelect = document.getElementById("modelSelect");
const loadBtn = document.getElementById("loadBtn");
const runStatus = document.getElementById("runStatus");
const runtimeInfo = document.getElementById("runtimeInfo");
const pathLine = document.getElementById("pathLine");
const metricsEl = document.getElementById("metrics");
const pointCount = document.getElementById("pointCount");
const rgbViewBtn = document.getElementById("rgbViewBtn");
const obliqueViewBtn = document.getElementById("obliqueViewBtn");
const topViewBtn = document.getElementById("topViewBtn");
const bevViewBtn = document.getElementById("bevViewBtn");

const imageIds = {
  rgb: "rgbImg",
  sparse_depth: "sparseImg",
  gt: "gtImg",
  ref_pred: "refImg",
  board_pred: "boardImg",
  abs_error: "errImg",
};

let manifest = null;
let currentPointCloud = null;
let currentPointView = "rgb";
let plotlyReady = false;

function currentModel() {
  return modelSelect.value || "completionformer";
}

const pointViewButtons = {
  rgb: rgbViewBtn,
  oblique: obliqueViewBtn,
  top: topViewBtn,
  bev: bevViewBtn,
};

const plotlyCameras = {
  rgb: {
    eye: { x: 0.0, y: 0.0, z: 2.2 },
    up: { x: 0.0, y: 1.0, z: 0.0 },
    center: { x: 0.0, y: 0.0, z: 0.0 },
    projection: { type: "orthographic" },
  },
  oblique: {
    eye: { x: 0.35, y: -1.35, z: 1.45 },
    up: { x: 0.0, y: 0.0, z: 1.0 },
    center: { x: 0.0, y: 0.0, z: 0.0 },
    projection: { type: "perspective" },
  },
  top: {
    eye: { x: 0.0, y: -2.1, z: 0.05 },
    up: { x: 0.0, y: 0.0, z: 1.0 },
    center: { x: 0.0, y: 0.0, z: 0.0 },
    projection: { type: "orthographic" },
  },
};

function metricRow(label, value) {
  const row = document.createElement("div");
  const dt = document.createElement("dt");
  const dd = document.createElement("dd");
  dt.textContent = label;
  dd.textContent = typeof value === "number" ? value.toFixed(5) : value;
  row.appendChild(dt);
  row.appendChild(dd);
  return row;
}

function renderMetrics(metrics) {
  const rows = [
    metricRow("abs mean", metrics.abs_mean),
    metricRow("abs p95", metrics.abs_p95),
    metricRow("rmse", metrics.rmse),
    metricRow("board min", metrics.board_min),
    metricRow("board max", metrics.board_max),
  ];
  if (metrics.latency_total_ms !== undefined) {
    rows.push(metricRow("latency ms", metrics.latency_total_ms));
    rows.push(metricRow("slowest ms", metrics.latency_slowest_ms));
    rows.push(metricRow("slowest op", metrics.latency_slowest_op));
  }
  metricsEl.replaceChildren(...rows);
}

function parseRgb(cssColor) {
  if (cssColor.startsWith("#")) return cssColor;
  const nums = cssColor.match(/\d+/g).map(Number);
  return `rgb(${nums[0]},${nums[1]},${nums[2]})`;
}

function projectedPoint(x, y, z, view) {
  if (view === "rgb") return { u: x, v: y, depth: -z };
  if (view === "top" || view === "bev") return { u: x, v: z, depth: -y };
  const yaw = -0.75;
  const pitch = 0.48;
  const cy = Math.cos(yaw), sy = Math.sin(yaw);
  const cp = Math.cos(pitch), sp = Math.sin(pitch);
  const x1 = cy * x + sy * z;
  const z1 = -sy * x + cy * z;
  const y1 = cp * y - sp * z1;
  const z2 = sp * y + cp * z1;
  return { u: x1, v: y1, depth: z2 };
}

function drawPointCloudCanvas(pc) {
  plotlyReady = false;
  const root = document.getElementById("pointCloud");
  let canvas = root.querySelector("canvas");
  if (!canvas) {
    root.replaceChildren();
    canvas = document.createElement("canvas");
    root.appendChild(canvas);
  }
  const rect = root.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const width = Math.max(320, Math.floor(rect.width));
  const height = Math.max(320, Math.floor(rect.height));
  canvas.width = Math.floor(width * dpr);
  canvas.height = Math.floor(height * dpr);
  canvas.style.width = `${width}px`;
  canvas.style.height = `${height}px`;
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, width, height);

  const projected = [];
  for (let i = 0; i < pc.x.length; i += 1) {
    projected.push(projectedPoint(pc.x[i], pc.y[i], pc.z[i], currentPointView));
  }
  const us = projected.map(p => p.u);
  const vs = projected.map(p => p.v);
  const minU = Math.min(...us), maxU = Math.max(...us);
  const minV = Math.min(...vs), maxV = Math.max(...vs);
  const spanU = Math.max(1e-6, maxU - minU);
  const spanV = Math.max(1e-6, maxV - minV);
  const pad = 22;
  const scale = Math.min((width - pad * 2) / spanU, (height - pad * 2) / spanV);
  const cx = width * 0.5;
  const cy = height * 0.5;
  const midU = (minU + maxU) * 0.5;
  const midV = (minV + maxV) * 0.5;
  const order = projected.map((p, i) => [p.depth, i]).sort((a, b) => b[0] - a[0]);
  const radius = currentPointView === "rgb" ? 1.2 : 1.5;
  for (const [, i] of order) {
    const p = projected[i];
    const px = cx + (p.u - midU) * scale;
    const py = cy - (p.v - midV) * scale;
    ctx.fillStyle = parseRgb(pc.color[i]);
    ctx.fillRect(px - radius * 0.5, py - radius * 0.5, radius, radius);
  }
  ctx.strokeStyle = "#d8dde3";
  ctx.strokeRect(0.5, 0.5, width - 1, height - 1);
}

function renderPointCloudPlotly(pc) {
  const trace = {
    type: "scatter3d",
    mode: "markers",
    x: pc.x,
    y: pc.y,
    z: pc.z,
    marker: { size: 1.6, color: pc.color, opacity: 0.95 },
  };
  const layout = {
    margin: { l: 0, r: 0, t: 0, b: 0 },
    paper_bgcolor: "#ffffff",
    plot_bgcolor: "#ffffff",
    scene: {
      aspectmode: "data",
      xaxis: { showgrid: true, zeroline: false, title: "x" },
      yaxis: { showgrid: true, zeroline: false, title: "y" },
      zaxis: { showgrid: true, zeroline: false, title: "z" },
      camera: plotlyCameras[currentPointView] || plotlyCameras.rgb,
    },
  };
  Plotly.react("pointCloud", [trace], layout, { responsive: true, displayModeBar: true });
  plotlyReady = true;
}

function renderBevPlotly(pc) {
  const trace = {
    type: "scattergl",
    mode: "markers",
    x: pc.x,
    y: pc.z,
    marker: { size: 3, color: pc.color, opacity: 0.95 },
    hovertemplate: "x=%{x:.3f}<br>z=%{y:.3f}<extra></extra>",
  };
  const layout = {
    margin: { l: 46, r: 12, t: 10, b: 42 },
    paper_bgcolor: "#ffffff",
    plot_bgcolor: "#ffffff",
    xaxis: {
      title: "x",
      zeroline: false,
      showgrid: true,
      scaleanchor: "y",
      scaleratio: 1,
    },
    yaxis: {
      title: "z / depth",
      zeroline: false,
      showgrid: true,
    },
  };
  Plotly.react("pointCloud", [trace], layout, { responsive: true, displayModeBar: true });
  plotlyReady = true;
}

function setPointControlState(view) {
  for (const [key, button] of Object.entries(pointViewButtons)) {
    button.classList.toggle("active", key === view);
  }
}

function renderPointCloud(pc) {
  currentPointCloud = pc;
  setPointControlState(currentPointView);
  pointCount.textContent = `${pc.count} points`;
  if (window.Plotly) {
    if (currentPointView === "bev") renderBevPlotly(pc);
    else renderPointCloudPlotly(pc);
  } else {
    drawPointCloudCanvas(pc);
  }
}

function applyPointView(view) {
  currentPointView = view;
  setPointControlState(view);
  if (!currentPointCloud) return;
  renderPointCloud(currentPointCloud);
}

async function loadSample(index) {
  runStatus.textContent = "Loading";
  const sample = manifest.samples.find(s => String(s.id || s.index) === String(index));
  const meta = await fetch(`${sample.base}/meta.json`).then(r => r.json());
  for (const [key, id] of Object.entries(imageIds)) {
    document.getElementById(id).src = `${sample.base}/${meta.images[key]}`;
  }
  pathLine.textContent = meta.source_npz;
  renderMetrics(meta.metrics);
  const pc = await fetch(`${sample.base}/${meta.point_cloud}`).then(r => r.json());
  renderPointCloud(pc);
  runStatus.textContent = `sample ${index}`;
}

async function init() {
  manifest = await fetch("data/manifest.json").then(r => r.json());
  modelSelect.replaceChildren();
  const models = manifest.models || [{ id: "completionformer", name: "CompletionFormer HW128" }];
  for (const model of models) {
    const opt = document.createElement("option");
    opt.value = model.id;
    opt.textContent = model.name;
    modelSelect.appendChild(opt);
  }
  renderSampleOptions();
  await loadSample(sampleSelect.value);
}

function renderSampleOptions() {
  sampleSelect.replaceChildren();
  const samples = manifest.samples.filter(s => (s.model || "completionformer") === currentModel());
  for (const sample of samples) {
    const opt = document.createElement("option");
    opt.value = sample.id || sample.index;
    opt.textContent = `${sample.title} | abs ${sample.metrics.abs_mean.toFixed(4)}`;
    sampleSelect.appendChild(opt);
  }
  runtimeInfo.textContent = `${samples.length} saved board outputs, ${manifest.point_cloud_sampling}`;
}

loadBtn.addEventListener("click", () => loadSample(sampleSelect.value));
modelSelect.addEventListener("change", async () => {
  renderSampleOptions();
  await loadSample(sampleSelect.value);
});
sampleSelect.addEventListener("change", () => loadSample(sampleSelect.value));
rgbViewBtn.addEventListener("click", () => applyPointView("rgb"));
obliqueViewBtn.addEventListener("click", () => applyPointView("oblique"));
topViewBtn.addEventListener("click", () => applyPointView("top"));
bevViewBtn.addEventListener("click", () => applyPointView("bev"));
window.addEventListener("resize", () => {
  if (currentPointCloud && !(window.Plotly && plotlyReady)) drawPointCloudCanvas(currentPointCloud);
});

init().catch(err => {
  runStatus.textContent = "Load failed";
  pathLine.textContent = err.message;
  console.error(err);
});
