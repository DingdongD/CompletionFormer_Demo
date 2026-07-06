const sampleSelect = document.getElementById("sampleSelect");
const loadBtn = document.getElementById("loadBtn");
const runBtn = document.getElementById("runBtn");
const uploadBtn = document.getElementById("uploadBtn");
const tofBtn = document.getElementById("tofBtn");
const runStatus = document.getElementById("runStatus");
const uploadStatus = document.getElementById("uploadStatus");
const tofStatus = document.getElementById("tofStatus");
const pathLine = document.getElementById("pathLine");
const metricsEl = document.getElementById("metrics");
const pointCount = document.getElementById("pointCount");
const rgbViewBtn = document.getElementById("rgbViewBtn");
const obliqueViewBtn = document.getElementById("obliqueViewBtn");
const topViewBtn = document.getElementById("topViewBtn");
const boardLog = document.getElementById("boardLog");
const logState = document.getElementById("logState");

const imageIds = {
  rgb: "rgbImg",
  sparse_depth: "sparseImg",
  gt: "gtImg",
  ref_pred: "refImg",
  board_pred: "boardImg",
  abs_error: "errImg",
};

function setBusy(text) {
  runStatus.textContent = text;
}

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

function renderSample(payload) {
  for (const [key, id] of Object.entries(imageIds)) {
    document.getElementById(id).src = payload.images[key];
  }
  pathLine.textContent = payload.paths.vis_npz;
  const metricRows = [
    metricRow("abs mean", payload.metrics.abs_mean),
    metricRow("abs p95", payload.metrics.abs_p95),
    metricRow("rmse", payload.metrics.rmse),
    metricRow("board min", payload.metrics.board_min),
    metricRow("board max", payload.metrics.board_max),
  ];
  if (payload.metrics.latency_total_ms !== undefined) {
    metricRows.push(metricRow("latency ms", payload.metrics.latency_total_ms));
    metricRows.push(metricRow("slowest ms", payload.metrics.latency_slowest_ms));
    metricRows.push(metricRow("slowest op", payload.metrics.latency_slowest_op));
  }
  metricsEl.replaceChildren(...metricRows);
  renderPointCloud(payload.point_cloud);
}

const pointViews = {
  rgb: "rgb",
  oblique: "oblique",
  top: "top",
  free: "free",
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

let currentPointView = pointViews.rgb;
let currentPointCloud = null;
let plotlyReady = false;

function applyPointView(view) {
  currentPointView = view;
  if (!currentPointCloud) return;
  if (window.Plotly && plotlyReady && view !== pointViews.free) {
    Plotly.relayout("pointCloud", { "scene.camera": plotlyCameras[view] });
    return;
  }
  drawPointCloudCanvas(currentPointCloud);
}

function parseRgb(cssColor) {
  const nums = cssColor.match(/\d+/g).map(Number);
  return `rgb(${nums[0]},${nums[1]},${nums[2]})`;
}

function projectedPoint(x, y, z, view) {
  if (view === pointViews.rgb || view === pointViews.free) {
    return { u: x, v: y, depth: -z };
  }
  if (view === pointViews.top) {
    return { u: x, v: z, depth: -y };
  }
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
  const radius = currentPointView === pointViews.rgb ? 1.6 : 1.9;
  for (const [, i] of order) {
    const p = projected[i];
    const px = cx + (p.u - midU) * scale;
    const py = cy - (p.v - midV) * scale;
    ctx.fillStyle = parseRgb(pc.color[i]);
    ctx.fillRect(px - radius * 0.5, py - radius * 0.5, radius, radius);
  }

  ctx.strokeStyle = "#d8dde3";
  ctx.lineWidth = 1;
  ctx.strokeRect(0.5, 0.5, width - 1, height - 1);
}

function renderPointCloudPlotly(pc) {
  const trace = {
    type: "scatter3d",
    mode: "markers",
    x: pc.x,
    y: pc.y,
    z: pc.z,
    marker: { size: 2, color: pc.color, opacity: 0.95 },
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

function renderPointCloud(pc) {
  currentPointCloud = pc;
  pointCount.textContent = `${pc.count} points`;
  if (window.Plotly) {
    renderPointCloudPlotly(pc);
  } else {
    drawPointCloudCanvas(pc);
  }
}

async function loadSamples() {
  const res = await fetch("/api/samples");
  const data = await res.json();
  sampleSelect.replaceChildren();
  for (const s of data.samples) {
    const opt = document.createElement("option");
    opt.value = s.index;
    opt.textContent = `sample ${s.index}${s.has_board_output ? " ✓" : ""}`;
    sampleSelect.appendChild(opt);
  }
}

async function loadCurrent() {
  const idx = sampleSelect.value || "0";
  setBusy("Loading");
  const res = await fetch(`/api/sample/${idx}`);
  const data = await res.json();
  if (!res.ok) {
    setBusy("Missing output");
    alert(data.error);
    return;
  }
  renderSample(data);
  setBusy("Loaded");
}

async function runCurrent() {
  const idx = sampleSelect.value || "0";
  setBusy("Running board");
  logState.textContent = `sample ${idx}`;
  boardLog.textContent = "Starting board pipeline...\nThis usually takes tens of seconds. Waiting for run_board_single_sample.sh to finish.";
  runBtn.disabled = true;
  try {
    const res = await fetch(`/api/run/${idx}`, { method: "POST" });
    const data = await res.json();
    if (data.run?.output_tail) {
      boardLog.textContent = data.run.output_tail;
    }
    if (!res.ok) {
      setBusy("Run failed");
      alert(data.error || data.run?.output_tail || "Board run failed");
      return;
    }
    renderSample(data.sample);
    setBusy(`Done ${data.run.elapsed_sec.toFixed(1)}s`);
    await loadSamples();
    sampleSelect.value = idx;
  } finally {
    runBtn.disabled = false;
  }
}

async function stageRgbd() {
  const res = await fetch("/api/upload-rgbd", { method: "POST" });
  const data = await res.json();
  uploadStatus.textContent = data.message;
}

async function checkTof() {
  const res = await fetch("/api/tof/status");
  const data = await res.json();
  tofStatus.textContent = data.message;
}

loadBtn.addEventListener("click", loadCurrent);
runBtn.addEventListener("click", runCurrent);
uploadBtn.addEventListener("click", stageRgbd);
tofBtn.addEventListener("click", checkTof);
rgbViewBtn.addEventListener("click", () => applyPointView(pointViews.rgb));
obliqueViewBtn.addEventListener("click", () => applyPointView(pointViews.oblique));
topViewBtn.addEventListener("click", () => applyPointView(pointViews.top));

loadSamples().then(() => {
  sampleSelect.value = "0";
  loadCurrent();
});
