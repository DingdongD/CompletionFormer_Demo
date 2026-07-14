const modelSelect = document.getElementById("modelSelect");
const modelStatus = document.getElementById("modelStatus");
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
const latencyTotal = document.getElementById("latencyTotal");
const latencyPie = document.getElementById("latencyPie");
const latencyTable = document.getElementById("latencyTable");
const latencyInsight = document.getElementById("latencyInsight");
const pointCount = document.getElementById("pointCount");
const rgbViewBtn = document.getElementById("rgbViewBtn");
const obliqueViewBtn = document.getElementById("obliqueViewBtn");
const topViewBtn = document.getElementById("topViewBtn");
const bevViewBtn = document.getElementById("bevViewBtn");
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

let modelInfo = {};

function currentModel() {
  return modelSelect.value || "completionformer";
}

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

function fmtMs(value) {
  if (typeof value !== "number" || !Number.isFinite(value)) return "n/a";
  if (value >= 1000) return `${(value / 1000).toFixed(2)} s`;
  return `${value.toFixed(1)} ms`;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function shortOpName(name) {
  return name
    .replace("nlspn_test.generated_ckpt.nlspn_ckpt_", "")
    .replace("cspn_test.generated_ckpt.", "")
    .replace("vit_patch_test.", "")
    .replace("completionformer_", "");
}

function isLatencyTotalMarker(name) {
  return name === "wall" || name.endsWith("_total") || name.includes("tracked_total");
}

function latencyCategory(name) {
  if (name.startsWith("load::")) return "Submodel load/switch";
  if (name.startsWith("host::") || name.includes("host") || name.includes("split_sum") || name.includes("resize")) {
    return "Host glue / transfer";
  }
  if (name.startsWith("sat_")) return "Quant/saturation checks";
  if (name.includes("nlspn_test.generated_ckpt")) return "RHB compute / switch-mixed";
  if (name.endsWith("_rhb") || name.includes("scale_compensated") || name.includes("cspn_test.")) return "RHB compute";
  return "Other";
}

function latencyBreakdown(metrics) {
  const latencies = metrics.latencies_ms || {};
  const raw = Object.entries(latencies)
    .filter(([, value]) => typeof value === "number" && Number.isFinite(value));
  const ops = raw.filter(([name]) => !isLatencyTotalMarker(name));
  if (!ops.length) return null;

  const categories = new Map();
  for (const [name, value] of ops) {
    const category = latencyCategory(name);
    categories.set(category, (categories.get(category) || 0) + value);
  }
  const measured = Array.from(categories.values()).reduce((a, b) => a + b, 0);
  const total = typeof metrics.latency_total_ms === "number" ? metrics.latency_total_ms : measured;
  if (total > measured + 1) categories.set("Untracked / scheduler", total - measured);

  return {
    rows: Array.from(categories.entries()).filter(([, value]) => value > 0.01).sort((a, b) => b[1] - a[1]),
    topOps: ops.sort((a, b) => b[1] - a[1]).slice(0, 8),
    total,
  };
}

function operatorPieRows(topOps, total, limit = 10) {
  const selected = topOps.slice(0, limit);
  const used = selected.reduce((acc, [, value]) => acc + value, 0);
  const rows = selected.map(([name, value]) => [shortOpName(name), value]);
  if (total > used + 1) rows.push(["other ops / overhead", total - used]);
  return rows;
}

function piePath(cx, cy, r, startAngle, endAngle) {
  const startX = cx + r * Math.cos(startAngle);
  const startY = cy + r * Math.sin(startAngle);
  const endX = cx + r * Math.cos(endAngle);
  const endY = cy + r * Math.sin(endAngle);
  const largeArc = endAngle - startAngle > Math.PI ? 1 : 0;
  return `M ${cx} ${cy} L ${startX} ${startY} A ${r} ${r} 0 ${largeArc} 1 ${endX} ${endY} Z`;
}

function renderOperatorPie(rows, total) {
  const colors = ["#126e82", "#d6a84f", "#5865a9", "#6f8f72", "#9b5c5c", "#4c6f91", "#a06a2d", "#7e6ca8", "#508d8a", "#8a8f98", "#d8dde3"];
  let angle = -Math.PI / 2;
  const paths = [];
  const legend = [];
  rows.forEach(([name, value], index) => {
    const slice = total > 0 ? (value / total) * Math.PI * 2 : 0;
    const next = angle + slice;
    const pct = total > 0 ? value / total * 100 : 0;
    const color = colors[index % colors.length];
    if (slice > 0.0001) {
      paths.push(`<path d="${piePath(100, 100, 86, angle, next)}" fill="${color}"><title>${escapeHtml(name)} ${fmtMs(value)} ${pct.toFixed(1)}%</title></path>`);
    }
    legend.push(`
      <div class="pieLegendRow">
        <span style="background:${color}"></span>
        <strong title="${escapeHtml(name)}">${escapeHtml(name)}</strong>
        <em>${fmtMs(value)} · ${pct.toFixed(1)}%</em>
      </div>
    `);
    angle = next;
  });
  latencyPie.innerHTML = `
    <div class="pieSvgWrap">
      <svg viewBox="0 0 200 200" role="img" aria-label="Fine-grained operator latency pie">
        ${paths.join("")}
        <circle cx="100" cy="100" r="42" fill="#ffffff"></circle>
        <text x="100" y="96" text-anchor="middle" class="pieCenterMain">${fmtMs(total)}</text>
        <text x="100" y="115" text-anchor="middle" class="pieCenterSub">total</text>
      </svg>
    </div>
    <div class="pieLegend">${legend.join("")}</div>
  `;
}

function bottleneckText(modelKey, breakdown, metrics) {
  if (!breakdown) {
    if (modelKey === "nlspn") {
      return "NLSPN saved outputs do not currently include fine-grained latency traces. Instrument the runner with LATENCY markers to expose the host propagation and split-head costs.";
    }
    return "No fine-grained latency trace was parsed from this board output.";
  }
  const [topName, topMs] = breakdown.rows[0];
  const pct = breakdown.total > 0 ? topMs / breakdown.total * 100 : 0;
  if (modelKey === "cspn" && topName === "Submodel load/switch") {
    return `Main bottleneck: repeated submodel load/switch dominates (${pct.toFixed(1)}%). Reduce launch count, reuse loaded packers, or fuse compiler-aligned blocks.`;
  }
  if (modelKey === "completionformer") {
    const slow = metrics.latency_slowest_op ? ` Slowest op: ${metrics.latency_slowest_op} (${fmtMs(metrics.latency_slowest_ms)}).` : "";
    return `Main bottleneck: full-resolution decoder/head RHB compute dominates (${pct.toFixed(1)}%).${slow}`;
  }
  if (modelKey === "nlspn") {
    const slow = metrics.latency_slowest_op ? ` Slowest op: ${shortOpName(metrics.latency_slowest_op)} (${fmtMs(metrics.latency_slowest_ms)}).` : "";
    return `Main bottleneck: many split decoder/head submodel launches. NLSPN logs mix model switch/load overhead into the first RUN after each switch, so the operator pie identifies the expensive launches rather than pure arithmetic only.${slow}`;
  }
  return `Main bottleneck: ${topName} (${pct.toFixed(1)}% of tracked latency).`;
}

function renderLatencyBreakdown(metrics, modelKey) {
  const breakdown = latencyBreakdown(metrics);
  if (!breakdown) {
    latencyTotal.textContent = "not instrumented";
    latencyTable.innerHTML = '<div class="emptyState">Fine-grained latency trace is not available for this output.</div>';
    latencyInsight.textContent = bottleneckText(modelKey, null, metrics);
    if (window.Plotly) Plotly.purge("latencyPie");
    latencyPie.replaceChildren();
    return;
  }

  latencyTotal.textContent = `total ${fmtMs(breakdown.total)}`;
  renderOperatorPie(operatorPieRows(breakdown.topOps, breakdown.total), breakdown.total);

  const categoryRows = breakdown.rows.map(([name, value]) => {
    const pct = breakdown.total > 0 ? value / breakdown.total * 100 : 0;
    return `<tr><td>${escapeHtml(name)}</td><td>${fmtMs(value)}</td><td>${pct.toFixed(1)}%</td></tr>`;
  }).join("");
  const opRows = breakdown.topOps.map(([name, value]) =>
    `<tr><td title="${escapeHtml(name)}">${escapeHtml(shortOpName(name))}</td><td>${fmtMs(value)}</td><td></td></tr>`
  ).join("");
  latencyTable.innerHTML = `
    <h3>Top Fine-Grained Ops</h3>
    <table><tbody>${opRows}</tbody></table>
    <h3>Categories</h3>
    <table><tbody>${categoryRows}</tbody></table>
  `;
  latencyInsight.textContent = bottleneckText(modelKey, breakdown, metrics);
}

function renderSample(payload) {
  for (const [key, id] of Object.entries(imageIds)) {
    document.getElementById(id).src = payload.images[key];
  }
  pathLine.textContent = `${payload.model.label}: ${payload.paths.vis_npz}`;
  const metricRows = [
    metricRow("model", payload.model.label),
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
  if (payload.metrics.csv_pred_l1 !== undefined) {
    metricRows.push(metricRow("csv pred l1", payload.metrics.csv_pred_l1));
  }
  if (payload.metrics.csv_pred_rmse !== undefined) {
    metricRows.push(metricRow("csv pred rmse", payload.metrics.csv_pred_rmse));
  }
  metricsEl.replaceChildren(...metricRows);
  renderLatencyBreakdown(payload.metrics, payload.model.key || currentModel());
  renderPointCloud(payload.point_cloud);
}

const pointViews = {
  rgb: "rgb",
  oblique: "oblique",
  top: "top",
  bev: "bev",
  free: "free",
};

const pointViewButtons = {
  [pointViews.rgb]: rgbViewBtn,
  [pointViews.oblique]: obliqueViewBtn,
  [pointViews.top]: topViewBtn,
  [pointViews.bev]: bevViewBtn,
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

function setPointControlState(view) {
  for (const [key, button] of Object.entries(pointViewButtons)) {
    button.classList.toggle("active", key === view);
  }
}

function applyPointView(view) {
  currentPointView = view;
  setPointControlState(view);
  if (!currentPointCloud) return;
  renderPointCloud(currentPointCloud);
}

function parseRgb(cssColor) {
  const nums = cssColor.match(/\d+/g).map(Number);
  return `rgb(${nums[0]},${nums[1]},${nums[2]})`;
}

function projectedPoint(x, y, z, view) {
  if (view === pointViews.rgb || view === pointViews.free) {
    return { u: x, v: y, depth: -z };
  }
  if (view === pointViews.top || view === pointViews.bev) {
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

function renderPointCloud(pc) {
  currentPointCloud = pc;
  setPointControlState(currentPointView);
  pointCount.textContent = `${pc.count} points`;
  if (window.Plotly) {
    if (currentPointView === pointViews.bev) {
      renderBevPlotly(pc);
    } else {
      renderPointCloudPlotly(pc);
    }
  } else {
    drawPointCloudCanvas(pc);
  }
}

async function loadModels() {
  const res = await fetch("/api/models");
  const data = await res.json();
  modelSelect.replaceChildren();
  modelInfo = {};
  for (const model of data.models) {
    modelInfo[model.key] = model;
    const opt = document.createElement("option");
    opt.value = model.key;
    opt.textContent = model.label;
    modelSelect.appendChild(opt);
  }
  modelSelect.value = "completionformer";
  updateModelStatus();
}

function updateModelStatus() {
  const info = modelInfo[currentModel()];
  if (!info) {
    modelStatus.textContent = "";
    runBtn.disabled = true;
    return;
  }
  modelStatus.textContent = `${info.description}${info.can_run_board ? "" : " · precomputed board outputs"}`;
  runBtn.disabled = !info.can_run_board;
}

async function loadSamples() {
  updateModelStatus();
  const res = await fetch(`/api/samples?model=${encodeURIComponent(currentModel())}`);
  const data = await res.json();
  if (!res.ok) {
    setBusy("Model error");
    alert(data.error);
    return;
  }
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
  const res = await fetch(`/api/sample/${idx}?model=${encodeURIComponent(currentModel())}`);
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
  const info = modelInfo[currentModel()];
  if (info && !info.can_run_board) {
    setBusy("Precomputed");
    boardLog.textContent = `${info.label} does not package a single-sample live board runner in this app. Loaded outputs are the saved board val32 results.`;
    return;
  }
  setBusy("Running board");
  logState.textContent = `${currentModel()} sample ${idx}`;
  boardLog.textContent = "Starting board pipeline...\nThis usually takes tens of seconds. Waiting for run_board_single_sample.sh to finish.";
  runBtn.disabled = true;
  try {
    const res = await fetch(`/api/run/${idx}?model=${encodeURIComponent(currentModel())}`, { method: "POST" });
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
modelSelect.addEventListener("change", async () => {
  await loadSamples();
  sampleSelect.value = "0";
  await loadCurrent();
});
uploadBtn.addEventListener("click", stageRgbd);
tofBtn.addEventListener("click", checkTof);
rgbViewBtn.addEventListener("click", () => applyPointView(pointViews.rgb));
obliqueViewBtn.addEventListener("click", () => applyPointView(pointViews.oblique));
topViewBtn.addEventListener("click", () => applyPointView(pointViews.top));
bevViewBtn.addEventListener("click", () => applyPointView(pointViews.bev));

loadModels().then(() => loadSamples()).then(() => {
  sampleSelect.value = "0";
  loadCurrent();
});
