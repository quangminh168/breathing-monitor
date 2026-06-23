const POLL_INTERVAL_MS = 5000;
const HISTORY_LIMIT = 100;
const STALE_AFTER_MS = 30000; // quá 30s không có dữ liệu mới -> coi là mất kết nối

const bpmValueEl = document.getElementById("bpm-value");
const lastUpdatedEl = document.getElementById("last-updated");
const lastSourceEl = document.getElementById("last-source");
const statusDotEl = document.getElementById("status-dot");
const statusTextEl = document.getElementById("status-text");
const statAvgEl = document.getElementById("stat-avg");
const statMinEl = document.getElementById("stat-min");
const statMaxEl = document.getElementById("stat-max");

let chart = null;

function formatTime(isoString) {
  const d = new Date(isoString);
  return d.toLocaleTimeString("vi-VN", { hour12: false });
}

function setStatus(online) {
  statusDotEl.classList.toggle("online", online);
  statusDotEl.classList.toggle("offline", !online);
  statusTextEl.textContent = online ? "Đang giám sát" : "Mất kết nối với cảm biến";
}

function updateLatest(record) {
  if (!record) {
    bpmValueEl.textContent = "--";
    lastUpdatedEl.textContent = "—";
    lastSourceEl.textContent = "—";
    setStatus(false);
    return;
  }

  const timestamp = record.recorded_at || record.created_at;

  bpmValueEl.textContent = record.bpm.toFixed(1);
  lastUpdatedEl.textContent = formatTime(timestamp);
  lastSourceEl.textContent = record.source || "—";

  const isAbnormal = record.bpm < 10 || record.bpm > 30;
  bpmValueEl.classList.toggle("warning", isAbnormal);

  const ageMs = Date.now() - new Date(timestamp).getTime();
  setStatus(ageMs <= STALE_AFTER_MS);
}

function updateStats(records) {
  if (!records.length) {
    statAvgEl.textContent = "--";
    statMinEl.textContent = "--";
    statMaxEl.textContent = "--";
    return;
  }
  const values = records.map((r) => r.bpm);
  const avg = values.reduce((a, b) => a + b, 0) / values.length;
  statAvgEl.textContent = avg.toFixed(1);
  statMinEl.textContent = Math.min(...values).toFixed(1);
  statMaxEl.textContent = Math.max(...values).toFixed(1);
}

function renderChart(records) {
  // API trả về mới nhất trước -> đảo lại để vẽ theo thời gian tăng dần
  const ordered = [...records].reverse();
  const labels = ordered.map((r) => formatTime(r.recorded_at || r.created_at));
  const values = ordered.map((r) => r.bpm);

  if (chart) {
    chart.data.labels = labels;
    chart.data.datasets[0].data = values;
    chart.update();
    return;
  }

  const ctx = document.getElementById("history-chart").getContext("2d");
  chart = new Chart(ctx, {
    type: "line",
    data: {
      labels,
      datasets: [
        {
          label: "BPM",
          data: values,
          borderColor: "#2F6B5E",
          backgroundColor: "rgba(79, 169, 140, 0.12)",
          fill: true,
          tension: 0.35,
          pointRadius: 0,
          borderWidth: 2,
        },
      ],
    },
    options: {
      responsive: true,
      plugins: { legend: { display: false } },
      scales: {
        x: { grid: { display: false }, ticks: { maxTicksLimit: 6 } },
        y: { grid: { color: "#E2E8E4" }, suggestedMin: 5, suggestedMax: 30 },
      },
    },
  });
}

async function fetchLatest() {
  try {
    const res = await fetch("/api/latest");
    if (res.status === 404) {
      updateLatest(null);
      return;
    }
    const data = await res.json();
    updateLatest(data);
  } catch (err) {
    setStatus(false);
    console.error("Lỗi khi lấy /api/latest:", err);
  }
}

async function fetchHistory() {
  try {
    const res = await fetch(`/api/history?limit=${HISTORY_LIMIT}`);
    const data = await res.json();
    renderChart(data);
    updateStats(data);
  } catch (err) {
    console.error("Lỗi khi lấy /api/history:", err);
  }
}

function poll() {
  fetchLatest();
  fetchHistory();
}

poll();
setInterval(poll, POLL_INTERVAL_MS);
