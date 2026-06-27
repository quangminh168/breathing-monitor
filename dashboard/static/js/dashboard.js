const POLL_INTERVAL_MS = 5000;
const HISTORY_LIMIT = 100;
// Worker hien gui BPM moi 60s (xem --interval trong breathing_worker.py).
// Nguong nay PHAI lon hon chu ky gui cua worker, neu khong dashboard se
// bao "mat ket noi" mot cach gia tao giua 2 lan gui. Neu ban doi
// --interval sang gia tri khac 60, nho cap nhat lai so nay cho khop
// (vd interval=30 -> dat khoang 45000-50000).
const STALE_AFTER_MS = 90000; // 90s = 60s chu ky + 30s du phong tre mang

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

  bpmValueEl.textContent = Number(record.bpm).toFixed(1);
  lastUpdatedEl.textContent = formatTime(timestamp);
  lastSourceEl.textContent = record.source || "—";

  const isAbnormal = record.bpm < 10 || record.bpm > 30;
  bpmValueEl.classList.toggle("warning", isAbnormal);

  const ageMs = Date.now() - new Date(timestamp).getTime();
  setStatus(ageMs <= STALE_AFTER_MS);
}

// Tách riêng try/catch cho updateStats và renderChart -- để nếu 1 trong 2
// bị lỗi thì cái còn lại vẫn chạy được, không "rớt" theo nhau.

function updateStats(records) {
  try {
    if (!records || !records.length) {
      statAvgEl.textContent = "--";
      statMinEl.textContent = "--";
      statMaxEl.textContent = "--";
      return;
    }
    const values = records.map((r) => Number(r.bpm)).filter((v) => !Number.isNaN(v));
    if (!values.length) {
      statAvgEl.textContent = "--";
      statMinEl.textContent = "--";
      statMaxEl.textContent = "--";
      return;
    }
    const avg = values.reduce((a, b) => a + b, 0) / values.length;
    statAvgEl.textContent = avg.toFixed(1);
    statMinEl.textContent = Math.min(...values).toFixed(1);
    statMaxEl.textContent = Math.max(...values).toFixed(1);
  } catch (err) {
    console.error("[dashboard] Lỗi khi tính avg/min/max:", err);
  }
}

function renderChart(records) {
  try {
    const ordered = [...records].reverse();
    const labels = ordered.map((r) => formatTime(r.recorded_at || r.created_at));
    const values = ordered.map((r) => Number(r.bpm));

    if (chart) {
      chart.data.labels = labels;
      chart.data.datasets[0].data = values;
      chart.update();
      return;
    }

    const canvas = document.getElementById("history-chart");
    if (!canvas) {
      console.error("[dashboard] Không tìm thấy canvas #history-chart trong DOM");
      return;
    }
    if (typeof Chart === "undefined") {
      console.error("[dashboard] Chart.js chưa được load (kiểm tra mạng/CDN)");
      return;
    }

    const ctx = canvas.getContext("2d");
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
            pointHoverRadius: 5,
            pointHitRadius: 12,
            borderWidth: 2,
          },
        ],
      },
      options: {
        responsive: true,
        // "nearest" + intersect:false -> chi can ren chuot GAN diem
        // (khong can trung tuyet doi vao diem an) la tooltip hien ra.
        interaction: {
          mode: "nearest",
          axis: "x",
          intersect: false,
        },
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              title: (items) => `Thời gian: ${items[0]?.label ?? ""}`,
              label: (item) => `BPM: ${Number(item.parsed.y).toFixed(1)} lần/phút`,
            },
          },
          zoom: {
            pan: {
              enabled: true,
              mode: "x",
            },
            zoom: {
              wheel: { enabled: true },
              pinch: { enabled: true },
              mode: "x",
            },
            limits: {
              x: { minRange: 5 }, // khong cho zoom sau hon 5 diem du lieu
            },
          },
        },
        scales: {
          x: { grid: { display: false }, ticks: { maxTicksLimit: 6 } },
          y: { grid: { color: "#E2E8E4" }, suggestedMin: 5, suggestedMax: 30 },
        },
      },
    });

    canvas.addEventListener("dblclick", () => {
      if (chart) chart.resetZoom();
    });
  } catch (err) {
    console.error("[dashboard] Lỗi khi vẽ chart lịch sử:", err);
  }
}

async function fetchLatest() {
  try {
    const res = await fetch("/api/latest");
    if (res.status === 404) {
      updateLatest(null);
      return;
    }
    if (!res.ok) {
      console.error(`[dashboard] /api/latest trả lỗi HTTP ${res.status}`);
      return;
    }
    const data = await res.json();
    updateLatest(data);
  } catch (err) {
    setStatus(false);
    console.error("[dashboard] Lỗi khi lấy /api/latest:", err);
  }
}

async function fetchHistory() {
  try {
    const res = await fetch(`/api/history?limit=${HISTORY_LIMIT}`);
    if (!res.ok) {
      console.error(`[dashboard] /api/history trả lỗi HTTP ${res.status}`);
      return;
    }
    const data = await res.json();
    const records = Array.isArray(data) ? data : [];

    // Log này giúp xác định ngay nguyên nhân: nếu thấy số > 0 ở console
    // nhưng chart/stat vẫn trống -> lỗi nằm ở renderChart/updateStats
    // (đã tách try/catch riêng ở trên). Nếu thấy 0 -> chưa có dữ liệu
    // thật trong DB (kiểm tra lại worker / migration DB).
    console.log(`[dashboard] /api/history trả về ${records.length} bản ghi`);

    renderChart(records);
    updateStats(records);
  } catch (err) {
    console.error("[dashboard] Lỗi khi lấy /api/history:", err);
  }
}

function poll() {
  fetchLatest();
  fetchHistory();
}

const resetZoomBtn = document.getElementById("reset-zoom-btn");
if (resetZoomBtn) {
  resetZoomBtn.addEventListener("click", () => {
    if (chart) chart.resetZoom();
  });
}

poll();
setInterval(poll, POLL_INTERVAL_MS);
