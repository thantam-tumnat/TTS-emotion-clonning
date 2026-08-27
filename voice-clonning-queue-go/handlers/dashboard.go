package handlers

import (
	"github.com/gofiber/fiber/v2"

	"voice-cloning-queue/queue"
)

// DashboardHandler renders the HTML Web Dashboard for the Go Queue.
type DashboardHandler struct {
	q            *queue.PriorityQueue
	pythonGPUURL string
}

// NewDashboardHandler creates a new DashboardHandler.
func NewDashboardHandler(q *queue.PriorityQueue, pythonGPUURL string) *DashboardHandler {
	return &DashboardHandler{
		q:            q,
		pythonGPUURL: pythonGPUURL,
	}
}

// Index serves the real-time Queue Web Dashboard.
func (d *DashboardHandler) Index(c *fiber.Ctx) error {
	c.Set("Content-Type", "text/html; charset=utf-8")
	return c.SendString(dashboardHTML)
}

const dashboardHTML = `<!doctype html>
<html lang="th">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>SiangTTS Go Central Queue Dashboard</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&family=Noto+Sans+Thai:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg-base: #0b0f19;
      --bg-card: #111827;
      --bg-card-hover: #172236;
      --border-subtle: #1f293d;
      --text-main: #f8fafc;
      --text-muted: #94a3b8;
      --text-dim: #64748b;
      --accent-cyan: #06b6d4;
      --accent-purple: #a855f7;
      --accent-green: #10b981;
      --accent-amber: #f59e0b;
      --accent-red: #ef4444;
      --accent-blue: #3b82f6;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background-color: var(--bg-base);
      color: var(--text-main);
      font-family: 'Plus Jakarta Sans', 'Noto Sans Thai', system-ui, sans-serif;
      padding: 28px;
      min-height: 100vh;
      padding-bottom: 90px; /* space for sticky player */
    }
    .header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 24px;
      padding-bottom: 20px;
      border-bottom: 1px solid var(--border-subtle);
    }
    .title-group h1 {
      font-size: 20px;
      font-weight: 700;
      letter-spacing: -0.02em;
      display: flex;
      align-items: center;
      gap: 10px;
    }
    .badge-go {
      background: linear-gradient(135deg, #00ADD8, #007d9c);
      color: #fff;
      font-size: 11px;
      font-weight: 700;
      padding: 2px 8px;
      border-radius: 6px;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }
    .sub {
      color: var(--text-dim);
      font-size: 13px;
      margin-top: 4px;
    }
    .target-gpu {
      background: var(--bg-card);
      border: 1px solid var(--border-subtle);
      padding: 8px 14px;
      border-radius: 8px;
      font-size: 12px;
      color: var(--text-muted);
      display: flex;
      align-items: center;
      gap: 8px;
      font-family: 'JetBrains Mono', monospace;
    }
    .dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; }
    .dot-green { background: var(--accent-green); box-shadow: 0 0 8px var(--accent-green); }
    .dot-red { background: var(--accent-red); box-shadow: 0 0 8px var(--accent-red); }

    /* Cards Grid */
    .cards-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
      gap: 14px;
      margin-bottom: 24px;
    }
    .card {
      background: var(--bg-card);
      border: 1px solid var(--border-subtle);
      border-radius: 12px;
      padding: 16px;
      display: flex;
      flex-direction: column;
      gap: 4px;
      transition: transform 0.15s ease, border-color 0.15s ease;
    }
    .card:hover {
      transform: translateY(-2px);
      border-color: #334155;
    }
    .card-title {
      font-size: 12px;
      font-weight: 600;
      color: var(--text-muted);
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }
    .card-val {
      font-size: 26px;
      font-weight: 700;
      color: var(--text-main);
      font-family: 'JetBrains Mono', monospace;
    }
    .card-sub {
      font-size: 11px;
      color: var(--text-dim);
    }

    /* Running Banner */
    .running-banner {
      background: linear-gradient(90deg, rgba(6, 182, 212, 0.12), rgba(168, 85, 247, 0.08));
      border: 1px solid rgba(6, 182, 212, 0.35);
      border-radius: 12px;
      padding: 16px 20px;
      margin-bottom: 24px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      box-shadow: 0 4px 20px rgba(6, 182, 212, 0.1);
    }
    .running-info {
      display: flex;
      align-items: center;
      gap: 14px;
    }
    .pulse-ring {
      position: relative;
      width: 12px;
      height: 12px;
      background: var(--accent-cyan);
      border-radius: 50%;
      box-shadow: 0 0 10px var(--accent-cyan);
      animation: pulse 1.8s infinite;
      flex-shrink: 0;
    }
    @keyframes pulse {
      0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(6, 182, 212, 0.7); }
      70% { transform: scale(1); box-shadow: 0 0 0 10px rgba(6, 182, 212, 0); }
      100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(6, 182, 212, 0); }
    }
    .running-actions {
      display: flex;
      align-items: center;
      gap: 12px;
    }

    /* Table */
    .table-container {
      background: var(--bg-card);
      border: 1px solid var(--border-subtle);
      border-radius: 12px;
      overflow: hidden;
    }
    .table-header {
      padding: 16px 20px;
      border-bottom: 1px solid var(--border-subtle);
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    .table-title {
      font-size: 14px;
      font-weight: 600;
      color: var(--text-main);
    }
    table {
      width: 100%;
      border-collapse: collapse;
      text-align: left;
    }
    th, td {
      padding: 12px 16px;
      border-bottom: 1px solid var(--border-subtle);
      font-size: 13px;
      vertical-align: middle;
    }
    th {
      background: #0d1320;
      color: var(--text-muted);
      font-weight: 600;
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }
    tr:hover { background: var(--bg-card-hover); }
    .mono { font-family: 'JetBrains Mono', monospace; font-size: 12px; }

    /* Badges */
    .pill {
      display: inline-block;
      padding: 3px 9px;
      border-radius: 999px;
      font-size: 11px;
      font-weight: 600;
      font-family: 'JetBrains Mono', monospace;
    }
    .pill-queued { background: rgba(245, 158, 11, 0.15); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.3); }
    .pill-running { background: rgba(6, 182, 212, 0.15); color: #38bdf8; border: 1px solid rgba(6, 182, 212, 0.3); }
    .pill-completed { background: rgba(16, 185, 129, 0.15); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.3); }
    .pill-failed { background: rgba(239, 68, 68, 0.15); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.3); }
    .pill-cancelled { background: rgba(100, 116, 139, 0.2); color: #94a3b8; border: 1px solid rgba(100, 116, 139, 0.3); }

    .lane-tag {
      font-size: 11px;
      font-weight: 600;
      padding: 2px 6px;
      border-radius: 4px;
    }
    .lane-interactive { background: rgba(168, 85, 247, 0.15); color: #c084fc; }
    .lane-batch { background: rgba(148, 163, 184, 0.1); color: #94a3b8; }

    /* Action Buttons */
    .btn-group {
      display: flex;
      align-items: center;
      gap: 6px;
    }
    .btn {
      padding: 4px 10px;
      border-radius: 6px;
      font-size: 11px;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.15s ease;
      font-family: 'Plus Jakarta Sans', system-ui, sans-serif;
      display: inline-flex;
      align-items: center;
      gap: 4px;
      text-decoration: none;
    }
    .btn-play {
      background: rgba(16, 185, 129, 0.15);
      color: #34d399;
      border: 1px solid rgba(16, 185, 129, 0.35);
    }
    .btn-play:hover {
      background: rgba(16, 185, 129, 0.3);
      border-color: #10b981;
      color: #fff;
    }
    .btn-play.playing {
      background: #10b981;
      color: #0b0f19;
      box-shadow: 0 0 10px rgba(16, 185, 129, 0.5);
    }
    .btn-cancel {
      background: rgba(239, 68, 68, 0.15);
      color: #f87171;
      border: 1px solid rgba(239, 68, 68, 0.35);
    }
    .btn-cancel:hover {
      background: rgba(239, 68, 68, 0.35);
      border-color: #ef4444;
      color: #ffffff;
    }
    .btn-detail {
      background: rgba(59, 130, 246, 0.15);
      color: #60a5fa;
      border: 1px solid rgba(59, 130, 246, 0.35);
    }
    .btn-detail:hover {
      background: rgba(59, 130, 246, 0.3);
      border-color: #3b82f6;
      color: #fff;
    }

    /* Modal / Drawer */
    .modal-backdrop {
      position: fixed;
      inset: 0;
      background: rgba(0, 0, 0, 0.75);
      backdrop-filter: blur(4px);
      display: none;
      justify-content: center;
      align-items: center;
      z-index: 999;
      padding: 20px;
    }
    .modal-backdrop.open {
      display: flex;
    }
    .modal-box {
      background: var(--bg-card);
      border: 1px solid var(--border-subtle);
      border-radius: 14px;
      width: 100%;
      max-width: 680px;
      max-height: 88vh;
      overflow-y: auto;
      box-shadow: 0 20px 40px rgba(0, 0, 0, 0.5);
      display: flex;
      flex-direction: column;
    }
    .modal-header {
      padding: 18px 22px;
      border-bottom: 1px solid var(--border-subtle);
      display: flex;
      justify-content: space-between;
      align-items: center;
      position: sticky;
      top: 0;
      background: var(--bg-card);
      z-index: 10;
    }
    .modal-title {
      font-size: 16px;
      font-weight: 700;
      color: var(--text-main);
      display: flex;
      align-items: center;
      gap: 10px;
    }
    .modal-close {
      background: none;
      border: none;
      color: var(--text-dim);
      font-size: 20px;
      cursor: pointer;
      padding: 4px 8px;
      border-radius: 6px;
    }
    .modal-close:hover { color: #fff; background: rgba(255,255,255,0.08); }
    .modal-body {
      padding: 22px;
      display: flex;
      flex-direction: column;
      gap: 18px;
    }
    .meta-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
      gap: 10px;
    }
    .meta-item {
      background: #0d1320;
      border: 1px solid var(--border-subtle);
      border-radius: 8px;
      padding: 10px 12px;
    }
    .meta-label {
      font-size: 11px;
      color: var(--text-dim);
      text-transform: uppercase;
      font-weight: 600;
      margin-bottom: 3px;
    }
    .meta-val {
      font-size: 13px;
      color: var(--text-main);
      font-weight: 600;
    }
    .section-label {
      font-size: 12px;
      font-weight: 700;
      color: var(--text-muted);
      text-transform: uppercase;
      letter-spacing: 0.05em;
      margin-bottom: 8px;
      display: flex;
      align-items: center;
      gap: 6px;
    }
    .chunk-box {
      background: #0d1320;
      border: 1px solid var(--border-subtle);
      border-radius: 8px;
      padding: 10px 14px;
      margin-bottom: 8px;
      font-size: 13px;
      line-height: 1.5;
      color: #e2e8f0;
    }
    .chunk-idx {
      font-family: 'JetBrains Mono', monospace;
      font-size: 11px;
      color: var(--accent-cyan);
      font-weight: 700;
      margin-right: 6px;
    }
    .error-box {
      background: rgba(239, 68, 68, 0.1);
      border: 1px solid rgba(239, 68, 68, 0.35);
      border-radius: 8px;
      padding: 12px 16px;
      color: #fca5a5;
      font-size: 13px;
      font-family: 'JetBrains Mono', monospace;
    }

    /* Sticky Audio Player Bar */
    .audio-bar {
      position: fixed;
      bottom: 0;
      left: 0;
      right: 0;
      background: rgba(17, 24, 39, 0.95);
      backdrop-filter: blur(10px);
      border-top: 1px solid rgba(6, 182, 212, 0.3);
      padding: 12px 28px;
      display: none;
      align-items: center;
      justify-content: space-between;
      gap: 20px;
      z-index: 900;
      box-shadow: 0 -4px 20px rgba(0, 0, 0, 0.4);
    }
    .audio-bar.active {
      display: flex;
    }
    .audio-track-info {
      display: flex;
      align-items: center;
      gap: 12px;
      min-width: 200px;
    }
    .audio-player-ctrl {
      flex: 1;
      max-width: 600px;
      display: flex;
      align-items: center;
      gap: 10px;
    }
    audio {
      width: 100%;
      height: 36px;
      outline: none;
    }
    .btn-close-audio {
      background: none;
      border: none;
      color: var(--text-dim);
      font-size: 18px;
      cursor: pointer;
      padding: 4px 8px;
    }
    .btn-close-audio:hover { color: #fff; }
  </style>
</head>
<body>

  <div class="header">
    <div class="title-group">
      <h1>🚀 SiangTTS Central Queue Gateway <span class="badge-go">Go Fiber</span></h1>
      <div class="sub">High-Performance Non-blocking Traffic Controller & Priority Dispatcher (:8020)</div>
    </div>
    <div class="target-gpu" id="gpu-status">
      <span class="dot dot-green" id="gpu-dot"></span>
      <span id="gpu-label">Python GPU: Checking...</span>
    </div>
  </div>

  <div class="cards-grid">
    <div class="card">
      <div class="card-title">⚡ Interactive (Studio)</div>
      <div class="card-val" id="val-wait-interactive" style="color: var(--accent-purple);">0</div>
      <div class="card-sub">Priority Lane Queue</div>
    </div>
    <div class="card">
      <div class="card-title">📦 Batch (Webhook)</div>
      <div class="card-val" id="val-wait-batch" style="color: var(--accent-cyan);">0</div>
      <div class="card-sub">Background Queue</div>
    </div>
    <div class="card">
      <div class="card-title">⚙️ Running</div>
      <div class="card-val" id="val-running" style="color: var(--accent-amber);">0</div>
      <div class="card-sub">Active on GPU :8021</div>
    </div>
    <div class="card">
      <div class="card-title">✅ Completed</div>
      <div class="card-val" id="val-completed" style="color: var(--accent-green);">0</div>
      <div class="card-sub">Successful Jobs</div>
    </div>
    <div class="card">
      <div class="card-title">❌ Failed</div>
      <div class="card-val" id="val-failed" style="color: var(--accent-red);">0</div>
      <div class="card-sub">Errors or Cancelled</div>
    </div>
  </div>

  <!-- Running Banner -->
  <div class="running-banner" id="running-container" style="display: none;">
    <div class="running-info">
      <div class="pulse-ring"></div>
      <div>
        <div style="font-weight: 700; font-size: 14px;" id="running-job-id">job_...</div>
        <div style="font-size: 12px; color: var(--text-muted);" id="running-details">Processing chunks on PyTorch CUDA...</div>
      </div>
    </div>
    <div class="running-actions">
      <div class="mono" style="color: var(--accent-cyan); font-weight: 700; font-size: 14px;" id="running-time">0.0s</div>
      <button class="btn btn-detail" onclick="openRunningDetail()">ℹ Details</button>
      <button class="btn btn-cancel" id="btn-cancel-running" onclick="cancelRunningJob()">✕ Cancel Job</button>
    </div>
  </div>

  <!-- Table -->
  <div class="table-container">
    <div class="table-header">
      <div class="table-title">Recent Jobs History</div>
      <div style="font-size: 12px; color: var(--text-dim);" id="last-sync">Syncing...</div>
    </div>
    <table>
      <thead>
        <tr>
          <th>Status</th>
          <th>Job ID</th>
          <th>Lane</th>
          <th>Client</th>
          <th>Chunks</th>
          <th>Waited</th>
          <th>Ran Time</th>
          <th>Created</th>
          <th>Action</th>
        </tr>
      </thead>
      <tbody id="jobs-tbody">
        <tr>
          <td colspan="9" style="text-align: center; color: var(--text-dim); padding: 24px;">No jobs in queue history yet.</td>
        </tr>
      </tbody>
    </table>
  </div>

  <!-- Details Modal -->
  <div class="modal-backdrop" id="job-modal" onclick="if(event.target===this) closeDetailModal()">
    <div class="modal-box">
      <div class="modal-header">
        <div class="modal-title">
          <span>📋 Job Details</span>
          <span class="pill" id="modal-status-badge">queued</span>
        </div>
        <button class="modal-close" onclick="closeDetailModal()">✕</button>
      </div>
      <div class="modal-body" id="modal-body-content">
        <!-- Injected via JavaScript -->
      </div>
    </div>
  </div>

  <!-- Sticky Bottom Audio Player -->
  <div class="audio-bar" id="sticky-audio-bar">
    <div class="audio-track-info">
      <span style="font-size: 18px;">🔊</span>
      <div>
        <div style="font-weight: 700; font-size: 13px;" id="audio-bar-job-id">job_...</div>
        <div style="font-size: 11px; color: var(--text-dim);" id="audio-bar-client">Audio Playback</div>
      </div>
    </div>
    <div class="audio-player-ctrl">
      <audio id="global-audio-element" controls preload="auto"></audio>
    </div>
    <button class="btn-close-audio" onclick="hideAudioBar()" title="Close Player">✕</button>
  </div>

  <script>
    var currentJobsData = [];
    var currentRunningJob = null;
    var activeAudioJobId = null;

    async function updateDashboard() {
      try {
        const res = await fetch('/v2/jobs');
        if (!res.ok) return;
        const data = await res.json();

        currentJobsData = data.jobs || [];
        currentRunningJob = data.running || null;

        // Update counts
        const counts = data.counts || {};
        const waiting = data.waiting || {};
        document.getElementById('val-wait-interactive').innerText = waiting.interactive || 0;
        document.getElementById('val-wait-batch').innerText = waiting.batch || 0;
        document.getElementById('val-running').innerText = data.running ? 1 : 0;
        document.getElementById('val-completed').innerText = counts.completed || 0;
        document.getElementById('val-failed').innerText = (counts.failed || 0) + (counts.cancelled || 0);

        // Update Running banner
        const runBanner = document.getElementById('running-container');
        if (data.running) {
          runBanner.style.display = 'flex';
          document.getElementById('running-job-id').innerText = data.running.job_id;
          const chunksTxt = data.running.chunks && data.running.chunks[0] ? data.running.chunks[0].substring(0, 60) + '...' : '';
          document.getElementById('running-details').innerText = 'Lane: ' + data.running.lane + ' | ' + chunksTxt;
          document.getElementById('running-time').innerText = (data.running.ran_s ? data.running.ran_s.toFixed(1) : '0.0') + 's';
        } else {
          runBanner.style.display = 'none';
        }

        // Update Table
        var tbody = document.getElementById('jobs-tbody');
        if (!currentJobsData || currentJobsData.length === 0) {
          tbody.innerHTML = '<tr><td colspan="9" style="text-align:center; color: var(--text-dim); padding: 24px;">No jobs in queue history yet.</td></tr>';
        } else {
          var rowsHtml = '';
          for (var i = 0; i < currentJobsData.length; i++) {
            var j = currentJobsData[i];
            var statusClass = 'pill-' + j.status;
            var laneClass = 'lane-' + j.lane;
            var waited = (j.waited_s !== undefined && j.waited_s !== null) ? j.waited_s.toFixed(2) + 's' : '-';
            var ran = (j.ran_s !== undefined && j.ran_s !== null) ? j.ran_s.toFixed(2) + 's' : '-';
            var created = new Date(j.created * 1000).toLocaleTimeString();
            var totalChunks = j.total_chunks || (j.chunks ? j.chunks.length : 1);
            var clientName = j.client || '-';

            // Build Actions
            var actionBtns = '';
            
            // 1. Play Audio (for completed jobs)
            if (j.status === 'completed') {
              var isPlaying = (activeAudioJobId === j.job_id);
              var playClass = isPlaying ? 'btn btn-play playing' : 'btn btn-play';
              var playText = isPlaying ? '⏸ Pause' : '▶ Play';
              actionBtns += '<button class="' + playClass + '" id="play-btn-' + j.job_id + '" onclick="togglePlayAudio(\'' + j.job_id + '\')">' + playText + '</button>';
            }

            // 2. Cancel Job (for any active queue: queued or running)
            if (j.status === 'queued' || j.status === 'running') {
              actionBtns += '<button class="btn btn-cancel" onclick="cancelJob(\'' + j.job_id + '\')">✕ Cancel</button>';
            }

            // 3. View Details
            actionBtns += '<button class="btn btn-detail" onclick="openDetailModal(\'' + j.job_id + '\')">ℹ Details</button>';

            rowsHtml += '<tr>' +
              '<td><span class="pill ' + statusClass + '">' + j.status + '</span></td>' +
              '<td class="mono"><strong>' + j.job_id + '</strong></td>' +
              '<td><span class="lane-tag ' + laneClass + '">' + j.lane + '</span></td>' +
              '<td>' + clientName + '</td>' +
              '<td>' + totalChunks + '</td>' +
              '<td class="mono">' + waited + '</td>' +
              '<td class="mono">' + ran + '</td>' +
              '<td style="color: var(--text-dim);">' + created + '</td>' +
              '<td><div class="btn-group">' + actionBtns + '</div></td>' +
              '</tr>';
          }
          tbody.innerHTML = rowsHtml;
        }

        document.getElementById('last-sync').innerText = 'Live Auto-Sync: ' + new Date().toLocaleTimeString();
      } catch (err) {
        console.error('Failed to sync queue dashboard:', err);
      }
    }

    // Audio Playback
    function togglePlayAudio(jobId) {
      var audioEl = document.getElementById('global-audio-element');
      var bar = document.getElementById('sticky-audio-bar');

      if (activeAudioJobId === jobId && !audioEl.paused) {
        audioEl.pause();
        activeAudioJobId = null;
        updatePlayButtons();
        return;
      }

      var audioUrl = '/v2/jobs/' + encodeURIComponent(jobId) + '/audio';
      audioEl.src = audioUrl;
      document.getElementById('audio-bar-job-id').innerText = jobId;
      
      var job = currentJobsData.find(function(item) { return item.job_id === jobId; });
      if (job) {
        document.getElementById('audio-bar-client').innerText = 'Client: ' + (job.client || 'API') + ' | ' + (job.total_chunks || 1) + ' chunk(s)';
      }

      bar.classList.add('active');
      audioEl.play().catch(function(e) {
        console.warn('Auto-play blocked or error:', e);
      });

      activeAudioJobId = jobId;
      updatePlayButtons();

      audioEl.onended = function() {
        activeAudioJobId = null;
        updatePlayButtons();
      };
      audioEl.onpause = function() {
        activeAudioJobId = null;
        updatePlayButtons();
      };
      audioEl.onplay = function() {
        activeAudioJobId = jobId;
        updatePlayButtons();
      };
    }

    function updatePlayButtons() {
      var playBtns = document.querySelectorAll('.btn-play');
      playBtns.forEach(function(btn) {
        var id = btn.id.replace('play-btn-', '');
        if (id === activeAudioJobId) {
          btn.classList.add('playing');
          btn.innerHTML = '⏸ Pause';
        } else {
          btn.classList.remove('playing');
          btn.innerHTML = '▶ Play';
        }
      });
    }

    function hideAudioBar() {
      var audioEl = document.getElementById('global-audio-element');
      audioEl.pause();
      document.getElementById('sticky-audio-bar').classList.remove('active');
      activeAudioJobId = null;
      updatePlayButtons();
    }

    // Cancel Job
    async function cancelJob(jobId) {
      if (!confirm('ยืนยันยกเลิก Task: ' + jobId + ' ?')) return;
      try {
        const res = await fetch('/v2/jobs/' + encodeURIComponent(jobId), {
          method: 'DELETE'
        });
        const data = await res.json();
        if (res.ok && data.cancelled) {
          updateDashboard();
        } else {
          alert(data.error || 'ไม่สามารถยกเลิก Task ได้');
        }
      } catch (err) {
        console.error('Failed to cancel job:', err);
        alert('เกิดข้อผิดพลาดในการเชื่อมต่อ');
      }
    }

    function cancelRunningJob() {
      if (currentRunningJob && currentRunningJob.job_id) {
        cancelJob(currentRunningJob.job_id);
      }
    }

    function openRunningDetail() {
      if (currentRunningJob && currentRunningJob.job_id) {
        openDetailModal(currentRunningJob.job_id);
      }
    }

    // Modal Details
    function openDetailModal(jobId) {
      var job = currentJobsData.find(function(item) { return item.job_id === jobId; });
      if (!job && currentRunningJob && currentRunningJob.job_id === jobId) {
        job = currentRunningJob;
      }
      if (!job) return;

      var modal = document.getElementById('job-modal');
      var statusBadge = document.getElementById('modal-status-badge');
      statusBadge.className = 'pill pill-' + job.status;
      statusBadge.innerText = job.status;

      var body = document.getElementById('modal-body-content');

      var waitedStr = (job.waited_s !== undefined && job.waited_s !== null) ? job.waited_s.toFixed(2) + 's' : '-';
      var ranStr = (job.ran_s !== undefined && job.ran_s !== null) ? job.ran_s.toFixed(2) + 's' : '-';
      var createdStr = new Date(job.created * 1000).toLocaleString();
      var startedStr = job.started ? new Date(job.started * 1000).toLocaleString() : '-';
      var finishedStr = job.finished ? new Date(job.finished * 1000).toLocaleString() : '-';

      var voiceInfo = '-';
      if (job.voice) {
        if (job.voice.speaker_id) voiceInfo = 'Speaker: ' + job.voice.speaker_id;
        else if (job.voice.handle) voiceInfo = 'Handle: ' + job.voice.handle;
        else if (job.voice.seed) voiceInfo = 'Seed Voice (Auto)';
      }

      var html = '<div class="meta-grid">' +
        '<div class="meta-item"><div class="meta-label">Job ID</div><div class="meta-val mono" style="font-size:11px;">' + job.job_id + '</div></div>' +
        '<div class="meta-item"><div class="meta-label">Client</div><div class="meta-val">' + (job.client || 'ToneStudio') + '</div></div>' +
        '<div class="meta-item"><div class="meta-label">Lane</div><div class="meta-val">' + job.lane + '</div></div>' +
        '<div class="meta-item"><div class="meta-label">CFG Scale</div><div class="meta-val">' + (job.cfg_value || 2.0) + '</div></div>' +
        '<div class="meta-item"><div class="meta-label">Timesteps</div><div class="meta-val">' + (job.timesteps || 10) + '</div></div>' +
        '<div class="meta-item"><div class="meta-label">Voice</div><div class="meta-val" style="font-size:11px;">' + voiceInfo + '</div></div>' +
        '<div class="meta-item"><div class="meta-label">Waited</div><div class="meta-val mono">' + waitedStr + '</div></div>' +
        '<div class="meta-item"><div class="meta-label">GPU Render Time</div><div class="meta-val mono" style="color:var(--accent-cyan);">' + ranStr + '</div></div>' +
        '</div>';

      // Timestamps
      html += '<div style="font-size:11px; color:var(--text-dim); display:flex; gap:16px; flex-wrap:wrap;">' +
        '<span>Created: ' + createdStr + '</span>' +
        '<span>Started: ' + startedStr + '</span>' +
        '<span>Finished: ' + finishedStr + '</span>' +
        '</div>';

      // Audio Player Section (if completed)
      if (job.status === 'completed') {
        html += '<div>' +
          '<div class="section-label">🔊 Audio Result</div>' +
          '<div style="background:#0d1320; border:1px solid var(--border-subtle); border-radius:8px; padding:12px 16px; display:flex; align-items:center; justify-content:space-between; gap:14px;">' +
          '<audio controls style="flex:1;" src="/v2/jobs/' + encodeURIComponent(job.job_id) + '/audio"></audio>' +
          '<a class="btn btn-play" href="/v2/jobs/' + encodeURIComponent(job.job_id) + '/audio" download="' + job.job_id + '.wav">⬇ Download WAV</a>' +
          '</div>' +
          '</div>';
      }

      // Error Box (if failed)
      if (job.error) {
        html += '<div>' +
          '<div class="section-label" style="color:var(--accent-red);">❌ Error Details</div>' +
          '<div class="error-box">' + job.error + '</div>' +
          '</div>';
      }

      // Text Chunks
      if (job.chunks && job.chunks.length > 0) {
        html += '<div>' +
          '<div class="section-label">💬 Text Chunks (' + job.chunks.length + ')</div>';
        for (var c = 0; c < job.chunks.length; c++) {
          html += '<div class="chunk-box"><span class="chunk-idx">[' + (c+1) + '/' + job.chunks.length + ']</span> ' + job.chunks[c] + '</div>';
        }
        html += '</div>';
      }

      // Cancel button in modal (if active)
      if (job.status === 'queued' || job.status === 'running') {
        html += '<div style="margin-top:10px; display:flex; justify-content:flex-end;">' +
          '<button class="btn btn-cancel" style="padding:8px 16px; font-size:12px;" onclick="cancelJob(\'' + job.job_id + '\'); closeDetailModal();">✕ Cancel This Task</button>' +
          '</div>';
      }

      body.innerHTML = html;
      modal.classList.add('open');
    }

    function closeDetailModal() {
      document.getElementById('job-modal').classList.remove('open');
    }

    // GPU Health
    async function checkGPUHealth() {
      try {
        const res = await fetch('/health');
        const gpuDot = document.getElementById('gpu-dot');
        const gpuLabel = document.getElementById('gpu-label');
        if (res.ok) {
          const data = await res.json();
          gpuDot.className = 'dot dot-green';
          gpuLabel.innerText = 'Python GPU (:8021): ' + (data.device || 'Online') + ' | ' + (data.status || 'OK');
        } else {
          gpuDot.className = 'dot dot-red';
          gpuLabel.innerText = 'Python GPU (:8021): Offline';
        }
      } catch (e) {
        document.getElementById('gpu-dot').className = 'dot dot-red';
        document.getElementById('gpu-label').innerText = 'Python GPU (:8021): Unreachable';
      }
    }

    setInterval(updateDashboard, 1000);
    setInterval(checkGPUHealth, 3000);
    updateDashboard();
    checkGPUHealth();
  </script>
</body>
</html>`
