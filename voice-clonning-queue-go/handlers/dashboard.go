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
      padding-bottom: 90px;
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
      align-items: flex-start;
      box-shadow: 0 4px 20px rgba(6, 182, 212, 0.1);
      gap: 16px;
    }
    .running-info {
      display: flex;
      align-items: flex-start;
      gap: 14px;
      flex: 1;
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
      margin-top: 5px;
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
      flex-shrink: 0;
    }

    /* Table */
    .table-container {
      background: var(--bg-card);
      border: 1px solid var(--border-subtle);
      border-radius: 12px;
      overflow-x: auto;
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
      padding: 12px 14px;
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
      white-space: nowrap;
    }
    tr:hover { background: var(--bg-card-hover); }
    .mono { font-family: 'JetBrains Mono', monospace; font-size: 12px; }

    /* Prompt column styling */
    .prompt-cell {
      max-width: 460px;
      min-width: 250px;
      line-height: 1.5;
      font-size: 12.5px;
      color: #f1f5f9;
      word-break: break-word;
    }
    .raw-prompt-container {
      background: rgba(168, 85, 247, 0.09);
      border: 1px solid rgba(168, 85, 247, 0.3);
      border-radius: 7px;
      padding: 7px 10px;
      margin-bottom: 6px;
      font-size: 12.5px;
      line-height: 1.5;
      color: #f3e8ff;
      word-break: break-word;
    }
    .raw-prompt-title {
      display: flex;
      align-items: center;
      gap: 6px;
      margin-bottom: 4px;
    }
    .badge-raw-prompt {
      display: inline-block;
      background: linear-gradient(135deg, rgba(168, 85, 247, 0.35), rgba(139, 92, 246, 0.25));
      color: #e9d5ff;
      border: 1px solid rgba(168, 85, 247, 0.5);
      border-radius: 4px;
      padding: 1px 6px;
      font-size: 10px;
      font-family: 'JetBrains Mono', monospace;
      font-weight: 700;
      letter-spacing: 0.04em;
      text-transform: uppercase;
    }
    .chunks-subgroup {
      border-top: 1px dashed rgba(255, 255, 255, 0.1);
      padding-top: 5px;
      margin-top: 5px;
    }
    .badge-chunk-header {
      font-size: 10px;
      color: var(--text-dim);
      font-weight: 600;
      text-transform: uppercase;
      margin-bottom: 3px;
    }
    .chunk-pill {
      display: inline-block;
      background: rgba(6, 182, 212, 0.14);
      color: var(--accent-cyan);
      border: 1px solid rgba(6, 182, 212, 0.3);
      border-radius: 4px;
      padding: 1px 5px;
      font-size: 10.5px;
      font-family: 'JetBrains Mono', monospace;
      font-weight: 700;
      margin-right: 4px;
      vertical-align: baseline;
      flex-shrink: 0;
    }
    .chunk-line {
      margin-bottom: 5px;
      display: flex;
      align-items: baseline;
      gap: 2px;
    }
    .chunk-line:last-child {
      margin-bottom: 0;
    }
    .chunk-text {
      flex: 1;
    }

    /* Badges */
    .pill {
      display: inline-block;
      padding: 3px 9px;
      border-radius: 999px;
      font-size: 11px;
      font-weight: 600;
      font-family: 'JetBrains Mono', monospace;
      white-space: nowrap;
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
      white-space: nowrap;
    }
    .lane-interactive { background: rgba(168, 85, 247, 0.15); color: #c084fc; }
    .lane-batch { background: rgba(148, 163, 184, 0.1); color: #94a3b8; }
    .voice-tag {
      font-size: 11px;
      font-weight: 600;
      padding: 2px 6px;
      border-radius: 4px;
      white-space: nowrap;
      background: rgba(236, 72, 153, 0.14);
      color: #f472b6;
      border: 1px solid rgba(236, 72, 153, 0.3);
    }

    /* Action Buttons */
    .btn-group {
      display: flex;
      align-items: center;
      gap: 6px;
      white-space: nowrap;
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
      max-width: 720px;
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
      line-height: 1.6;
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

    /* --- Request cards -------------------------------------------------
       One card per upstream request. A take is split into one GPU job per
       emotion, so a flat job table showed three unrelated rows for what the
       user asked once -- and no way to see which chunks belonged together. */
    .req-list { display: flex; flex-direction: column; gap: 10px; }
    .req-card {
      background: var(--bg-card);
      border: 1px solid var(--border-subtle);
      border-radius: 12px;
      overflow: hidden;
    }
    .req-card.oom { border-color: rgba(239, 68, 68, 0.55); }
    .req-head {
      display: flex; align-items: center; gap: 12px; flex-wrap: wrap;
      padding: 11px 15px; cursor: pointer; transition: background .15s ease;
    }
    .req-head:hover { background: var(--bg-card-hover); }
    .req-caret { color: var(--text-dim); font-size: 10px; width: 10px; flex-shrink: 0; }
    .req-id { font-family: 'JetBrains Mono', monospace; font-size: 12.5px; font-weight: 700; }
    .req-prompt-peek {
      color: var(--text-muted); font-size: 12px; max-width: 460px;
      overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
    }
    .req-meta {
      display: flex; gap: 13px; flex-wrap: wrap; align-items: center;
      font-size: 11.5px; color: var(--text-dim); margin-left: auto;
    }
    .req-meta b { color: var(--text-muted); font-weight: 600; }
    .req-count {
      background: rgba(168, 85, 247, 0.14); color: #c084fc;
      border: 1px solid rgba(168, 85, 247, 0.3);
      padding: 2px 9px; border-radius: 99px; font-size: 11px; font-weight: 700;
      white-space: nowrap;
    }
    .req-body { display: none; padding: 0 15px 14px; }
    .req-card.open .req-body { display: block; }
    .req-section {
      font-size: 10.5px; text-transform: uppercase; letter-spacing: .07em;
      color: var(--text-dim); font-weight: 700; margin: 12px 0 7px;
    }
    .chunk-row {
      display: flex; gap: 10px; align-items: flex-start;
      padding: 7px 11px; border-radius: 7px; background: #0d1320;
      border: 1px solid var(--border-subtle); margin-bottom: 5px;
      font-size: 12.5px; line-height: 1.55;
    }
    .chunk-row .n {
      font-family: 'JetBrains Mono', monospace; font-size: 11px;
      color: var(--accent-purple); font-weight: 700; white-space: nowrap; padding-top: 1px;
    }
    .chunk-row .tx { flex: 1; word-break: break-word; }
    .chunk-row .st { margin-left: auto; flex-shrink: 0; }
    .subjob-row {
      display: flex; gap: 10px; align-items: center; flex-wrap: wrap;
      padding: 6px 11px; border-radius: 7px; border: 1px dashed var(--border-subtle);
      margin-bottom: 5px; font-size: 11.5px; color: var(--text-muted);
    }
    .oom-banner {
      display: flex; gap: 11px; align-items: flex-start;
      background: rgba(239, 68, 68, 0.10); border: 1px solid rgba(239, 68, 68, 0.35);
      border-radius: 8px; padding: 11px 14px; margin-top: 12px; font-size: 12.5px;
    }
    .oom-banner .msg { color: #fca5a5; line-height: 1.6; word-break: break-word; }
    .badge-oom {
      background: #ef4444; color: #fff; padding: 3px 8px; border-radius: 5px;
      font-size: 10px; font-weight: 800; letter-spacing: .05em; white-space: nowrap;
    }
    .badge-upstream {
      background: rgba(100, 116, 139, 0.25); color: #cbd5e1; padding: 2px 7px;
      border-radius: 5px; font-size: 10px; font-weight: 700; white-space: nowrap;
    }
    /* A default quietly standing in for something the caller did not send. Amber,
       not red: the request still ran, it just did not run as asked. */
    .badge-fallback {
      background: rgba(245, 158, 11, 0.16); color: #fbbf24;
      border: 1px solid rgba(245, 158, 11, 0.35);
      padding: 2px 7px; border-radius: 5px; font-size: 10px; font-weight: 700;
      white-space: nowrap;
    }
    .resolved-grid {
      display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 9px;
    }
    .resolved-item {
      background: #0d1320; border: 1px solid var(--border-subtle);
      border-radius: 7px; padding: 7px 11px; min-width: 132px;
    }
    .resolved-item.fallback {
      border-color: rgba(245, 158, 11, 0.4); background: rgba(245, 158, 11, 0.07);
    }
    /* A callback URL is an order of magnitude longer than a voice handle. Capped so
       it wraps inside its own tile instead of stretching the row and pushing every
       other resolved field onto a line of its own. */
    .resolved-item.wide { max-width: 340px; }
    .resolved-key {
      font-size: 10px; text-transform: uppercase; letter-spacing: .06em;
      color: var(--text-dim); font-weight: 700;
    }
    .resolved-val {
      font-family: 'JetBrains Mono', monospace; font-size: 12px;
      color: var(--text-muted); word-break: break-all; margin-top: 2px;
    }
    .resolved-src { font-size: 10px; color: var(--text-dim); margin-top: 3px; }
    .btn-json {
      background: rgba(139, 92, 246, 0.14);
      color: #a78bfa;
      border: 1px solid rgba(139, 92, 246, 0.35);
      font-family: 'JetBrains Mono', monospace;
    }
    .btn-json.on { background: rgba(139, 92, 246, 0.3); color: #ddd6fe; }
    .btn-engine {
      background: rgba(6, 182, 212, 0.14);
      color: #22d3ee;
      border: 1px solid rgba(6, 182, 212, 0.35);
    }
    .btn-engine.on { background: rgba(6, 182, 212, 0.3); color: #cffafe; }
    .vc-on { color: #22d3ee; font-size: 11px; }
    .vc-off { color: #fbbf24; font-size: 11px; }
    .json-toggle {
      background: none; border: none; cursor: pointer; padding: 0 0 6px;
      color: var(--text-dim); font-size: 11px; font-weight: 700;
      letter-spacing: .04em; font-family: inherit;
    }
    .json-toggle:hover { color: var(--accent-purple); }
    .req-json {
      background: #0d1320; border: 1px solid var(--border-subtle); border-radius: 8px;
      padding: 10px 13px; margin: 0; max-height: 260px; overflow: auto;
      font-family: 'JetBrains Mono', monospace; font-size: 11.5px; line-height: 1.55;
      color: var(--text-muted); white-space: pre; word-break: normal;
    }
    /* Actions live in the header, not the body: a collapsed card still has to
       offer Play and Cancel, or every playback costs an extra click. */
    .req-actions { display: flex; gap: 6px; flex-shrink: 0; margin-left: 4px; }
    /* Why a request failed, visible without expanding it -- the whole point of
       the status pill is lost if the reason is one click away. */
    .req-error-strip {
      display: flex; gap: 9px; align-items: baseline;
      padding: 9px 15px 11px;
      border-top: 1px dashed rgba(239, 68, 68, 0.25);
      color: #fca5a5; font-size: 11.5px; line-height: 1.5;
    }
    .req-error-strip.muted {
      color: var(--text-muted); border-top-color: rgba(100, 116, 139, 0.25);
    }
    .req-error-strip .lbl {
      font-weight: 800; font-size: 10px; letter-spacing: .05em;
      text-transform: uppercase; flex-shrink: 0;
    }
    .req-error-strip .txt { word-break: break-word; }
    .fail-box {
      background: rgba(239, 68, 68, 0.08); border: 1px solid rgba(239, 68, 68, 0.3);
      border-radius: 8px; padding: 10px 14px; margin-top: 12px;
      color: #fca5a5; font-size: 12px; line-height: 1.6; word-break: break-word;
      font-family: 'JetBrains Mono', monospace;
    }
    .fail-box.muted {
      background: rgba(100, 116, 139, 0.10); border-color: rgba(100, 116, 139, 0.3);
      color: var(--text-muted);
    }
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
    <div class="card">
      <div class="card-title">🧠 VRAM OOM</div>
      <div class="card-val" id="val-oom" style="color: var(--accent-red);">0</div>
      <div class="card-sub">CUDA out of memory</div>
    </div>
  </div>

  <!-- Running Banner -->
  <div class="running-banner" id="running-container" style="display: none;">
    <div class="running-info">
      <div class="pulse-ring"></div>
      <div style="flex:1;">
        <div style="font-weight: 700; font-size: 14px;" id="running-job-id">job_...</div>
        <div style="font-size: 12px; color: var(--text-muted); margin-top: 4px;" id="running-details">Processing chunks on PyTorch CUDA...</div>
      </div>
    </div>
    <div class="running-actions">
      <div class="mono" style="color: var(--accent-cyan); font-weight: 700; font-size: 14px;" id="running-time">0.0s</div>
      <button class="btn btn-detail" onclick="openRunningDetail()">ℹ Details</button>
      <button class="btn btn-cancel" id="btn-cancel-running" onclick="cancelRunningJob()">✕ Cancel Job</button>
    </div>
  </div>

  <!-- Requests -->
  <div class="table-container">
    <div class="table-header">
      <div class="table-title">Requests
        <span style="color: var(--text-dim); font-weight: 500; font-size: 12px;">— one card per upstream request, click to expand its chunks</span>
      </div>
      <div style="font-size: 12px; color: var(--text-dim);" id="last-sync">Syncing...</div>
    </div>
    <div style="padding: 14px 15px;">
      <div class="req-list" id="req-list">
        <div style="text-align: center; color: var(--text-dim); padding: 24px;">No requests in history yet.</div>
      </div>
    </div>
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

    function voiceIdOf(job) {

      var vs = job && job.voice;

      if (!vs) return 'auto';

      return vs.speaker_id || vs.handle || 'auto';

    }


    function escapeHtml(str) {
      if (!str) return '';
      return String(str)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#039;');
    }

    async function updateDashboard() {
      try {
        // No hide_internal: the per-emotion render jobs ARE the request, and the
        // card view groups them by parent_id instead of hiding them.
        const res = await fetch('/v2/jobs?limit=300');
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
          
          var runningRawHtml = '';
          if (data.running.raw_prompt && data.running.raw_prompt.trim().length > 0) {
            runningRawHtml = '<div class="raw-prompt-container" style="margin-bottom:6px;">' +
              '<div class="raw-prompt-title"><span class="badge-raw-prompt">RAW PROMPT</span></div>' +
              '<div style="font-weight:500;">' + escapeHtml(data.running.raw_prompt) + '</div>' +
            '</div>';
          }
          var runningChunksHtml = '';
          if (data.running.chunks && data.running.chunks.length > 0) {
            for (var rc = 0; rc < data.running.chunks.length; rc++) {
              runningChunksHtml += '<div class="chunk-line"><span class="chunk-pill">[' + (rc+1) + '/' + data.running.chunks.length + ']</span> <span class="chunk-text">' + escapeHtml(data.running.chunks[rc]) + '</span></div>';
            }
          }
          document.getElementById('running-details').innerHTML = '<div style="margin-bottom:6px; font-weight:600; color:var(--text-muted);">Lane: ' + data.running.lane + ' | Voice: ' + escapeHtml(voiceIdOf(data.running)) + ' | Client: ' + (data.running.client || 'tone-studio') + ' (' + (data.running.chunks ? data.running.chunks.length : 0) + ' chunks)</div>' + runningRawHtml + runningChunksHtml;
          document.getElementById('running-time').innerText = (data.running.ran_s ? data.running.ran_s.toFixed(1) : '0.0') + 's';
        } else {
          runBanner.style.display = 'none';
        }

        renderRequests(currentJobsData);

        document.getElementById('last-sync').innerText = 'Live Auto-Sync: ' + new Date().toLocaleTimeString();
      } catch (err) {
        console.error('Failed to sync queue dashboard:', err);
      }
    }

    // ------------------------------------------------------------------ //
    // Request cards
    // ------------------------------------------------------------------ //

    // Which cards the operator has expanded. Kept outside the render so a 1s
    // refresh does not slam an open card shut mid-read.
    var openGroups = {};

    function toggleGroup(gid) {
      openGroups[gid] = !openGroups[gid];
      renderRequests(currentJobsData);
    }

    // Raw JSON is off by default and remembered per card: it is the thing you open
    // when a take came out wrong, not something to scroll past on every other card.
    // Opening it opens the card too, or the button would appear to do nothing.
    var openJson = {};

    function toggleJson(gid) {
      openJson[gid] = !openJson[gid];
      if (openJson[gid]) openGroups[gid] = true;
      renderRequests(currentJobsData);
    }

    // Same idea for the receipt: what the pipeline actually fed the model, which
    // only exists once the take has run.
    var openEngine = {};

    function toggleEngine(gid) {
      openEngine[gid] = !openEngine[gid];
      if (openEngine[gid]) openGroups[gid] = true;
      renderRequests(currentJobsData);
    }

    // Worst-first, so a request shows the state that needs attention: something
    // still on the GPU outranks something waiting, and a failure outranks a
    // sibling that happened to finish before the OOM hit.
    var STATUS_RANK = { running: 0, queued: 1, failed: 2, cancelled: 3, completed: 4 };

    function groupStatus(members) {
      var best = null;
      for (var i = 0; i < members.length; i++) {
        var s = members[i].status;
        if (STATUS_RANK[s] === undefined) continue;
        if (best === null || STATUS_RANK[s] < STATUS_RANK[best]) best = s;
      }
      return best || 'queued';
    }

    // A request is its parent job plus every job that named it as parent. A job
    // with no parent is a request of one, so every job lands in exactly one card.
    function buildGroups(jobs) {
      var byGroup = {};
      var order = [];
      for (var i = 0; i < jobs.length; i++) {
        var j = jobs[i];
        var gid = j.parent_id || j.job_id;
        if (!byGroup[gid]) {
          byGroup[gid] = { id: gid, parent: null, children: [], newest: j.created };
          order.push(gid);
        }
        var g = byGroup[gid];
        if (j.job_id === gid) { g.parent = j; } else { g.children.push(j); }
        if (j.created > g.newest) g.newest = j.created;
      }
      order.sort(function (a, b) { return byGroup[b].newest - byGroup[a].newest; });
      return order.map(function (id) { return byGroup[id]; });
    }

    function fmtSecs(v) {
      return (v !== undefined && v !== null) ? v.toFixed(2) + 's' : '-';
    }

    function renderRequests(jobs) {
      var host = document.getElementById('req-list');
      if (!jobs || jobs.length === 0) {
        host.innerHTML = '<div style="text-align:center;color:var(--text-dim);padding:24px;">No requests in history yet.</div>';
        document.getElementById('val-oom').innerText = 0;
        return;
      }

      var groups = buildGroups(jobs);
      var oomCount = 0;
      var html = '';

      for (var gi = 0; gi < groups.length; gi++) {
        var g = groups[gi];
        // Oldest-first so chunk numbering follows generation order.
        g.children.sort(function (a, b) { return a.created - b.created; });
        var members = (g.parent ? [g.parent] : []).concat(g.children);

        var oomJob = null, upstreamCancelled = [];
        for (var m = 0; m < members.length; m++) {
          if (members[m].error_kind === 'oom') oomJob = members[m];
          if (members[m].error_kind === 'upstream_oom') upstreamCancelled.push(members[m]);
        }
        if (oomJob) oomCount++;

        var status = oomJob ? 'failed' : groupStatus(members);
        var head = g.parent || g.children[0] || {};
        var client = head.client || '-';
        var lane = head.lane || 'batch';

        // What the caller actually asked for, when the request payload is attached:
        // the :8013 pipeline generates in a *donor's* voice and only swaps timbre
        // to the target afterwards, so its render children carry donor handles here.
        // Reading those would label the card with the donor, not the voice_id.
        var reqPayload = null, enginePayload = null;
        for (var q = 0; q < members.length; q++) {
          if (!reqPayload && members[q].request) reqPayload = members[q].request;
          if (!enginePayload && members[q].engine) enginePayload = members[q].engine;
        }
        var voiceId = '';
        if (reqPayload && reqPayload.resolved && reqPayload.resolved.voice_id) {
          voiceId = reqPayload.resolved.voice_id;
        }
        // Otherwise the voice_id lands in the render spec as speaker_id (handle for
        // an uploaded clip). Take it from whichever member actually carries a voice
        // -- an external parent row often has none while its render children do.
        for (var v = 0; v < members.length && !voiceId; v++) {
          var vs = members[v].voice;
          if (!vs) continue;
          if (vs.speaker_id) { voiceId = vs.speaker_id; break; }
          if (vs.handle && !voiceId) voiceId = vs.handle;
        }
        if (!voiceId) voiceId = 'auto';

        // Flag the silent substitutions on the collapsed card: a request that named
        // no voice_id or no sex is served by a default, and a take in the wrong
        // voice otherwise looks like a correct one.
        // The badge names the fields, not the reasons -- the reasons are a sentence
        // each and belong in the expanded card. Hovering gives them back.
        var fellBack = [], fellBackWhy = [];
        if (reqPayload && reqPayload.resolved) {
          var rs = reqPayload.resolved;
          if (isSubstituted(rs.voice_id_source)) { fellBack.push('voice_id'); fellBackWhy.push('voice_id: ' + rs.voice_id_source); }
          if (isSubstituted(rs.sex_source)) { fellBack.push('sex'); fellBackWhy.push('sex: ' + rs.sex_source); }
        }

        var rawPrompt = '';
        for (var r = 0; r < members.length; r++) {
          if (members[r].raw_prompt && members[r].raw_prompt.trim().length > 0) {
            rawPrompt = members[r].raw_prompt; break;
          }
        }

        // Chunk rows. Prefer the pieces the GPU jobs actually carried; fall back
        // to the parent's planned chunks when no render job exists yet.
        var chunkRows = [];
        if (g.children.length > 0) {
          for (var ci = 0; ci < g.children.length; ci++) {
            var ch = g.children[ci];
            var cl = ch.chunks || [];
            for (var k = 0; k < cl.length; k++) {
              chunkRows.push({ text: cl[k], status: ch.status, jobId: ch.job_id, kind: ch.error_kind });
            }
          }
        } else if (g.parent && g.parent.chunks) {
          for (var pk = 0; pk < g.parent.chunks.length; pk++) {
            chunkRows.push({ text: g.parent.chunks[pk], status: g.parent.status, jobId: g.parent.job_id, kind: g.parent.error_kind });
          }
        }

        var audioJob = null, activeJob = null, ran = null, waited = null, failedJob = null;
        for (var a = 0; a < members.length; a++) {
          // The parent's audio is the finished take -- converted, assembled, the
          // thing that was actually delivered. A child's is one raw pre-VC chunk,
          // so it is only ever the fallback for a request that has no parent row.
          if (members[a].has_audio && (!audioJob || members[a].job_id === g.id)) {
            audioJob = members[a];
          }
          if (!activeJob && (members[a].status === 'queued' || members[a].status === 'running')) activeJob = members[a];
          // First member that actually says why it stopped. Prefer a real failure
          // over a cancellation, so an OOM is never masked by the sibling it took
          // down with it.
          if (members[a].error && (!failedJob || (failedJob.status !== 'failed' && members[a].status === 'failed'))) {
            failedJob = members[a];
          }
          if (members[a].ran_s !== undefined && members[a].ran_s !== null) ran = (ran || 0) + members[a].ran_s;
          if (waited === null && members[a].waited_s !== undefined && members[a].waited_s !== null) waited = members[a].waited_s;
        }
        if (oomJob) failedJob = oomJob;

        var isOpen = !!openGroups[g.id];
        var created = new Date(g.newest * 1000).toLocaleTimeString();

        var actions = '';
        if (audioJob) {
          var playing = (activeAudioJobId === audioJob.job_id);
          actions += '<button class="btn btn-play' + (playing ? ' playing' : '') + '" id="play-btn-' + audioJob.job_id +
            '" onclick="event.stopPropagation();togglePlayAudio(\'' + audioJob.job_id + '\')">' + (playing ? '⏸ Pause' : '▶ Play') + '</button>';
        }
        if (activeJob) {
          actions += '<button class="btn btn-cancel" onclick="event.stopPropagation();cancelJob(\'' + activeJob.job_id + '\')">✕ Cancel request</button>';
        }
        if (reqPayload) {
          actions += '<button class="btn btn-json' + (openJson[g.id] ? ' on' : '') + '" title="the JSON this request arrived with, and what the pipeline resolved it to"' +
            ' onclick="event.stopPropagation();toggleJson(\'' + g.id + '\')">{ } JSON</button>';
        }
        if (enginePayload) {
          actions += '<button class="btn btn-engine' + (openEngine[g.id] ? ' on' : '') + '" title="what actually went into the model: donor clip per emotion, the target SeedVC converted into, and the knobs"' +
            ' onclick="event.stopPropagation();toggleEngine(\'' + g.id + '\')">&#9881; Engine</button>';
        }
        actions += '<button class="btn btn-detail" onclick="event.stopPropagation();openDetailModal(\'' + (head.job_id || g.id) + '\')">ℹ Details</button>';

        html += '<div class="req-card' + (isOpen ? ' open' : '') + (oomJob ? ' oom' : '') + '">' +
          '<div class="req-head" onclick="toggleGroup(\'' + g.id + '\')">' +
            '<span class="req-caret">' + (isOpen ? '&#9660;' : '&#9654;') + '</span>' +
            '<span class="pill pill-' + status + '">' + status + '</span>' +
            (oomJob ? '<span class="badge-oom">VRAM OOM</span>' : '') +
            '<span class="req-id">' + escapeHtml(g.id) + '</span>' +
            '<span class="req-count">' + chunkRows.length + ' chunk' + (chunkRows.length === 1 ? '' : 's') + '</span>' +
            (g.children.length > 1 ? '<span class="req-count" style="background:rgba(6,182,212,.14);color:#38bdf8;border-color:rgba(6,182,212,.3);">' + g.children.length + ' jobs</span>' : '') +
            (rawPrompt ? '<span class="req-prompt-peek">' + escapeHtml(rawPrompt) + '</span>' : '') +
            '<span class="req-meta">' +
              '<span class="lane-tag lane-' + lane + '">' + lane + '</span>' +
              '<span class="voice-tag" title="voice_id">&#127908; ' + escapeHtml(voiceId) + '</span>' +
              (fellBack.length ? '<span class="badge-fallback" title="' + escapeHtml(fellBackWhy.join(' | ')) + '">&#9888; defaulted: ' + escapeHtml(fellBack.join(', ')) + '</span>' : '') +
              '<span><b>' + escapeHtml(client) + '</b></span>' +
              '<span>wait ' + fmtSecs(waited) + '</span>' +
              '<span>gpu ' + fmtSecs(ran) + '</span>' +
              '<span>' + created + '</span>' +
            '</span>' +
            '<div class="req-actions">' + actions + '</div>' +
          '</div>' +
          renderErrorStrip(failedJob, status) +
          '<div class="req-body">' +
            (rawPrompt ?
              '<div class="req-section">Raw prompt</div>' +
              '<div class="raw-prompt-container" style="padding:10px 13px;border-radius:8px;">' +
                '<div style="white-space:pre-wrap;line-height:1.6;">' + escapeHtml(rawPrompt) + '</div>' +
              '</div>' : '') +
            renderRequestPayload(reqPayload, !!openJson[g.id], g.id) +
            renderEnginePayload(enginePayload, !!openEngine[g.id], g.id) +
            '<div class="req-section">Chunks sent to the GPU (' + chunkRows.length + ')</div>' +
            renderChunkRows(chunkRows) +
            (g.children.length > 1 ? '<div class="req-section">Render jobs in this request (' + g.children.length + ')</div>' + renderSubJobs(g.children) : '') +
            renderFailureBanner(oomJob, failedJob, upstreamCancelled) +
          '</div>' +
        '</div>';
      }

      host.innerHTML = html;
      document.getElementById('val-oom').innerText = oomCount;
      updatePlayButtons();
    }

    // A source string that means the caller did not choose this value. A donor set
    // drawn at random is *not* one of these -- varying the actor is the documented
    // behaviour, and flagging it would drown the fields that really were guessed.
    function isSubstituted(src) {
      if (!src) return false;
      return src.indexOf('not sent') === 0 || src.indexOf('unrecognised') === 0;
    }

    // The upstream request exactly as it arrived, above the pipeline's reading of
    // it. Shown as JSON rather than as fields: the point is to see what the caller
    // really sent -- including keys the studio ignores and values it overrode.
    function renderRequestPayload(req, showJson, gid) {
      if (!req) return '';
      var received = req.received || req;
      var resolved = req.resolved || null;
      var out = '<div class="req-section">Request received from the caller</div>';

      if (resolved) {
        var rows = [
          ['voice_id', resolved.voice_id || '(none)', resolved.voice_id_source],
          ['sex', resolved.sex || '(none)', resolved.sex_source],
          ['donor_set', resolved.donor_set || '(auto)', resolved.donor_set_source]
        ];
        // Where the finished take was announced. Worth its own tile rather than
        // leaving it in the raw JSON: the caller's callback_url is optional, so a
        // request that omits it is delivered to whatever the studio's .env names --
        // and an upload that succeeded while the callback went to a dead endpoint
        // reads as a completed job from every other field on this card.
        // Only rendered when the studio reported it: rows from before it did would
        // otherwise all claim '(none)'.
        if (resolved.callback_url || resolved.callback_url_source) {
          rows.push(['callback_url', resolved.callback_url || '(none)', resolved.callback_url_source]);
        }
        out += '<div class="resolved-grid">';
        for (var i = 0; i < rows.length; i++) {
          var src = rows[i][2] || '';
          var isFallback = isSubstituted(src);
          var wide = rows[i][0] === 'callback_url' ? ' wide' : '';
          out += '<div class="resolved-item' + (isFallback ? ' fallback' : '') + wide + '">' +
            '<div class="resolved-key">' + rows[i][0] + '</div>' +
            '<div class="resolved-val">' + escapeHtml(String(rows[i][1])) + '</div>' +
            (src ? '<div class="resolved-src">from ' + escapeHtml(src) + '</div>' : '') +
            '</div>';
        }
        out += '</div>';
      }

      var text;
      try { text = JSON.stringify(received, null, 2); }
      catch (e) { text = String(received); }

      // In the modal there is no card to toggle, so the JSON is simply shown.
      if (gid === undefined) return out + '<pre class="req-json">' + escapeHtml(text) + '</pre>';

      out += '<button class="json-toggle" onclick="event.stopPropagation();toggleJson(\'' + gid + '\')">' +
        (showJson ? '&#9660; hide raw JSON' : '&#9654; show raw JSON') + '</button>';
      if (showJson) out += '<pre class="req-json">' + escapeHtml(text) + '</pre>';
      return out;
    }

    // The receipt: what the model was actually given. Read against the request
    // block above it -- a take in the wrong voice is almost always a field that
    // did not survive the trip from one to the other.
    function renderEnginePayload(eng, showJson, gid) {
      if (!eng) return '';
      var out = '<div class="req-section">What actually went into the model</div>';

      var clip = eng.target_clip || '';
      var clipShort = clip ? clip.split(/[/\\]/).pop() : '(none)';
      var rows = [
        ['voice_id (SeedVC target)', eng.voice_id || '(none)', eng.target_from || ''],
        ['target clip', clipShort, clip],
        ['sex', eng.sex || '(not used)', 'picks the donor, never reaches the model directly'],
        ['donor_set', eng.donor_set || '(none)', 'the actor every emotion was cloned from']
      ];
      out += '<div class="resolved-grid">';
      for (var i = 0; i < rows.length; i++) {
        out += '<div class="resolved-item" title="' + escapeHtml(String(rows[i][2] || '')) + '">' +
          '<div class="resolved-key">' + escapeHtml(rows[i][0]) + '</div>' +
          '<div class="resolved-val">' + escapeHtml(String(rows[i][1])) + '</div>' +
          '</div>';
      }
      out += '</div>';

      var groups = eng.groups || [];
      for (var k = 0; k < groups.length; k++) {
        var gp = groups[k];
        var vc = gp.voice_converted
          ? '<span class="vc-on">donor &#8594; SeedVC &#8594; target</span>'
          : '<span class="vc-off">cloned from the target clip directly (VC skipped)</span>';
        out += '<div class="subjob-row">' +
          '<span class="pill pill-completed">' + escapeHtml(gp.emotion || '?') + '</span>' +
          '<span class="mono">' + escapeHtml(gp.donor_clip || '-') + '</span>' +
          vc +
          '<span style="margin-left:auto;">' + ((gp.pieces || []).length) + ' piece(s) &middot; cfg ' +
          escapeHtml(String(gp.cfg_value)) + ' &middot; ' + escapeHtml(String(gp.timesteps)) +
          ' steps &middot; lora ' + escapeHtml(String(gp.lora)) + '</span>' +
          '</div>';
      }

      var text;
      try { text = JSON.stringify(eng, null, 2); }
      catch (e) { text = String(eng); }

      if (gid === undefined) return out + '<pre class="req-json">' + escapeHtml(text) + '</pre>';

      out += '<button class="json-toggle" onclick="event.stopPropagation();toggleEngine(\'' + gid + '\')">' +
        (showJson ? '&#9660; hide raw JSON' : '&#9654; show raw JSON') + '</button>';
      if (showJson) out += '<pre class="req-json">' + escapeHtml(text) + '</pre>';
      return out;
    }

    function renderChunkRows(rows) {
      if (rows.length === 0) {
        return '<div style="color:var(--text-dim);font-size:12px;padding:4px 2px;">No chunks recorded for this request.</div>';
      }
      var out = '';
      for (var i = 0; i < rows.length; i++) {
        var r = rows[i];
        out += '<div class="chunk-row">' +
          '<span class="n">[' + (i + 1) + '/' + rows.length + ']</span>' +
          '<span class="tx">' + escapeHtml(r.text) + '</span>' +
          '<span class="st"><span class="pill pill-' + r.status + '">' + r.status + '</span></span>' +
          '</div>';
      }
      return out;
    }

    function renderSubJobs(children) {
      var out = '';
      for (var i = 0; i < children.length; i++) {
        var ch = children[i];
        var kindBadge = '';
        if (ch.error_kind === 'oom') kindBadge = '<span class="badge-oom">VRAM OOM</span>';
        else if (ch.error_kind === 'upstream_oom') kindBadge = '<span class="badge-upstream">cancelled by upstream OOM</span>';
        out += '<div class="subjob-row">' +
          '<span class="pill pill-' + ch.status + '">' + ch.status + '</span>' +
          kindBadge +
          '<span class="mono">' + escapeHtml(ch.job_id) + '</span>' +
          '<span>' + ((ch.chunks || []).length) + ' chunk(s)</span>' +
          '<span style="margin-left:auto;">gpu ' + fmtSecs(ch.ran_s) + '</span>' +
          '</div>';
      }
      return out;
    }

    // One line of "why", on the collapsed card. Every failure gets one, not just
    // OOM: an unknown voice or an unreachable GPU service is exactly as worth
    // seeing, and burying it behind a click is how a broken pipeline reads as an
    // idle one.
    function renderErrorStrip(failedJob, status) {
      if (!failedJob || !failedJob.error) return '';
      if (status !== 'failed' && status !== 'cancelled') return '';

      var msg = String(failedJob.error);
      var short = msg.length > 190 ? msg.slice(0, 190) + '…' : msg;
      var oom = failedJob.error_kind === 'oom';
      var muted = (failedJob.status === 'cancelled' && !oom);
      var label = oom ? 'VRAM OOM' : (failedJob.status === 'cancelled' ? 'Cancelled' : 'Failed');

      return '<div class="req-error-strip' + (muted ? ' muted' : '') + '" title="' + escapeHtml(msg) + '">' +
        '<span class="lbl">' + label + '</span>' +
        '<span class="txt">' + escapeHtml(short) + '</span>' +
        '</div>';
    }

    function renderFailureBanner(oomJob, failedJob, upstream) {
      if (oomJob) {
        var extra = '';
        if (upstream.length > 0) {
          var ids = upstream.map(function (u) { return u.job_id; }).join(', ');
          extra = '<div style="margin-top:6px;color:var(--text-muted);font-size:11.5px;">' +
            'The rest of this request was cancelled so it would not keep asking the same GPU for memory: ' +
            escapeHtml(ids) + '</div>';
        }
        return '<div class="oom-banner">' +
          '<span class="badge-oom">VRAM OOM</span>' +
          '<div><div class="msg">' + escapeHtml(oomJob.error || 'CUDA out of memory') + '</div>' + extra + '</div>' +
          '</div>';
      }
      if (!failedJob || !failedJob.error) return '';
      var muted = failedJob.status === 'cancelled';
      return '<div class="req-section" style="color:' + (muted ? 'var(--text-dim)' : 'var(--accent-red)') + ';">' +
          (muted ? 'Cancelled' : 'Failure') + ' — ' + escapeHtml(failedJob.job_id) +
        '</div>' +
        '<div class="fail-box' + (muted ? ' muted' : '') + '">' + escapeHtml(failedJob.error) + '</div>';
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
      // Cancelling one piece abandons the whole request server-side, so say so
      // rather than letting the operator think they stopped one chunk.
      if (!confirm('ยกเลิกทั้ง request นี้ (ทุก chunk / ทุก job ที่เหลือ)?\n\n' + jobId)) return;
      try {
        const res = await fetch('/v2/jobs/' + encodeURIComponent(jobId), {
          method: 'DELETE'
        });
        const data = await res.json();
        if (res.ok && data.cancelled) {
          var also = data.also_cancelled || [];
          if (also.length > 0) {
            console.log('[dashboard] cancelled ' + (also.length + 1) + ' job(s) of this request:', jobId, also);
          }
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

      // What the caller posted, before the studio applied any default to it.
      if (job.request) {
        html += '<div>' +
          '<div class="section-label" style="color:var(--accent-purple);">&#128231; Request received from the caller</div>' +
          renderRequestPayload(job.request) +
          '</div>';
      }

      if (job.engine) {
        html += '<div>' +
          '<div class="section-label" style="color:var(--accent-cyan);">&#9881; What actually went into the model</div>' +
          renderEnginePayload(job.engine) +
          '</div>';
      }

      // Raw User Prompt Section
      if (job.raw_prompt && job.raw_prompt.trim().length > 0) {
        html += '<div>' +
          '<div class="section-label" style="color:var(--accent-purple);">📝 Raw User Prompt (:8013 / Studio)</div>' +
          '<div class="raw-prompt-container" style="padding:12px 16px; font-size:13.5px; border-radius:8px;">' +
          '<div class="raw-prompt-title" style="margin-bottom:6px;"><span class="badge-raw-prompt">RAW PROMPT INPUT</span></div>' +
          '<div style="white-space:pre-wrap; line-height:1.6;">' + escapeHtml(job.raw_prompt) + '</div>' +
          '</div>' +
          '</div>';
      }

      // Text Chunks
      if (job.chunks && job.chunks.length > 0) {
        html += '<div>' +
          '<div class="section-label">💬 Processed Text Chunks (' + job.chunks.length + ')</div>';
        for (var c = 0; c < job.chunks.length; c++) {
          html += '<div class="chunk-box"><span class="chunk-idx">[' + (c+1) + '/' + job.chunks.length + ']</span> ' + escapeHtml(job.chunks[c]) + '</div>';
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
