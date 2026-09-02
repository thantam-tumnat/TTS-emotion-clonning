# SiangTTS Go Fiber Queue Service (:8020)

High-performance, ultra-low latency Job Queue Gateway for SiangTTS and VoxCPM2 Voice Cloning, built with **Go (Golang)** and **Fiber v2** (`fasthttp`).

---

## Architecture

```
[Clients: Webhook (:8010) & Tone Studio (:8011)]
                       │
                       ▼ HTTP (:8020)
┌─────────────────────────────────────────────────────────────┐
│  Go Fiber Queue Gateway (:8020)                             │
│  • High-concurrency job dispatcher (Goroutines & Channels)  │
│  • Priority scheduling (Interactive burst vs Batch queue)   │
│  • Non-blocking state management (sync.RWMutex)             │
│  • Reverse proxy for voice caching & health endpoints       │
└──────────────────────────────┬──────────────────────────────┘
                               │ HTTP (:8021) /v2/direct_render
                               ▼
┌─────────────────────────────────────────────────────────────┐
│  Python GPU Worker (:8021)                                  │
│  • Pure PyTorch / CUDA Inference (VoxCPM2 + Thai LoRA)      │
└─────────────────────────────────────────────────────────────┘
```

---

## Features
* **Zero GIL Bottleneck**: Handles tens of thousands of concurrent polling connections with minimal RAM and CPU.
* **Fault Isolation**: If the Python GPU worker encounters CUDA OOM and crashes, the Go queue preserves all waiting jobs.
* **Multi-Lane Priority**: Studio interactive requests (`lane=interactive`) jump ahead of background batch requests (`lane=batch`).
* **100% Backward Compatible**: Implements the exact same `/v2/jobs` API contract as the original Python service.

### Requests, not jobs

A take from the :8013 studio is split into **one render job per emotion**, because
every piece of an emotion has to share that emotion's donor prompt cache. Those
jobs carry a `parent_id` naming the request they came from, and the dashboard
renders **one card per request**: chunk count, every chunk's text, per-chunk
status, and the render jobs underneath. A job with no `parent_id` is a request of
one, so nothing is hidden and nothing is duplicated.

```jsonc
POST /v2/jobs/render
{ "parent_id": "req_1756...", "chunks": ["(angry) ..."], ... }
```

### CUDA OOM tears down the whole request

An OOM is not the job's fault and cannot be retried against the same GPU state, so
it is classified rather than reported as a generic failure:

* the GPU service answers **503** with `"error_kind": "oom"` (the SeedVC worker
  on :8022 does the same for a failed conversion);
* the queue marks that job `failed` / `error_kind=oom` and **cancels every
  sibling of the same request** with `error_kind=upstream_oom` — a take missing a
  chunk cannot be assembled, and finishing the rest would ask the card that just
  ran out for more memory;
* the dashboard shows the card with a **VRAM OOM** badge, the allocator's own
  message, and which siblings went down with it. The **VRAM OOM** counter at the
  top counts requests, not chunks.

Cancelling from the dashboard follows the same rule: one click abandons the whole
request, and the response lists what else was cancelled in `also_cancelled`.

---

## Quick Start

### 1. Build and Run
```bash
cd voice-clonning-queue-go
go run main.go
```
Or run `start_queue.bat` on Windows.

### 2. Environment Variables
| Variable | Default | Description |
| :--- | :--- | :--- |
| `PORT` | `8020` | Port for the Go Fiber Queue Gateway |
| `PYTHON_GPU_URL` | `http://127.0.0.1:8021` | Target URL for the Python PyTorch GPU Worker |
