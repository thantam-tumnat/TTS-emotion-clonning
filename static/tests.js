'use strict';

(function () {
  const resultsEl = document.getElementById('results');
  const emptyEl = document.getElementById('empty');
  const summaryEl = document.getElementById('summary');
  const runAllBtn = document.getElementById('run-all');

  const ICON = { passed: '✓', failed: '✗', error: '✗', skipped: '⊘' };

  let suites = [];

  // Load the suite catalogue up front so the page shows the groups and a per-suite
  // run button even before anything has run.
  async function loadSuites() {
    try {
      const res = await fetch('/api/tests/suites');
      const data = await res.json();
      suites = data.suites || [];
      renderSkeleton();
    } catch (e) {
      emptyEl.textContent = 'โหลดรายการ test ไม่ได้: ' + e.message;
    }
  }

  function renderSkeleton() {
    if (!suites.length) return;
    emptyEl.style.display = 'none';
    resultsEl.innerHTML = suites.map(s => `
      <div class="suite" id="suite-${s.id}">
        <div class="suite-head">
          <span class="suite-status" id="status-${s.id}">•</span>
          <span class="suite-title">${escapeHtml(s.label)}</span>
          <span class="badge" id="badge-${s.id}"></span>
          <button class="btn-suite" data-suite="${s.id}">รัน</button>
          <span class="suite-desc">${escapeHtml(s.description || '')}</span>
        </div>
        <ul class="tests" id="tests-${s.id}"></ul>
      </div>
    `).join('');
    resultsEl.querySelectorAll('button[data-suite]').forEach(btn => {
      btn.addEventListener('click', () => runSuite(btn.dataset.suite));
    });
  }

  function setSuiteRunning(id) {
    const status = document.getElementById(`status-${id}`);
    if (status) status.innerHTML = '<span class="spinner"></span>';
    const badge = document.getElementById(`badge-${id}`);
    if (badge) badge.textContent = 'กำลังรัน...';
    const list = document.getElementById(`tests-${id}`);
    if (list) list.innerHTML = '';
    const old = document.querySelector(`#suite-${id} .run-error`);
    if (old) old.remove();
  }

  function renderSuiteResult(r) {
    const status = document.getElementById(`status-${r.id}`);
    const badge = document.getElementById(`badge-${r.id}`);
    const list = document.getElementById(`tests-${r.id}`);
    if (!list) return;

    if (status) {
      status.textContent = r.ok ? '✅' : (r.total ? '❌' : '⚠️');
      status.style.color = r.ok ? 'var(--pass)' : 'var(--fail)';
    }
    if (badge) {
      const c = r.counts || {};
      badge.innerHTML =
        `<span class="p">${c.passed || 0} ผ่าน</span>` +
        ((c.failed || c.error) ? ` <span class="f">${(c.failed || 0) + (c.error || 0)} ไม่ผ่าน</span>` : '') +
        ((c.skipped) ? ` <span class="s">${c.skipped} ข้าม</span>` : '') +
        ` <span>· ${r.elapsed_s}s</span>`;
    }

    list.innerHTML = (r.tests || []).map((t, i) => `
      <li class="test ${t.status}" data-idx="${i}">
        <span class="t-icon ${t.status}">${ICON[t.status] || '•'}</span>
        <span class="t-name">${escapeHtml(t.name)}</span>
        <span class="t-time">${t.duration_s}s</span>
      </li>
      ${t.message ? `<div class="t-msg" id="msg-${r.id}-${i}">${escapeHtml(t.message)}</div>` : ''}
    `).join('');

    // Click a failed test to reveal its assertion message.
    list.querySelectorAll('li.test.failed, li.test.error').forEach(li => {
      li.addEventListener('click', () => {
        const msg = document.getElementById(`msg-${r.id}-${li.dataset.idx}`);
        if (msg) msg.style.display = msg.style.display === 'block' ? 'none' : 'block';
      });
    });

    if (r.run_error) {
      const div = document.createElement('div');
      div.className = 'run-error';
      div.textContent = 'รัน suite ไม่สำเร็จ:\n' + r.run_error;
      document.getElementById(`suite-${r.id}`).appendChild(div);
    }
  }

  async function runSuite(id) {
    setBusy(true);
    setSuiteRunning(id);
    try {
      const res = await fetch(`/api/tests/run?suite_id=${encodeURIComponent(id)}`, { method: 'POST' });
      const r = await res.json();
      renderSuiteResult(r);
    } catch (e) {
      renderSuiteResult({ id, ok: false, total: 0, counts: {}, tests: [], elapsed_s: 0, run_error: e.message });
    } finally {
      setBusy(false);
      updateSummary();
    }
  }

  async function runAll() {
    setBusy(true);
    suites.forEach(s => setSuiteRunning(s.id));
    summaryEl.innerHTML = '<span class="spinner"></span> กำลังรันทั้งหมด...';
    try {
      const res = await fetch('/api/tests/run', { method: 'POST' });
      const data = await res.json();
      (data.suites || []).forEach(renderSuiteResult);
      renderGlobalSummary(data);
    } catch (e) {
      summaryEl.textContent = 'รันไม่สำเร็จ: ' + e.message;
    } finally {
      setBusy(false);
    }
  }

  function renderGlobalSummary(data) {
    const failed = data.failed || 0;
    summaryEl.innerHTML =
      `${data.ok ? '✅ ผ่านทั้งหมด' : '❌ มีที่ไม่ผ่าน'} — ` +
      `<span class="pass">${data.passed || 0} ผ่าน</span>` +
      (failed ? ` / <span class="fail">${failed} ไม่ผ่าน</span>` : '') +
      ` / ${data.total || 0} ทั้งหมด`;
  }

  function updateSummary() {
    // Recompute from what is currently rendered (used after single-suite runs).
    let passed = 0, failed = 0, total = 0;
    document.querySelectorAll('li.test').forEach(li => {
      total++;
      if (li.classList.contains('passed')) passed++;
      else if (li.classList.contains('failed') || li.classList.contains('error')) failed++;
    });
    if (!total) return;
    summaryEl.innerHTML =
      `<span class="pass">${passed} ผ่าน</span>` +
      (failed ? ` / <span class="fail">${failed} ไม่ผ่าน</span>` : '') +
      ` / ${total} ทั้งหมด`;
  }

  function setBusy(busy) {
    runAllBtn.disabled = busy;
    document.querySelectorAll('button[data-suite]').forEach(b => (b.disabled = busy));
  }

  function escapeHtml(s) {
    return String(s == null ? '' : s)
      .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;');
  }

  runAllBtn.addEventListener('click', runAll);
  loadSuites();
})();
