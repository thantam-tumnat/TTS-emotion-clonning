document.addEventListener('DOMContentLoaded', () => {
  // DOM Elements - Input Side
  const textInput = document.getElementById('text-input');
  const guidanceInput = document.getElementById('guidance-input');
  const charCounter = document.getElementById('char-counter');
  const engineSelect = document.getElementById('engine-select');
  const btnProcess = document.getElementById('btn-process');
  const btnClear = document.getElementById('btn-clear');
  const presetButtons = document.querySelectorAll('.preset-btn');
  const healthStatus = document.getElementById('health-status');
  
  // DOM Elements - Output / Editor Side
  const modelBadge = document.getElementById('model-badge');
  const modelName = document.getElementById('model-name');
  const fallbackIndicator = document.getElementById('fallback-indicator');

  const tabButtons = document.querySelectorAll('.tab-btn');
  const tabPanes = document.querySelectorAll('.tab-pane');
  
  const loadingState = document.getElementById('loading-state');
  const emptyState = document.getElementById('empty-state');
  
  const outputEditableText = document.getElementById('output-editable-text');
  const outputCharCounter = document.getElementById('output-char-counter');
  const liveTagPreview = document.getElementById('live-tag-preview');
  
  const geminiPromptSection = document.getElementById('gemini-prompt-section');
  const geminiPromptEditable = document.getElementById('gemini-prompt-editable');
  
  const segmentsContainer = document.getElementById('segments-container');
  const rawJson = document.getElementById('raw-json');

  const btnCopyOutput = document.getElementById('btn-copy-output');
  const btnCopyPrompt = document.getElementById('btn-copy-prompt');
  const tagInsertButtons = document.querySelectorAll('.tag-insert-btn');

  // Presets Data
  const PRESETS = {
    shift: {
      text: 'ขอโทษนะ ฉันไม่ได้ตั้งใจ แต่เธอก็ไม่ฟังฉันเลย',
      guidance: 'ท่อนแรกขอเศร้าขอโทษจากใจ ท่อนหลังตัดพ้อโกรธ'
    },
    sarcastic: {
      text: 'แหม เก่งจังเลยนะ ทำพังหมดทั้งห้องแล้วเนี่ย',
      guidance: 'ประชดประชันแดกดันอย่างแรง'
    },
    happy: {
      text: 'ยินดีด้วยนะ! ในที่สุดก็ทำสำเร็จแล้ว สุดยอดไปเลย!',
      guidance: 'ดีใจสุดขีด ร่าเริงมาก'
    },
    calm: {
      text: 'หายใจเข้าลึกๆ ผ่อนคลาย แล้วค่อยๆ ปล่อยวางทุกอย่างลงนะ',
      guidance: 'สงบ นุ่มนวล ช้าๆ'
    },
    news: {
      text: 'กรมอุตุนิยมวิทยาประกาศเตือน จะมีฝนตกหนักถึงหนักมากในหลายพื้นที่ ประชาชนควรระมัดระวังน้ำท่วมฉับพลัน',
      guidance: 'อ่านข่าว สุภาพ เป็นทางการ เป็นกลาง'
    }
  };

  // Check API Health
  async function checkHealth() {
    try {
      const res = await fetch('/health');
      if (res.ok) {
        healthStatus.className = 'status-indicator online';
        healthStatus.querySelector('.status-label').textContent = 'API พร้อมใช้งาน (Online)';
      } else {
        throw new Error('Health check failed');
      }
    } catch (e) {
      healthStatus.className = 'status-indicator offline';
      healthStatus.querySelector('.status-label').textContent = 'ไม่สามารถเชื่อมต่อ API ได้';
    }
  }
  checkHealth();
  setInterval(checkHealth, 30000);

  // Character Counter for Input
  textInput.addEventListener('input', () => {
    const len = textInput.value.length;
    charCounter.textContent = `${len.toLocaleString()} ตัวอักษร`;
  });

  // Character Counter & Live Highlight for Output Textarea
  function updateOutputPreview() {
    const val = outputEditableText.value;
    outputCharCounter.textContent = `${val.length.toLocaleString()} ตัวอักษร`;
    liveTagPreview.innerHTML = highlightAudioTags(escapeHtml(val)) || '<span style="color:var(--text-muted);">ไม่มีข้อความ</span>';
  }

  outputEditableText.addEventListener('input', updateOutputPreview);

  // Presets click
  presetButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      const key = btn.getAttribute('data-preset');
      if (PRESETS[key]) {
        textInput.value = PRESETS[key].text;
        guidanceInput.value = PRESETS[key].guidance || '';
        textInput.dispatchEvent(new Event('input'));
        textInput.focus();
      }
    });
  });

  // Clear button
  btnClear.addEventListener('click', () => {
    textInput.value = '';
    guidanceInput.value = '';
    textInput.dispatchEvent(new Event('input'));
    showEmptyState();
  });

  // Tab switching
  tabButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      const targetTab = btn.getAttribute('data-tab');
      
      tabButtons.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');

      tabPanes.forEach(pane => {
        if (pane.id === `tab-${targetTab}`) {
          pane.classList.remove('hidden');
        } else {
          pane.classList.add('hidden');
        }
      });
    });
  });

  function showLoading(isLoading) {
    if (isLoading) {
      loadingState.classList.remove('hidden');
      emptyState.classList.add('hidden');
      tabPanes.forEach(pane => pane.classList.add('hidden'));
      btnProcess.disabled = true;
    } else {
      loadingState.classList.add('hidden');
      btnProcess.disabled = false;
    }
  }

  function showEmptyState() {
    emptyState.classList.remove('hidden');
    loadingState.classList.add('hidden');
    tabPanes.forEach(pane => pane.classList.add('hidden'));
    modelBadge.classList.add('hidden');
    outputEditableText.value = '';
    liveTagPreview.innerHTML = '';
  }

  function formatIntensityStars(intensity) {
    if (intensity === 1) return '●○○ (Mild)';
    if (intensity === 3) return '●●● (Strong)';
    return '●●○ (Standard)';
  }

  function highlightAudioTags(text) {
    // Replace [tag] with highlighted badge span
    return text.replace(/(\[[a-zA-Z\s]+\])/g, '<span class="tag-highlight">$1</span>');
  }

  function escapeHtml(str) {
    if (!str) return '';
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
  }

  // Tag Inserter Toolbar
  tagInsertButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      const tag = btn.getAttribute('data-tag');
      insertTextAtCursor(outputEditableText, tag);
      updateOutputPreview();
      outputEditableText.focus();
    });
  });

  function insertTextAtCursor(textarea, textToInsert) {
    const startPos = textarea.selectionStart;
    const endPos = textarea.selectionEnd;
    const currentVal = textarea.value;

    textarea.value = currentVal.substring(0, startPos) + textToInsert + currentVal.substring(endPos);
    textarea.selectionStart = textarea.selectionEnd = startPos + textToInsert.length;
    textarea.dispatchEvent(new Event('input'));
  }

  // Process Annotation & Speak
  async function handleAnnotate() {
    const text = textInput.value.trim();
    if (!text) {
      alert('กรุณากรอกข้อความภาษาไทยก่อนกดวิเคราะห์');
      textInput.focus();
      return;
    }

    const guidance = guidanceInput.value.trim();
    const engine = engineSelect.value;
    showLoading(true);

    try {
      const response = await fetch('/speak', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          text: text,
          guidance: guidance || null,
          engine: engine
        })
      });

      if (!response.ok) {
        const errData = await response.json().catch(() => ({}));
        throw new Error(errData.detail || `Server error: ${response.status}`);
      }

      const data = await response.json();
      renderResults(data, engine);
    } catch (err) {
      alert(`เกิดข้อผิดพลาดในการประมวลผล: ${err.message}`);
      showEmptyState();
    } finally {
      showLoading(false);
    }
  }

  function renderResults(data, engine) {
    // Show model badge
    modelBadge.classList.remove('hidden');
    modelName.textContent = data.model_used;
    if (data.fallback) {
      fallbackIndicator.className = 'badge-tag fallback';
      fallbackIndicator.textContent = 'Fallback Neutral';
    } else {
      fallbackIndicator.className = 'badge-tag normal';
      fallbackIndicator.textContent = 'Normal';
    }

    // 1. Populate Editable Output Textarea
    outputEditableText.value = data.text;
    updateOutputPreview();

    // Populate Gemini Prompt if available
    if (data.prompt) {
      geminiPromptSection.classList.remove('hidden');
      geminiPromptEditable.value = data.prompt;
    } else {
      geminiPromptSection.classList.add('hidden');
      geminiPromptEditable.value = '';
    }

    // 2. Render Segments Tab
    segmentsContainer.innerHTML = '';
    data.segments.forEach((seg, idx) => {
      const item = document.createElement('div');
      item.className = `segment-item border-${seg.tone}`;
      item.innerHTML = `
        <div class="segment-meta">
          <div class="segment-meta-left">
            <span class="seg-index">#${idx + 1}</span>
            <span class="tone-chip tone-${seg.tone}">${seg.tone}</span>
          </div>
          <span class="intensity-stars">${formatIntensityStars(seg.intensity)}</span>
        </div>
        <div class="segment-text">${escapeHtml(seg.text)}</div>
      `;
      segmentsContainer.appendChild(item);
    });

    // 3. Raw JSON Tab
    rawJson.textContent = JSON.stringify(data, null, 2);

    // Default to editor tab or preserve current
    const activeTab = document.querySelector('.tab-btn.active').getAttribute('data-tab');
    document.getElementById(`tab-${activeTab}`).classList.remove('hidden');
  }

  // Copy helper
  function setupCopyBtn(btn, getSourceText) {
    btn.addEventListener('click', async () => {
      const text = getSourceText();
      if (!text) return;
      try {
        await navigator.clipboard.writeText(text);
        const originalHtml = btn.innerHTML;
        btn.innerHTML = `
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#4ade80" stroke-width="2"><polyline points="20 6 9 17 4 12"></polyline></svg>
          <span style="color:#4ade80;">คัดลอกแล้ว!</span>
        `;
        setTimeout(() => {
          btn.innerHTML = originalHtml;
        }, 1800);
      } catch (e) {
        alert('ไม่สามารถคัดลอกข้อความได้');
      }
    });
  }

  setupCopyBtn(btnCopyOutput, () => outputEditableText.value);
  setupCopyBtn(btnCopyPrompt, () => geminiPromptEditable.value);

  btnProcess.addEventListener('click', handleAnnotate);

  // Allow Ctrl+Enter to trigger annotate
  textInput.addEventListener('keydown', (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
      e.preventDefault();
      handleAnnotate();
    }
  });
  if (guidanceInput) {
    guidanceInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') {
        e.preventDefault();
        handleAnnotate();
      }
    });
  }
});
