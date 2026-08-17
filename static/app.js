document.addEventListener('DOMContentLoaded', () => {
  // DOM Elements - Input Side
  const textInput = document.getElementById('text-input');
  const guidanceInput = document.getElementById('guidance-input');
  const charCounter = document.getElementById('char-counter');
  const engineSelect = document.getElementById('engine-select');
  const speakerSelect = document.getElementById('speaker-select');
  const speakerGroup = document.getElementById('speaker-group');
  const uploadVoiceArea = document.getElementById('upload-voice-area');
  const audioFileInput = document.getElementById('audio-file-input');
  const dropZone = document.getElementById('drop-zone');
  const dropZoneContent = document.getElementById('drop-zone-content');
  const fileInfoBadge = document.getElementById('file-info-badge');
  const selectedFilename = document.getElementById('selected-filename');
  const btnRemoveFile = document.getElementById('btn-remove-file');
  const btnSaveSpeaker = document.getElementById('btn-save-speaker');

  const paramCfg = document.getElementById('param-cfg');
  const paramSteps = document.getElementById('param-steps');
  const valCfg = document.getElementById('val-cfg');
  const valSteps = document.getElementById('val-steps');

  const btnProcess = document.getElementById('btn-process');
  const btnSynthesizeDirect = document.getElementById('btn-synthesize-direct');
  const btnClear = document.getElementById('btn-clear');
  const presetButtons = document.querySelectorAll('.preset-btn');
  const healthStatus = document.getElementById('health-status');
  
  // DOM Elements - Output / Editor & Audio Player Side
  const modelBadge = document.getElementById('model-badge');
  const modelName = document.getElementById('model-name');
  const fallbackIndicator = document.getElementById('fallback-indicator');

  const tabButtons = document.querySelectorAll('.tab-btn');
  const tabPanes = document.querySelectorAll('.tab-pane');
  
  const loadingState = document.getElementById('loading-state');
  const loadingText = document.getElementById('loading-text');
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

  const audioPlayerCard = document.getElementById('audio-player-card');
  const audioPlayer = document.getElementById('audio-player');
  const btnDownloadAudio = document.getElementById('btn-download-audio');

  let selectedAudioFile = null;
  let currentAudioUrl = null;

  // Presets Data
  const PRESETS = {
    calm: {
      text: '[calm] หายใจเข้าลึกๆ ผ่อนคลาย แล้วค่อยๆ ปล่อยวางทุกอย่างลงนะ',
      guidance: 'สงบ นุ่มนวล ช้าๆ ผ่อนคลาย'
    },
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
    news: {
      text: 'กรมอุตุนิยมวิทยาประกาศเตือน จะมีฝนตกหนักถึงหนักมากในหลายพื้นที่ ประชาชนควรระมัดระวังน้ำท่วมฉับพลัน',
      guidance: 'อ่านข่าว สุภาพ เป็นทางการ เป็นกลาง'
    }
  };

  // API Base URL (auto-detect port or direct file load)
  const API_BASE = (window.location.protocol === 'file:' || (window.location.port && window.location.port !== '8000'))
    ? 'http://127.0.0.1:8000'
    : '';

  // Slider updates
  paramCfg.addEventListener('input', () => {
    valCfg.textContent = parseFloat(paramCfg.value).toFixed(1);
  });
  paramSteps.addEventListener('input', () => {
    valSteps.textContent = paramSteps.value;
  });

  // Check API Health & Fetch Speakers
  async function checkHealthAndSpeakers() {
    try {
      const res = await fetch(`${API_BASE}/health`);
      if (res.ok) {
        const data = await res.json();
        healthStatus.className = 'status-indicator online';
        healthStatus.querySelector('.status-label').textContent = `API พร้อมใช้งาน (${data.speakers_count || 0} voices)`;
      } else {
        throw new Error('Health check failed');
      }
    } catch (e) {
      healthStatus.className = 'status-indicator offline';
      healthStatus.querySelector('.status-label').textContent = 'ไม่สามารถเชื่อมต่อ API ได้ (กรุณารันเซิร์ฟเวอร์ที่พอร์ต 8000)';
    }

    loadSpeakersList();
  }

  async function loadSpeakersList() {
    try {
      const res = await fetch(`${API_BASE}/speakers`);
      if (res.ok) {
        const data = await res.json();
        const prevVal = speakerSelect.value;
        speakerSelect.innerHTML = '<option value="">-- ไม่ใช้เสียงโคลน (Base Voice) --</option>';
        (data.speakers || []).forEach(spk => {
          const opt = document.createElement('option');
          opt.value = spk.id;
          opt.textContent = `🎙️ ${spk.name} (${spk.filename})`;
          speakerSelect.appendChild(opt);
        });
        if (prevVal) speakerSelect.value = prevVal;
      }
    } catch (e) {
      console.warn('Could not load speakers:', e);
    }
  }

  checkHealthAndSpeakers();
  setInterval(checkHealthAndSpeakers, 30000);

  // Engine Switch Visibility
  function updateEngineVisibility() {
    const eng = engineSelect.value;
    if (eng === 'voxcpm' || eng === 'siangtts') {
      speakerGroup.classList.remove('hidden');
      uploadVoiceArea.classList.remove('hidden');
    } else {
      speakerGroup.classList.add('hidden');
      uploadVoiceArea.classList.add('hidden');
    }
  }
  engineSelect.addEventListener('change', updateEngineVisibility);
  updateEngineVisibility();

  // File Upload & Drag-and-Drop
  dropZone.addEventListener('click', () => {
    audioFileInput.click();
  });

  audioFileInput.addEventListener('change', (e) => {
    if (e.target.files && e.target.files[0]) {
      handleAudioFileSelected(e.target.files[0]);
    }
  });

  ['dragenter', 'dragover'].forEach(eventName => {
    dropZone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropZone.classList.add('drag-over');
    });
  });

  ['dragleave', 'drop'].forEach(eventName => {
    dropZone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropZone.classList.remove('drag-over');
    });
  });

  dropZone.addEventListener('drop', (e) => {
    if (e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleAudioFileSelected(e.dataTransfer.files[0]);
    }
  });

  function handleAudioFileSelected(file) {
    selectedAudioFile = file;
    selectedFilename.textContent = `🎵 ${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
    dropZoneContent.classList.add('hidden');
    fileInfoBadge.classList.remove('hidden');
    btnSaveSpeaker.classList.remove('hidden');
  }

  btnRemoveFile.addEventListener('click', (e) => {
    e.stopPropagation();
    selectedAudioFile = null;
    audioFileInput.value = '';
    dropZoneContent.classList.remove('hidden');
    fileInfoBadge.classList.add('hidden');
    btnSaveSpeaker.classList.add('hidden');
  });

  // Save Speaker Profile
  btnSaveSpeaker.addEventListener('click', async (e) => {
    e.stopPropagation();
    if (!selectedAudioFile) return;

    const defaultName = selectedAudioFile.name.replace(/\.[^/.]+$/, "");
    const speakerName = prompt("ตั้งชื่อโปรไฟล์เสียง (Speaker ID / Name):", defaultName);
    if (!speakerName) return;

    const formData = new FormData();
    formData.append('file', selectedAudioFile);
    formData.append('speaker_id', speakerName);

    try {
      btnSaveSpeaker.disabled = true;
      btnSaveSpeaker.textContent = "กำลังบันทึก...";
      const res = await fetch(`${API_BASE}/speakers`, {
        method: 'POST',
        body: formData
      });
      if (!res.ok) throw new Error("Upload failed");
      const result = await res.json();
      alert(`บันทึกเสียง '${result.name}' สำเร็จ!`);
      await loadSpeakersList();
      speakerSelect.value = result.id;
    } catch (err) {
      alert(`ไม่สามารถบันทึกเสียงได้: ${err.message}`);
    } finally {
      btnSaveSpeaker.disabled = false;
      btnSaveSpeaker.textContent = "บันทึกเป็นโปรไฟล์เสียง";
    }
  });

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

  function showLoading(isLoading, customText = 'กำลังวิเคราะห์ข้อความด้วย LLM...') {
    if (isLoading) {
      loadingText.textContent = customText;
      loadingState.classList.remove('hidden');
      emptyState.classList.add('hidden');
      tabPanes.forEach(pane => pane.classList.add('hidden'));
      btnProcess.disabled = true;
      btnSynthesizeDirect.disabled = true;
    } else {
      loadingState.classList.add('hidden');
      btnProcess.disabled = false;
      btnSynthesizeDirect.disabled = false;
    }
  }

  function showEmptyState() {
    emptyState.classList.remove('hidden');
    loadingState.classList.add('hidden');
    tabPanes.forEach(pane => pane.classList.add('hidden'));
    modelBadge.classList.add('hidden');
    outputEditableText.value = '';
    liveTagPreview.innerHTML = '';
    audioPlayerCard.classList.add('hidden');
  }

  function formatIntensityStars(intensity) {
    if (intensity === 1) return '●○○ (Mild)';
    if (intensity === 3) return '●●● (Strong)';
    return '●●○ (Standard)';
  }

  function highlightAudioTags(text) {
    // Replace [tag] or (Instruction) with highlighted badge span
    let formatted = text.replace(/(\[[a-zA-Z\s]+\])/g, '<span class="tag-highlight">$1</span>');
    formatted = formatted.replace(/(\([a-zA-Z\s,.-]+\))/g, '<span class="instruction-highlight">$1</span>');
    return formatted;
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

  // Process Annotation & Render Script
  async function handleAnnotate() {
    const text = textInput.value.trim();
    if (!text) {
      alert('กรุณากรอกข้อความภาษาไทย หรือคลิกเลือกตัวอย่าง Preset ด้านบนก่อนกดวิเคราะห์');
      textInput.focus();
      return;
    }

    const guidance = guidanceInput.value.trim();
    const engine = engineSelect.value;
    showLoading(true, 'กำลังวิเคราะห์ข้อความและจัดเรียงโทนอารมณ์...');

    try {
      const response = await fetch(`${API_BASE}/speak`, {
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
      alert(`เกิดข้อผิดพลาดในการประมวลผล: ${err.message}\n(โปรดตรวจสอบว่าเปิดเซิร์ฟเวอร์ backend หรือยัง)`);
      showEmptyState();
    } finally {
      showLoading(false);
    }
  }

  // Synthesize Speech
  async function handleSynthesize() {
    const text = (outputEditableText.value.trim() || textInput.value.trim());
    if (!text) {
      alert('กรุณากรอกข้อความภาษาไทยก่อนสังเคราะห์เสียง');
      textInput.focus();
      return;
    }

    const engine = engineSelect.value;
    const speakerId = speakerSelect.value || null;
    const cfgValue = parseFloat(paramCfg.value) || 2.5;
    const timesteps = parseInt(paramSteps.value, 10) || 10;
    const guidance = guidanceInput.value.trim();

    showLoading(true, '🎙️ กำลังสังเคราะห์เสียงด้วย SiangTTS (VoxCPM2)...');

    try {
      let response;
      if (selectedAudioFile) {
        // Upload custom audio file directly
        const formData = new FormData();
        formData.append('text', text);
        formData.append('file', selectedAudioFile);
        if (guidance) formData.append('guidance', guidance);
        formData.append('cfg_value', cfgValue);
        formData.append('inference_timesteps', timesteps);
        formData.append('auto_annotate', 'true');

        response = await fetch(`${API_BASE}/synthesize/upload`, {
          method: 'POST',
          body: formData
        });
      } else {
        // JSON Synthesize request
        response = await fetch(`${API_BASE}/synthesize`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({
            text: text,
            speaker_id: speakerId,
            guidance: guidance || null,
            engine: engine,
            cfg_value: cfgValue,
            inference_timesteps: timesteps,
            auto_annotate: true
          })
        });
      }

      if (!response.ok) {
        const errData = await response.json().catch(() => ({}));
        throw new Error(errData.detail || `Synthesis failed: ${response.status}`);
      }

      const audioBlob = await response.blob();
      if (currentAudioUrl) {
        URL.revokeObjectURL(currentAudioUrl);
      }
      currentAudioUrl = URL.createObjectURL(audioBlob);

      // Play audio in player
      audioPlayer.src = currentAudioUrl;
      btnDownloadAudio.href = currentAudioUrl;
      btnDownloadAudio.download = `siangtts_${speakerId || 'voice'}_${Date.now()}.wav`;
      audioPlayerCard.classList.remove('hidden');
      audioPlayer.play().catch(() => {});

      // Also trigger text annotation render if output was empty
      if (!outputEditableText.value.trim()) {
        handleAnnotate();
      } else {
        const activeTab = document.querySelector('.tab-btn.active').getAttribute('data-tab');
        document.getElementById(`tab-${activeTab}`).classList.remove('hidden');
      }
    } catch (err) {
      alert(`เกิดข้อผิดพลาดในการสังเคราะห์เสียง: ${err.message}`);
    } finally {
      showLoading(false);
    }
  }

  function renderResults(data, engine) {
    emptyState.classList.add('hidden');
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
    if (data.prompt && engine === 'gemini') {
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

    // Default to editor tab
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
  btnSynthesizeDirect.addEventListener('click', handleSynthesize);

  // Allow Ctrl+Enter to trigger annotate or synthesize
  textInput.addEventListener('keydown', (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
      e.preventDefault();
      handleSynthesize();
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
