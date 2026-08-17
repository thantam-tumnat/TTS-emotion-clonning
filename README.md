# Thai TTS Tone Annotation & Voice Cloning Studio (SiangTTS / VoxCPM2)

FastAPI service for analyzing emotional tones and intensities in Thai text and synthesizing expressive speech with **Zero-Shot & Cached Voice Cloning** via **SiangTTS (VoxCPM2 + Thai LoRA)**, ElevenLabs audio tags, or Gemini prompt instructions.

---

## Architecture Overview

```
Thai Raw Text / Script (e.g. '[calm] หายใจเข้า...') ───► 1. SEGMENT & ANNOTATE (PyThaiNLP + LLM)
                                                                 │
                                                                 ▼
Uploaded Reference Audio / Registered Speaker ────────► 2. VOICE CLONING (SiangTTS Speaker Cache)
                                                                 │
                                                                 ▼
                                                        3. EMOTION RENDERER
                                                           (Tone -> VoxCPM Instruction)
                                                                 │
                                                                 ▼
                                                        4. SYNTHESIS ENGINE
                                                           (VoxCPM2 + dubbing-ai/SiangTTS-VoxCPM2-Thai-LoRA)
                                                                 │
                                                                 ▼
                                                        5. 48kHz WAV Audio Output
```

---

## Tone Enum & Engine Mapping

| Tone | VoxCPM2 / SiangTTS Instruction | ElevenLabs Tag | Gemini Prompt (Thai) |
|---|---|---|---|
| `neutral` | *(no instruction)* | *(no tag)* | น้ำเสียงปกติ เป็นกลาง |
| `sad` | `(Sad and melancholic voice, slight sighs)` | `[sad]` | เศร้า สะเทือนใจ |
| `happy` | `(Happy and cheerful voice, smiling while speaking)` | `[happily]` | ร่าเริง ยิ้มขณะพูด |
| `angry` | `(Angry, firm and aggressive tone)` | `[angry]` | โกรธ เสียงแข็ง |
| `excited` | `(Excited and energetic tone)` | `[excited]` | ตื่นเต้น กระตือรือร้น |
| `calm` | `(Calm and soothing voice, speaking softly)` | `[calm]` | สงบ นุ่มนวล พูดช้า |
| `nervous` | `(Nervous and trembling voice, hesitant)` | `[nervous]` | ประหม่า ลังเล |
| `sarcastic` | `(Sarcastic and mocking tone)` | `[sarcastic]` | ประชด แดกดัน |

Intensity levels for VoxCPM (1: Mild / 2: Standard / 3: Strong).

---

## Setup & Running

### 1. Install Dependencies
```bash
py -m pip install -r requirements.txt
```
*(For GPU inference: install `voxcpm`, `soundfile`, `torch` with CUDA)*

### 2. Configure Environment
Copy `.env.example` to `.env` and configure:
```env
GEMINI_API_KEY=your_gemini_api_key
LLM_PROVIDER=gemini
SIANGTTS_BASE_MODEL=openbmb/VoxCPM2
SIANGTTS_ADAPTER=dubbing-ai/SiangTTS-VoxCPM2-Thai-LoRA
```

### 3. Run FastAPI Server & Web Studio
```bash
py -m uvicorn app.main:app --reload --port 8000
```
Then open your browser and navigate to:
👉 **`http://localhost:8000/`** to access the interactive **Thai TTS & Voice Cloning Studio**.

### 4. Run Test Suite
```bash
py -m pytest -v
```

---

## Web Studio Features

- **Interactive Script & Emotion Editor**: Real-time parsing of bracket tags like `[calm] ...` into audio instructions.
- **Voice Cloning & Speaker Manager**:
  - Select from pre-cached registered voices in `ref/` & `voice_cache/`.
  - Drag-and-drop / Upload any reference audio clip (`.wav`, `.mp3`) for zero-shot voice cloning.
- **Engine Selector**: Switch seamlessly between **SiangTTS (VoxCPM2 Thai LoRA)**, **ElevenLabs**, and **Gemini**.
- **Live Visual Tag & Instruction Preview**: Dynamic highlighting of tags and emotional instructions.
- **Built-in Audio Player Studio**: 1-click **"🎙️ สร้างเสียงพูด (Synthesize Audio)"** with Play/Pause, Timeline, and WAV download.

---

## API Endpoints

### `POST /synthesize`
Synthesizes speech with registered voice or base model:
```json
{
  "text": "[calm] หายใจเข้าลึกๆ ผ่อนคลาย แล้วค่อยๆ ปล่อยวางทุกอย่างลงนะ",
  "speaker_id": "speaker_1",
  "engine": "voxcpm",
  "cfg_value": 2.5,
  "inference_timesteps": 10,
  "auto_annotate": true
}
```
*Returns: `audio/wav` binary stream (48kHz)*

### `POST /synthesize/upload`
Synthesizes speech with a direct one-off uploaded reference audio file (Multipart Form).

### `GET /speakers`
Lists all registered voice profiles and their prompt cache status.

### `POST /speakers`
Uploads a reference audio clip (`file`) to register a new custom voice profile in `ref/` and caches prompt latents in `voice_cache/*.pt`.

### `DELETE /speakers/{speaker_id}`
Deletes a registered voice profile.

### `POST /annotate`
Analyzes raw Thai text into emotional clauses and intensities.

### `POST /render`
Renders annotated segments into engine-specific formats (`voxcpm`, `elevenlabs`, `gemini`).

### `GET /health`
Health check endpoint returning system status and registered speaker count.
