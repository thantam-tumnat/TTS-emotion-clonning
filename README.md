# Thai TTS Tone Annotation Layer (Tier 1)

FastAPI service for analyzing emotional tones and intensities in Thai text and rendering audio tag annotations for TTS engines (ElevenLabs, Gemini) without altering the user's text.

---

## Architecture Overview

```
Thai Raw Text
   │
   ├─ 1. SEGMENT   pythainlp sent_tokenize (crfcut) + whitespace preservation
   │              Strict invariant: "".join(clauses) == original_text
   │
   ├─ 2. LABEL     Claude tool-use structured JSON response
   │              LLM returns [{"i": 0, "tone": "sad", "intensity": 2}, ...]
   │              (LLM cannot return or modify text)
   │
   ├─ 3. MERGE     Merges adjacent clauses with identical tone (max intensity)
   │
   ├─ 4. VALIDATE  Checks indices, enum values, intensity bounds, text invariant
   │
   └─ 5. RENDER    ElevenLabs audio tags ([sad] text) or Gemini prompt instructions
```

---

## Tone Enum & ElevenLabs / Gemini Mapping

| Tone | ElevenLabs Tag | Gemini Prompt (Thai) |
|---|---|---|
| `neutral` | *(no tag)* | น้ำเสียงปกติ เป็นกลาง |
| `sad` | `[sad]` | เศร้า สะเทือนใจ |
| `happy` | `[happily]` | ร่าเริง ยิ้มขณะพูด |
| `angry` | `[angry]` | โกรธ เสียงแข็ง |
| `excited` | `[excited]` | ตื่นเต้น กระตือรือร้น |
| `calm` | `[calm]` | สงบ นุ่มนวล พูดช้า |
| `nervous` | `[nervous]` | ประหม่า ลังเล |
| `sarcastic` | `[sarcastic]` | ประชด แดกดัน |

Intensity levels:
- `1`: `[slightly {tag}] `
- `2`: `[{tag}] `
- `3`: `[very {tag}] `

---

## Setup & Running

### 1. Install Dependencies
```bash
py -m pip install -r requirements.txt
```

### 2. Configure Environment
Copy `.env.example` to `.env` and configure:
```env
ANTHROPIC_API_KEY=your_api_key
LLM_MODEL=claude-haiku-4-5
LLM_ESCALATE_MODEL=claude-sonnet-5
MAX_SEGMENTS=20
SEGMENTER_ENGINE=crfcut
```

### 3. Run FastAPI Server & Web UI
```bash
py -m uvicorn app.main:app --reload --port 8000
```
Then open your browser and navigate to:
👉 **`http://localhost:8000/`** to access the interactive **Thai TTS Tone Annotation Studio**.

### 4. Run Test Suite
```bash
py -m pytest -v
```

---

## Web UI Test Studio Features

- **Interactive Thai Input**: Real-time character count and preset quick-testing buttons (เศร้าปนโกรธ, ข่าวสาร, ประชด, ดีใจ ฯลฯ).
- **Engine Selector**: Switch between ElevenLabs tags and Gemini prompt rendering on the fly.
- **Visual Segments Breakdown**: Interactive cards showing each clause, emotion badge, intensity dots (`●●○`), and border accents.
- **TTS Payload Preview**: Formatted ElevenLabs tag highlights and Gemini instruction prompts with 1-click clipboard copying.
- **Raw JSON Inspector**: View the underlying API payload, model used, and fallback status.


---

## API Endpoints

### `POST /annotate`
Request:
```json
{
  "text": "ขอโทษนะ ฉันไม่ได้ตั้งใจ แต่เธอก็ไม่ฟังฉันเลย"
}
```
Response:
```json
{
  "original": "ขอโทษนะ ฉันไม่ได้ตั้งใจ แต่เธอก็ไม่ฟังฉันเลย",
  "segments": [
    { "text": "ขอโทษนะ ฉันไม่ได้ตั้งใจ ", "tone": "sad", "intensity": 2 },
    { "text": "แต่เธอก็ไม่ฟังฉันเลย", "tone": "angry", "intensity": 2 }
  ],
  "model_used": "claude-haiku-4-5",
  "fallback": false
}
```

### `POST /render`
Request:
```json
{
  "segments": [
    { "text": "ขอโทษนะ ฉันไม่ได้ตั้งใจ ", "tone": "sad", "intensity": 2 },
    { "text": "แต่เธอก็ไม่ฟังฉันเลย", "tone": "angry", "intensity": 2 }
  ],
  "engine": "elevenlabs"
}
```
Response:
```json
{
  "text": "[sad] ขอโทษนะ ฉันไม่ได้ตั้งใจ [angry] แต่เธอก็ไม่ฟังฉันเลย",
  "prompt": null
}
```

### `POST /speak`
End-to-end pipeline returning ready-to-synthesize payload.

### `GET /health`
Health check endpoint.
