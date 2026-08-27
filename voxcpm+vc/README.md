# VoxCPM2 + SeedVC Studio (:8013)

A second studio beside the one on :8011. Same annotation, same segmentation, same
assembly, same shared GPU service — the only thing that changes is **where the
emotion comes from**.

```
:8011  ref voice  +  script + (emotion tag)  ─────────────► VoxCPM2 ──► take
                     the model reads the tag

:8013  donor clip (5 emotions, one actor)                                 
         │                                                                 
         └─ script ──► VoxCPM2 (continuation mode) ──► SeedVC ──► take     
                       clones the donor's delivery      swaps timbre       
                       (donor's voice, right emotion)   to the ref voice   
```

On :8011 the emotion is a word the model interprets, and the same word does not land
the same way twice. Here it is a recording. VoxCPM2 is given the donor clip **with
its transcript**, which selects its continuation ("ultimate cloning") mode — the mode
that reproduces the prompt clip's own delivery and ignores control instructions. That
gets the emotion right and the speaker wrong, so SeedVC then converts the timbre onto
the reference voice, with `f0_condition` so the emotional pitch contour survives.

VoxCPM2's own emotion feature is unused: the leading style parenthetical is stripped
before generation, because in continuation mode it would be read aloud.

## n8n LiveAI webhook (same port)

The async webhook contract from the production `:8010` service is also hosted here, on
this same port, so a script posted from n8n is synthesized through the emotion pipeline
above instead of the plain LoRA path. It answers `{"status":"success"}` immediately,
runs the job on a single FIFO worker, then POSTs the uploaded audio URL to `callback_url`.

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/webhook/live-ai-create-new` | accept a script, enqueue (bare alias: `/live-ai-create-new`) |
| GET  | `/webhook` | monitoring dashboard |
| GET  | `/webhook/jobs`, `/webhook/jobs/{id}` | job state as JSON |
| GET  | `/webhook/health` | queue + delivery status |
| GET  | `/webhook/voices` | target (SeedVC) voices |
| GET  | `/webhook/audio/{queue_id}` | locally-rendered take |

Request body — the `:8010` fields are all accepted (unknown ones ignored): `prompt`,
`job_id`, `queue_id`, `voice_id` (→ SeedVC target; blank = auto seed voice),
`callback_url` (blank = default). `voice_text`, `ref_text`, `audio_speed` and
`country_code` are accepted for compatibility but not used by this pipeline. Emotion is
auto-annotated per chunk from the text — no emotion field is needed.

Two optional extensions choose the **donor** whose emotion is cloned:

| Field | Meaning |
|-------|---------|
| `sex` | `"male"` / `"female"` — which donor gender to clone emotion from. Blank → `default_gender` (female). |
| `donor_set` | pin one specific actor set (e.g. `female_002`). Blank → **a random complete set of `sex` is drawn per job**, so takes vary. |

The random pick is made when the job is enqueued and shown on the dashboard, so it is
stable across a job's retries. Delivery reuses the `:8010` env vars, so one `.env`
points both services at the same upload endpoint:

```
SIANGTTS_UPLOAD_URL=...        # upload endpoint (returns {"file_url": ...})
SIANGTTS_UPLOAD_TOKEN=...      # bearer for the upload
SIANGTTS_DEFAULT_CALLBACK=...  # used when a request omits callback_url
```

Output is 44.1 kHz WAV (SeedVC's native rate). `audio_speed` from the n8n body is
currently ignored — the pipeline has no tempo stage.

## Ports

| Port | Service | Shared with |
|------|---------|-------------|
| 8013 | this studio | — |
| 8022 | SeedVC worker | — |
| 8020 | Go queue gateway | :8010, :8011 |
| 8021 | Python GPU service (VoxCPM2 ×1) | :8010, :8011 |

Generation goes through the same queue as the other pipelines, so this adds no second
copy of VoxCPM2 to the box. SeedVC is the one new resident model.

## Running it

```bash
start_voxcpm_vc_studio.bat     # this studio  (:8013)
start_seedvc_worker.bat        # SeedVC worker (:8022)
```

or `start_all_services.bat` for everything. The studio starts without the worker —
`/health` reports `pipeline.seedvc.reachable: false` — but a synthesis request returns
**503** rather than a take, because a take that skipped conversion would come back in
the donor actor's voice, which sounds like a success.

### SeedVC setup (once)

SeedVC pins torch 2.4, which the studio env cannot hold, so it runs from its own
virtualenv:

```bash
git clone https://github.com/Plachtaa/seed-vc.git
python -m venv seedvc-venv
seedvc-venv/Scripts/pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121
seedvc-venv/Scripts/pip install accelerate scipy==1.13.1 librosa==0.10.2 "huggingface-hub>=0.28.1" munch==4.0.0 einops==0.8.0 descript-audio-codec==1.0.0 pydub==0.25.1 resemblyzer jiwer==3.0.3 transformers==4.46.3 soundfile==0.12.1 modelscope==1.18.1 funasr==1.1.5 numpy==1.26.4 hydra-core==1.3.2 pyyaml python-dotenv fastapi uvicorn "pydantic>=2"
```

Then point `start_seedvc_worker.bat` at both with `SEEDVC_REPO` and `SEEDVC_PYTHON`.
Weights (~2 GB) download on first run; set `HF_HOME` to an existing cache to reuse a
copy you already have.

## Donor sets

`ref/emotions/<set_id>/<emotion>_1.wav` + `.txt`, one folder per actor, five emotions
each: **neutral, angry, happy, sad, frustrated**. Ten sets ship (five female, five
male), built from `airesearch/thai-ser` and listed in `donors_manifest.json` with the
rater agreement behind each clip.

One set per take is the rule. Mixing actors between chunks changes the speaker
mid-script, and SeedVC converts *timbre* — the second actor's pacing and accent still
come through.

Rebuild or extend them with `tools/build_donor_sets.py`.

### Ten tones, five recordings

The annotator's vocabulary is larger than the donor library, so the rest is mapped to
the nearest donor by arousal and valence:

| annotator tone | donor |
|---|---|
| neutral, calm | neutral |
| sad, tired | sad |
| happy, excited | happy |
| angry | angry |
| frustrated, nervous, scared, sarcastic | frustrated |

The map is served at `GET /api/donors` and shown under the donor picker, because a
silently substituted emotion is otherwise indistinguishable from one the model got
right. A tone outside the map is a 422, not a quiet fall back to neutral.

## API

Same surface as the :8011 studio, plus:

| Endpoint | What it does |
|---|---|
| `GET /api/donors` | donor sets on disk, their emotions and transcripts, and the tone map |
| `GET /api/donors/{set}/{emotion}/audio` | stream one donor clip, to audition the emotion |

`/synthesize`, `/synthesize/upload` and `/synthesize/ab` take two extra fields:

- `donor_set` — which actor supplies the emotion (omit to pick automatically)
- `gender` — preferred donor gender when `donor_set` is omitted

Output is 44.1 kHz (SeedVC's rate), not the 48 kHz the :8011 studio returns.

### Target voice

SeedVC converts *into* someone, so unlike :8011 there is no meaningful "unpinned"
take. Precedence: an uploaded clip, then `speaker_id`, then the first clip in `ref/`
as a house voice. `X-Voice-Anchor` on the response is therefore always `speaker` or
`reference`, never `seed`/`none`.

## Tests

```bash
py -m pytest tests/test_voxcpm_vc.py
```

The suite stubs the SeedVC worker (a pass-through converter) and the GPU service, so
it checks routing, batching and error handling rather than audio quality — the two
things it does assert about the pipeline are that the donor is cloned *with its
transcript*, and that a missing worker fails loudly.
