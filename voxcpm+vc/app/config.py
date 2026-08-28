import os
from typing import Optional, Literal
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Provider selection: "gemini", "anthropic", or "openai"
    llm_provider: Literal["gemini", "anthropic", "openai"] = "gemini"

    # Gemini (Google AI Studio)
    gemini_api_key: str = ""
    google_api_key: str = ""
    gemini_model: str = "gemini-3.6-flash"
    # Escalation deliberately hops to a different model family: the usual primary
    # failure here is a transient 503 overload, which a sibling model shares.
    gemini_escalate_model: str = "gemini-3.5-flash-lite"

    # Anthropic Claude
    anthropic_api_key: str = ""
    llm_model: str = "claude-haiku-4-5"
    llm_escalate_model: str = "claude-sonnet-5"

    # 9arm Gateway / OpenAI-Compatible
    openai_api_key: str = ""
    openai_base_url: str = "https://gateway.9arm.co/v1"
    openai_model: str = "qwen3.8-27b-fp8"
    openai_escalate_model: str = "deepseek-v4-flash-0731"

    # Custom pronunciation overrides, applied to the text just before synthesis.
    # See app/services/pronunciation.py -- matching is token-level, so "ไฟล์" can be
    # respelled without touching "โปรไฟล์", which is a genuinely different vowel.
    pronunciation_path: str = "pronunciation.json"

    # Pipeline & Segmenter
    max_segments: int = 20
    reanchor_chars: Optional[int] = None
    segmenter_engine: str = "crfcut"

    # Shared SiangTTS GPU service — the one process that holds VoxCPM2 for every
    # pipeline on the host (voice-cloning/src/gpu_service.py). The studio sends
    # generation there and keeps annotation, chunking and assembly local.
    voxcpm_service_url: str = "http://127.0.0.1:8020"
    # Refuse to run without it. Loading a second copy of the model here would fight
    # the shared one for VRAM on a single-GPU host, which is the exact problem the
    # split exists to solve -- and it would do it silently, several minutes into a
    # request. Set false only when nothing else is using the GPU.
    voxcpm_remote_required: bool = True
    service_port: int = 8013

    # ------------------------------------------------------------------ #
    # n8n LiveAI webhook — same async contract as the :8010 service, but hosted
    # in this process so a script posted from n8n is synthesized through this
    # studio's emotion pipeline (donor -> VoxCPM2 -> SeedVC) instead of the plain
    # LoRA path. Endpoints live under /webhook/* on this same port (8013).
    #
    # The upload + callback settings share the :8010 env var names on purpose, so
    # one .env line points both services at the same delivery endpoint.
    # ------------------------------------------------------------------ #
    siangtts_upload_url: str = "https://looklike.ai/api/v1/live-gpt/upload"
    siangtts_upload_token: str = ""
    siangtts_default_callback: str = (
        "https://test.looklike.ai/api/v1/live-gpt/n8n/audio-callback"
    )
    # Emotion annotation via LLM. When on, a script with no hand-written style tags is
    # sent to the LLM (Gemini/OpenAI) to label each clause's emotion, and the accept
    # endpoint blocks on that round-trip before it replies. Turn OFF (WEBHOOK_USE_LLM=false)
    # to skip the LLM entirely: hand-written [tags] are still honoured, but un-tagged
    # text is synthesized neutral. Kills the "accept hangs on a slow/overloaded LLM"
    # stall at the cost of auto-emotion. See app/webhook.py:_plan_chunks.
    webhook_use_llm: bool = True
    # Job scratch dir (finished takes are uploaded, then deleted unless keep is on).
    webhook_work_dir: str = "webhook_work"
    webhook_keep_work: bool = False
    # Finished jobs kept in memory for /webhook/jobs. Bounds a long-lived process.
    webhook_max_history: int = 500
    # Target voice (SeedVC) when a request names none. Empty -> the shared auto
    # seed voice, matching an unpinned /synthesize call.
    webhook_default_voice: str = ""
    http_timeout: float = 120.0

    # ------------------------------------------------------------------ #
    # Donor -> VoxCPM2 -> SeedVC pipeline
    #
    # Emotion no longer comes from a style parenthetical the model reads. It comes
    # from a donor clip: one recording per emotion from ONE actor. VoxCPM2 is given
    # that clip *with its transcript*, which selects its continuation ("ultimate
    # cloning") mode -- the mode that reproduces the prompt clip's own delivery and
    # ignores control instructions. That is exactly backwards for the :8011 studio
    # and exactly right here: the donor's anger becomes the script's anger.
    #
    # What that costs is the speaker: the output is the donor's voice, not the
    # user's. SeedVC then swaps the timbre onto the user's reference clip, with
    # f0_condition so the emotional pitch contour survives the swap.
    # ------------------------------------------------------------------ #

    # Donor sets live at <emotion_donor_dir>/<set_id>/<emotion>_1.wav + .txt.
    # A *set* is one actor across all five emotions, which is what keeps the
    # emotion the only thing changing between chunks.
    emotion_donor_dir: str = "ref/emotions"
    # Set used when a request names none. Empty picks the first set matching the
    # requested gender from the manifest.
    default_donor_set: str = ""
    default_gender: str = "female"

    # SeedVC worker (its own venv/process -- torch 2.4 conflicts with this env).
    seedvc_url: str = os.getenv("SEEDVC_URL", "http://127.0.0.1:8022")
    # Generous: a convert can queue behind VoxCPM2 for the same GPU.
    seedvc_timeout: int = int(os.getenv("SEEDVC_TIMEOUT", "180"))
    # Required to keep the donor's pitch emotion. Timbre-only conversion flattens
    # it -- measured on the sibling pipeline as angry falling from +99 Hz median-F0
    # over neutral to -40 Hz. With f0_condition it comes back at +97 Hz.
    seedvc_f0_condition: bool = True
    # Keeps the emotional contour but re-centres it in the target's own register,
    # so a female target does not end up speaking in a male donor's range.
    seedvc_auto_f0_adjust: bool = True
    # Which of the test page's F0-compare treatments the shipped take uses (the
    # baseline / A / B buttons on /static/test.html), so a mode judged best by ear
    # there can be turned on for real synthesis without touching code:
    #   baseline -> auto_f0_adjust re-centres every emotion's pitch to the target
    #               (flat register, the donor's pitch-emotion is largely lost)
    #   A        -> keep VoxCPM2's absolute pitch (emotion survives) but in the
    #               donor's register, not the target's
    #   B        -> keep the pitch contour, shifted by a constant so the donor's
    #               NEUTRAL register lands on the target's -> emotion AND register
    # Only "B" needs the extra per-take shift computed in voxcpm_vc_service; the
    # other two are pure SeedVC auto_f0_adjust flips. See render_chunks / render_f0_compare.
    seedvc_f0_mode: str = "baseline"
    # Too few steps smear consonants into slurred articulation; 35 is the measured
    # floor for clean Thai plosives at a modest speed cost.
    seedvc_diffusion_steps: int = int(os.getenv("SEEDVC_DIFFUSION_STEPS", "35"))
    seedvc_inference_cfg_rate: float = 0.7
    # Fail the request when the worker is down rather than returning audio in the
    # donor's voice, which is a wrong-speaker take that sounds like a success.
    seedvc_required: bool = True
    # VoxCPM2 can overshoot [-1, 1]; WAVs are written as PCM_16 with no peak guard,
    # so a loud emotion clips on the way into SeedVC. Scale anything above this
    # ceiling down to it first. 0 disables.
    voxcpm_peak: float = float(os.getenv("VOXCPM_PEAK", "0.95"))

    # VoxCPM2 continuation mode clones the donor's timbre/prosody but NOT its speaking
    # rate: it renders new text at its own, more neutral pace, so an emotional donor
    # (a slow, drawn-out sad clip especially) comes back faster than the recording it
    # was cloned from. SeedVC is length-preserving and cannot fix it downstream. When
    # this is on, each generated piece is WSOLA-stretched (pitch untouched) to the
    # donor clip's own measured pace -- seconds of voiced audio per spoken character --
    # before it enters SeedVC, so the take carries the donor's timing, not the model's.
    voxcpm_vc_match_donor_pace: bool = os.getenv("VOXCPM_VC_MATCH_DONOR_PACE", "1") not in ("0", "false", "False", "")
    # Clamp on that stretch, as a fraction. 0.35 lets a piece run from 0.65x to 1.35x
    # of its rendered length; wider risks WSOLA artefacts and a robotic drag, tighter
    # leaves a fast sad read still faster than the donor.
    voxcpm_vc_max_pace_stretch: float = float(os.getenv("VOXCPM_VC_MAX_PACE_STRETCH", "0.35"))
    # Neutral has no emotion-specific pitch contour to steal from a donor, so the
    # donor+SeedVC round trip buys nothing there and only adds a GPU stage and its
    # conversion artefacts. When on, neutral chunks clone the target ref directly
    # (VoxCPM2 zero-shot, no donor, no SeedVC) instead of going through the normal
    # donor -> VoxCPM2 -> SeedVC path. Other emotions are untouched.
    voxcpm_vc_skip_neutral: bool = os.getenv("VOXCPM_VC_SKIP_NEUTRAL", "1") not in ("0", "false", "False", "")

    # SiangTTS / VoxCPM2 Voice Cloning
    siangtts_base_model: str = "openbmb/VoxCPM2"
    siangtts_adapter: str = "dubbing-ai/SiangTTS-VoxCPM2-Thai-LoRA"
    siangtts_device: str = ""
    siangtts_ref_dir: str = "ref"
    siangtts_cache_dir: str = "voice_cache"

    # The denoiser (ZipEnhancer) is only used when generate(denoise=True), which we
    # never do. Loading it costs memory for nothing, so it is off by default.
    siangtts_load_denoiser: bool = False
    # torch.compile warm-up. Faster steady-state, but slow to start and fragile on
    # Windows; enable once the model is confirmed loading.
    siangtts_optimize: bool = False
    # How hard the Thai LoRA is applied, per side of the model. The adapter ships
    # r=64 alpha=128, i.e. strength 2.0 on both the LM (which reads the style
    # parenthetical and the Thai text) and the DiT (which generates the acoustics).
    #
    # Measured with tools/expr_sweep.py --stage lora over 5 emotions x 3 paired
    # reps: at the shipped strength "[angry]" comes out 3.1 dB *quieter* and 3.6
    # semitones *lower* than the same sentence read neutrally -- the opposite of
    # anger on every axis. The DiT side is what does it. Keeping the LM side at
    # full strength and taking the DiT side to zero turns that into +1.2 dB, +1.9
    # st of pitch range and 18% faster, and roughly triples the angry-vs-sad
    # contrast, while the LM side still carries the Thai the adapter was trained
    # for. Nothing else measured helped: cfg 4-6, level-3 wording, Thai-language
    # directions and dropping the seed voice were all neutral or worse.
    #
    # Set siangtts_lora_dit_scale=2.0 to restore the adapter's shipped behaviour.
    siangtts_lora_lm_scale: float = 2.0
    siangtts_lora_dit_scale: float = 0.0
    # When the real model fails to load, fall back to the sine-tone mock instead of
    # raising. Only ever useful for tests -- a silent fallback in production sounds
    # exactly like a broken model.
    siangtts_allow_mock: bool = False
    # With no speaker pinned, VoxCPM2 resamples the timbre on every call, so a
    # multi-chunk utterance changes speaker mid-sentence. Generate a short neutral
    # seed line, clone its timbre, and condition every chunk on that -- including the
    # first, so no chunk inherits another chunk's emotion.
    siangtts_auto_voice_consistency: bool = True
    # VoxCPM2's "ultimate cloning" (prompt audio + its transcript) buys timbre
    # fidelity by reproducing the prompt clip's own rhythm and emotion -- and the
    # docs are explicit that it *ignores the control instruction* while doing so.
    # Every style tag in this studio is a control instruction, so pairing the two
    # silently discards the emotion. Off by default: a sidecar ref/<id>.txt is
    # treated as documentation unless this is deliberately turned on.
    siangtts_hifi_cloning: bool = False
    # Short, emotionally flat, gender-neutral Thai used only to mint that seed voice.
    # Never spoken in the output.
    siangtts_voice_seed_text: str = "วันนี้อากาศปกติ อุณหภูมิยี่สิบห้าองศา"
    # Longest run of spoken characters handed to VoxCPM2 in one generation. It has
    # no internal splitter, and past roughly this length the speaker identity drifts
    # mid-utterance. Anything longer is broken at the best available seam and the
    # pieces are conditioned on one shared voice. Unset or 0 falls back to the
    # module default rather than disabling the split -- there is no safe "off".
    siangtts_max_chunk_chars: int = 140

    @field_validator("reanchor_chars", mode="before")
    @classmethod
    def parse_reanchor_chars(cls, v):
        if v is None or v == "" or (isinstance(v, str) and not v.strip()):
            return None
        try:
            return int(v)
        except (ValueError, TypeError):
            return None

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    @property
    def effective_gemini_api_key(self) -> str:
        return self.gemini_api_key or self.google_api_key or os.getenv("GEMINI_API_KEY", "") or os.getenv("GOOGLE_API_KEY", "")

    @property
    def effective_openai_api_key(self) -> str:
        return self.openai_api_key or os.getenv("OPENAI_API_KEY", "")


settings = Settings()
