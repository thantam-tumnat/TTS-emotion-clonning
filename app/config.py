import os
from typing import Optional, Literal
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Provider selection: "gemini" or "anthropic"
    llm_provider: Literal["gemini", "anthropic"] = "gemini"

    # Gemini (Google AI Studio)
    gemini_api_key: str = ""
    google_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash"
    gemini_escalate_model: str = "gemini-2.5-pro"

    # Anthropic Claude
    anthropic_api_key: str = ""
    llm_model: str = "claude-haiku-4-5"
    llm_escalate_model: str = "claude-sonnet-5"

    # Pipeline & Segmenter
    max_segments: int = 20
    reanchor_chars: Optional[int] = None
    segmenter_engine: str = "crfcut"

    # SiangTTS / VoxCPM2 Voice Cloning
    siangtts_base_model: str = "openbmb/VoxCPM2"
    siangtts_adapter: str = "dubbing-ai/SiangTTS-VoxCPM2-Thai-LoRA"
    siangtts_device: str = ""
    siangtts_ref_dir: str = "ref"
    siangtts_cache_dir: str = "voice_cache"

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


settings = Settings()
