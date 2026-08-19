import json
import logging
from typing import List, Tuple, Any, Optional
from app.config import settings
from app.models import Segment, Tone, AnnotateResponse, LLMAnnotationResult, LLMTagConversionResult
from app.prompts import (
    SYSTEM_PROMPT,
    FEW_SHOT_EXAMPLES,
    ANNOTATE_TOOL,
    TAG_CONVERSION_SYSTEM_PROMPT,
    CONVERT_TAG_TOOL,
)
from app.validator import validate_and_build_segments, ValidationError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Anthropic Helpers
# ---------------------------------------------------------------------------

def build_anthropic_system_blocks() -> list:
    """Build Anthropic system message with prompt caching enabled."""
    return [
        {
            "type": "text",
            "text": SYSTEM_PROMPT,
            "cache_control": {"type": "ephemeral"}
        }
    ]


def build_anthropic_messages(clauses: List[str], guidance: Optional[str] = None) -> list:
    """Construct Anthropic conversation messages with few-shot examples."""
    messages = []
    for eg in FEW_SHOT_EXAMPLES:
        messages.append({
            "role": "user",
            "content": json.dumps(eg["input"], ensure_ascii=False)
        })
        messages.append({
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": f"call_{eg['output']['labels'][0]['i']}",
                    "name": "annotate_clauses",
                    "input": eg["output"]
                }
            ]
        })

    user_payload: dict[str, Any] = {
        "clauses": [{"i": idx, "text": clause} for idx, clause in enumerate(clauses)]
    }
    if guidance and guidance.strip():
        user_payload["user_tone_guidance"] = guidance.strip()

    messages.append({
        "role": "user",
        "content": json.dumps(user_payload, ensure_ascii=False)
    })
    return messages


# ---------------------------------------------------------------------------
# Gemini (Google AI Studio) Helpers
# ---------------------------------------------------------------------------

def build_gemini_prompt(clauses: List[str], guidance: Optional[str] = None) -> str:
    """Build prompt for Gemini containing instructions, few-shot examples, and target clauses."""
    few_shot_text = ""
    for idx, eg in enumerate(FEW_SHOT_EXAMPLES, 1):
        few_shot_text += f"\n--- Example {idx} ---\nInput:\n{json.dumps(eg['input'], ensure_ascii=False)}\nOutput:\n{json.dumps(eg['output'], ensure_ascii=False)}\n"

    target_payload: dict[str, Any] = {
        "clauses": [{"i": idx, "text": clause} for idx, clause in enumerate(clauses)]
    }
    if guidance and guidance.strip():
        target_payload["user_tone_guidance"] = guidance.strip()

    return f"{few_shot_text}\n--- Target Task ---\nInput:\n{json.dumps(target_payload, ensure_ascii=False)}\nOutput:"


# ---------------------------------------------------------------------------
# Main Annotator Engine
# ---------------------------------------------------------------------------

class Annotator:
    def __init__(self, anthropic_client: Optional[Any] = None, gemini_client: Optional[Any] = None):
        self._anthropic_client = anthropic_client
        self._gemini_client = gemini_client

    def get_anthropic_client(self):
        if self._anthropic_client is not None:
            return self._anthropic_client
        import anthropic
        return anthropic.Anthropic(api_key=settings.anthropic_api_key or "dummy-key")

    def get_gemini_client(self):
        if self._gemini_client is not None:
            return self._gemini_client
        from google import genai
        return genai.Client(api_key=settings.effective_gemini_api_key or "dummy-key")

    def _call_anthropic(self, client: Any, model: str, clauses: List[str], guidance: Optional[str] = None) -> List[Any]:
        """Execute structured tool use call via Anthropic."""
        system_blocks = build_anthropic_system_blocks()
        messages = build_anthropic_messages(clauses, guidance=guidance)

        response = client.messages.create(
            model=model,
            max_tokens=2048,
            temperature=0,
            system=system_blocks,
            messages=messages,
            tools=[ANNOTATE_TOOL],
            tool_choice={"type": "tool", "name": "annotate_clauses"}
        )

        for block in response.content:
            if getattr(block, "type", None) == "tool_use" and getattr(block, "name", None) == "annotate_clauses":
                input_data = getattr(block, "input", {})
                if isinstance(input_data, dict) and "labels" in input_data:
                    return input_data["labels"]
                elif isinstance(input_data, list):
                    return input_data

        raise ValidationError("Anthropic LLM did not invoke annotate_clauses tool properly")

    def _call_gemini(self, client: Any, model: str, clauses: List[str], guidance: Optional[str] = None) -> List[Any]:
        """Execute Structured Output JSON call via Google AI Studio (Gemini)."""
        from google.genai import types

        prompt = build_gemini_prompt(clauses, guidance=guidance)
        config = types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            response_mime_type="application/json",
            response_schema=LLMAnnotationResult,
            temperature=0.0,
        )

        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=config
        )

        if not response.text:
            raise ValidationError("Gemini returned empty response")

        try:
            parsed = LLMAnnotationResult.model_validate_json(response.text)
            return [label.model_dump() for label in parsed.labels]
        except Exception as e:
            # Fallback to json dict parsing
            try:
                data = json.loads(response.text)
                if isinstance(data, dict) and "labels" in data:
                    return data["labels"]
                elif isinstance(data, list):
                    return data
            except Exception:
                pass
            raise ValidationError(f"Failed to parse Gemini output: {e}")

    def _run_provider(self, provider: str, model: str, clauses: List[str], guidance: Optional[str] = None) -> List[Any]:
        if provider == "gemini":
            client = self.get_gemini_client()
            return self._call_gemini(client, model, clauses, guidance=guidance)
        else:
            client = self.get_anthropic_client()
            return self._call_anthropic(client, model, clauses, guidance=guidance)

    def annotate(
        self,
        original_text: str,
        clauses: List[str],
        guidance: Optional[str] = None,
        custom_model: Optional[str] = None
    ) -> AnnotateResponse:
        """
        Annotate clauses with emotional tones.
        Executes primary model -> escalation model -> fallback neutral.
        """
        if not clauses:
            return AnnotateResponse(
                original=original_text,
                segments=[],
                model_used="none",
                fallback=False
            )

        provider = settings.llm_provider.lower()
        if provider == "gemini":
            primary_model = (custom_model.strip() if custom_model and custom_model.strip() else settings.gemini_model)
            escalate_model = settings.gemini_escalate_model
        else:
            primary_model = (custom_model.strip() if custom_model and custom_model.strip() else settings.llm_model)
            escalate_model = settings.llm_escalate_model

        attempts: List[dict] = []

        # Attempt 1: Primary Model
        try:
            raw_labels = self._run_provider(provider, primary_model, clauses, guidance=guidance)
            segments = validate_and_build_segments(
                original_text=original_text,
                clauses=clauses,
                raw_labels=raw_labels,
                max_segments=settings.max_segments
            )
            attempts.append({"model": primary_model, "provider": provider, "status": "success"})
            return AnnotateResponse(
                original=original_text,
                segments=segments,
                model_used=primary_model,
                fallback=False,
                attempts=attempts
            )
        except Exception as err:
            err_msg = str(err)
            logger.warning(f"Primary model {primary_model} ({provider}) failed: {err_msg}. Escalating to {escalate_model}")
            attempts.append({"model": primary_model, "provider": provider, "status": "failed", "error": err_msg})

        # Attempt 2: Escalation Model (only if different from primary)
        if escalate_model and escalate_model != primary_model:
            try:
                raw_labels = self._run_provider(provider, escalate_model, clauses, guidance=guidance)
                segments = validate_and_build_segments(
                    original_text=original_text,
                    clauses=clauses,
                    raw_labels=raw_labels,
                    max_segments=settings.max_segments
                )
                attempts.append({"model": escalate_model, "provider": provider, "status": "success"})
                return AnnotateResponse(
                    original=original_text,
                    segments=segments,
                    model_used=escalate_model,
                    fallback=False,
                    attempts=attempts
                )
            except Exception as err:
                err_msg = str(err)
                logger.error(f"Escalation model {escalate_model} ({provider}) failed: {err_msg}. Falling back to neutral.")
                attempts.append({"model": escalate_model, "provider": provider, "status": "failed", "error": err_msg})

        # Attempt 3: Safe Fallback
        fallback_segments = [
            Segment(
                text=original_text,
                tone=Tone.NEUTRAL,
                intensity=2
            )
        ]
        error_summary = "; ".join([f"[{a['model']}]: {a.get('error', 'unknown error')}" for a in attempts if a.get('status') == 'failed'])
        return AnnotateResponse(
            original=original_text,
            segments=fallback_segments,
            model_used="fallback-neutral",
            fallback=True,
            error="LLM annotation failed (e.g. Quota Exceeded or API error)",
            error_detail=error_summary,
            attempts=attempts
        )

    def convert_style_tag(
        self,
        tag: str,
        intensity: int = 2,
        custom_model: Optional[str] = None
    ) -> Optional[dict]:
        """Convert a free-form emotion tag (e.g. 'sad and cry') to a VoxCPM-compatible instruction."""
        if not tag or not tag.strip():
            return None

        clean_tag = tag.strip()
        provider = settings.llm_provider.lower()
        if provider == "gemini":
            primary_model = (custom_model.strip() if custom_model and custom_model.strip() else settings.gemini_model)
            escalate_model = settings.gemini_escalate_model
        else:
            primary_model = (custom_model.strip() if custom_model and custom_model.strip() else settings.llm_model)
            escalate_model = settings.llm_escalate_model

        for model in filter(None, [primary_model, escalate_model]):
            try:
                if provider == "gemini":
                    from google.genai import types
                    client = self.get_gemini_client()
                    config = types.GenerateContentConfig(
                        system_instruction=TAG_CONVERSION_SYSTEM_PROMPT,
                        response_mime_type="application/json",
                        response_schema=LLMTagConversionResult,
                        temperature=0.0,
                    )
                    prompt = f"Convert this emotion tag to a VoxCPM2 style instruction: '{clean_tag}' (default intensity: {intensity})"
                    response = client.models.generate_content(
                        model=model,
                        contents=prompt,
                        config=config
                    )
                    if response.text:
                        parsed = LLMTagConversionResult.model_validate_json(response.text)
                        instr = parsed.instruction.strip()
                        if not instr.startswith("("):
                            instr = f"({instr}"
                        if not instr.endswith(")"):
                            instr = f"{instr})"
                        return {
                            "instruction": instr,
                            "tone": parsed.tone,
                            "intensity": parsed.intensity,
                        }
                else:
                    client = self.get_anthropic_client()
                    response = client.messages.create(
                        model=model,
                        max_tokens=512,
                        temperature=0,
                        system=TAG_CONVERSION_SYSTEM_PROMPT,
                        messages=[
                            {"role": "user", "content": f"Convert this emotion tag to a VoxCPM2 style instruction: '{clean_tag}' (default intensity: {intensity})"}
                        ],
                        tools=[CONVERT_TAG_TOOL],
                        tool_choice={"type": "tool", "name": "convert_style_tag"}
                    )
                    for block in response.content:
                        if getattr(block, "type", None) == "tool_use" and getattr(block, "name", None) == "convert_style_tag":
                            input_data = getattr(block, "input", {})
                            instr = str(input_data.get("instruction", "")).strip()
                            if instr:
                                if not instr.startswith("("):
                                    instr = f"({instr}"
                                if not instr.endswith(")"):
                                    instr = f"{instr})"
                                tone_str = str(input_data.get("tone", "neutral")).lower()
                                tone_enum = Tone(tone_str) if tone_str in Tone._value2member_map_ else Tone.NEUTRAL
                                return {
                                    "instruction": instr,
                                    "tone": tone_enum,
                                    "intensity": int(input_data.get("intensity", intensity)),
                                }
            except Exception as e:
                logger.warning(f"[convert_style_tag] Failed with model {model} ({provider}): {e}")
                continue

        return None


annotator = Annotator()

