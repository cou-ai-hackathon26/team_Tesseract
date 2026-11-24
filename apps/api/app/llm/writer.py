"""The Bangla narrative writer.

Implementations behind the `Writer` protocol:

* `GeminiWriter`   - primary Google Gemini Flash writer (with model fallback) + hard validator.
* `LlmWriter`      - optional OpenAI-compatible writer + hard validator.
* `TemplateWriter` - composes from the same facts with no network at all.

The template path is not a stub. It is what runs automatically whenever no LLM
key is configured, the provider is unreachable/rate-limited, or the model
returns unusable text, so the product never depends on an external API.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
import unicodedata
from collections.abc import Callable
from typing import Any, Protocol

import httpx
from pydantic import BaseModel

from ..config import settings

log = logging.getLogger(__name__)

#: Bengali block plus whitespace, the danda, and the usual Indic punctuation.
_BANGLA = re.compile(r"[\u0980-\u09FF]")
_LATIN = re.compile(r"[A-Za-z]")
_MARKDOWN = re.compile(r"[*_`#>\[\]{}|]")
_EMOJI = re.compile(
    "[\U0001f300-\U0001faff\U00002600-\U000027bf\U0001f1e6-\U0001f1ff\ufe0f\u2b00-\u2bff]"
)
_WHITESPACE = re.compile(r"\s+")

MIN_WORDS = 25
MAX_WORDS = 120


class Writer(Protocol):
    async def write(self, facts: dict[str, Any]) -> tuple[str, str]: ...


def normalise(text: str) -> str:
    """NFC-normalise, collapse whitespace.

    Bengali text has more than one valid encoding for the same glyphs (`য়` as one
    code point or two). Normalising before comparing or caching keeps equality
    checks honest.
    """
    return _WHITESPACE.sub(" ", unicodedata.normalize("NFC", text)).strip()


class InvalidNarrative(ValueError):
    """Raised when model output fails a safety or quality check."""


def validate_narrative(text: str, allowed_chemicals: list[str]) -> str:
    """Reject anything unsafe or unusable. Returns the cleaned text.

    The Latin-script check is the most important one here: the TTS normaliser
    *silently drops* Latin spans, so a Latin chemical name would vanish from the
    audio while leaving a fluent, complete-sounding sentence behind.
    """
    cleaned = normalise(text)

    if not cleaned:
        raise InvalidNarrative("empty")
    if not _BANGLA.search(cleaned):
        raise InvalidNarrative("no_bangla_script")
    if _LATIN.search(cleaned):
        raise InvalidNarrative(f"latin_script:{_LATIN.search(cleaned).group(0)}")  # type: ignore[union-attr]
    if _MARKDOWN.search(cleaned):
        raise InvalidNarrative("markdown")
    if _EMOJI.search(cleaned):
        raise InvalidNarrative("emoji")

    words = cleaned.split()
    if not (MIN_WORDS <= len(words) <= MAX_WORDS):
        raise InvalidNarrative(f"word_count:{len(words)}")

    # Every chemical the model names must be one the knowledge base supplied for
    # *this* crop. Latin is already rejected above, so the realistic failure is a
    # plausible Bangla name for a product that was never in the facts - which
    # reads perfectly and would be wrong.
    for candidate in KNOWN_CHEMICALS_BN:
        if candidate in cleaned and not any(candidate in allowed for allowed in allowed_chemicals):
            raise InvalidNarrative(f"unknown_medicine:{candidate}")

    return cleaned


#: Bangla names of active ingredients common in Bangladesh, used only to *detect*
#: a chemical in the text. Presence in this list grants nothing; a detected name
#: still has to appear in the facts handed to the model for that crop. Grown from
#: the knowledge base plus products a model plausibly reaches for.
KNOWN_CHEMICALS_BN: tuple[str, ...] = (
    # -- in the knowledge base ------------------------------------------
    "ট্রাইসাইক্লাজল",
    "আইসোপ্রোথায়োলেন",
    "কপার হাইড্রোক্সাইড",
    "ম্যানকোজেব",
    "মেটালাক্সিল",
    "ক্লোরোথ্যালোনিল",
    "ইমিডাক্লোপ্রিড",
    "কার্বেন্ডাজিম",
    # -- plausible but not supplied for every crop ---------------------
    "প্রোপিকোনাজল",
    "হেক্সাকোনাজল",
    "টেবুকোনাজল",
    "অ্যাজোক্সিস্ট্রবিন",
    "ডাইফেনোকোনাজল",
    "কার্বারিল",
    "ক্লোরপাইরিফস",
    "সাইপারমেথ্রিন",
    "ডাইমেথোয়েট",
    "থায়ামেথোক্সাম",
    "ফিপ্রোনিল",
    "অ্যাবামেক্টিন",
    "বুপ্রোফেজিন",
    "পাইমেট্রোজিন",
    "অ্যাসিটামিপ্রিড",
    "থায়োফানেট মিথাইল",
    "কপার অক্সিক্লোরাইড",
    "সালফার",
    "জিংক থায়োফসফেট",
)

SYSTEM_PROMPT = """\
তুমি বাংলাদেশের একজন কৃষি পরামর্শক। নিচে দেওয়া তথ্য ও সরকারি (বারি/ব্রি) অনুমোদিত অংশগুলো থেকে
কৃষকের জন্য ছোট, সহজ, মুখে বলার মতো একটি বাংলা পরামর্শ লেখো।

নিয়ম:
- সবসময় চলিত বাংলায়, কেবল বাংলা অক্ষরে লিখবে। কোনো ইংরেজি অক্ষর, রোমান হরফ বা ইংরেজি শব্দ ব্যবহার করবে না।
- ২ থেকে ৪ বাক্য। ২৫ থেকে ৯০ শব্দের মধ্যে রাখবে।
- একজন কৃষক আরেকজন কৃষককে যেভাবে বলেন, সেভাবে সরল ভাষায় লিখবে। সম্মানসূচক 'আপনি' ব্যবহার করবে।
- কোনো মার্কডাউন, বুলেট, ইমোজি, লিংক বা তালিকা নয়। শুধু টানা বাক্য।
- ওষুধ, সক্রিয় উপাদান ও মাত্রা কেবল নিচে দেওয়া অনুমোদিত তথ্য ও পুনরুদ্ধার করা অংশ থেকে নেবে।
  নিজের থেকে কোনো নতুন ওষুধ, মাত্রা বা রোগের নাম বানাবে না।
- যদি অনুমোদিত ওষুধের তালিকা খালি থাকে বা রোগ নিশ্চিত না হয়, তবে কোনো ওষুধ বা ছত্রাকনাশকের নাম লিখবে না—
  কৃষক ওষুধ জানতে চাইলেও না।
- যদি 'কালচারাল ওনলি' বা ছত্রাকনাশক অকার্যকর বলে উল্লেখ থাকে (যেমন ব্যাকটেরিয়াল লিফ ব্লাইট বা টুংরো),
  তবে স্পষ্ট জানাবে যে ছত্রাকনাশক কাজ করবে না এবং শুধু পরিচর্যা ও বাহক নিয়ন্ত্রণের ধাপ বলবে।
- যদি গাছ সুস্থ বলে উল্লেখ থাকে, তবে কোনো স্প্রে না করার কথা স্পষ্টভাবে বলবে।
- ওষুধের নাম বাংলা অক্ষরে লিখবে, যেমনটি তথ্যে দেওয়া আছে।
- রাসায়নিক স্প্রে উল্লেখ থাকলে সংক্ষেপে গ্লাভস ও মাস্ক পরা এবং সাবান দিয়ে হাত ধোয়ার কথা মনে করিয়ে দেবে।
- সংখ্যা বাংলা অঙ্কে লিখবে (০-৯)।
- আবহাওয়ার তথ্য না থাকলে আবহাওয়ার কথা একেবারেই বলবে না। আবহাওয়ায় স্প্রে অনুকূল না থাকলে
  অপেক্ষা করতে ও এখনই স্প্রে না করতে বলবে এবং স্প্রে প্রযোজ্য হলে নিরাপদ সময় জানাবে।
- রোগ নিশ্চিত না হলে বা কৃষি কর্মকর্তার পরামর্শের নির্দেশ থাকলে আন্দাজে স্প্রে করতে নিষেধ করবে
  এবং স্থানীয় উপ-সহকারী কৃষি কর্মকর্তার পরামর্শ নিতে বলবে।
- তথ্যে কৃষকের প্রশ্ন থাকলে শুধু দেওয়া বারি/ব্রি তথ্য ও আবহাওয়ার ভিত্তিতে
  সরাসরি সেই প্রশ্নের উত্তর বাংলায় দেবে; উত্তর দিতে গিয়ে কোনো নতুন ওষুধ বানাবে না।
- তথ্যে 'resolved_place' বা 'extension_office_bn' থাকলে পরামর্শে সেই এলাকার নাম উল্লেখ করবে
  এবং ওই এলাকার স্থানীয় উপজেলা কৃষি অফিস ও উপ-সহকারী কৃষি কর্মকর্তার সাথে যোগাযোগের কথা বলবে
  (সবসময় কেবল বাংলা অক্ষরে লিখবে, কোনো ইংরেজি অক্ষর লিখবে না)।
- তথ্যে 'spoken_plot_hint_bn' থাকলে জমির সেই অবস্থা অনুযায়ী (যেমন নিচু জমি বা পুকুরের পাশে হলে
  দ্রুত পানি নিষ্কাশন ও জলাবদ্ধতা এড়ানো) সংক্ষিপ্ত ব্যবহারিক পরামর্শ যোগ করবে।
- তথ্যে 'crop_mismatch' সত্য হলে প্রথমেই বাংলায় স্পষ্ট জানাবে যে ছবির পাতা এবং প্রশ্নের ফসল আলাদা,
  তাই কেবল ছবির ফসলের অনুমোদিত বারি/ব্রি নির্দেশিকা থেকেই পরামর্শ দেওয়া হচ্ছে;
  প্রশ্নের অন্য ফসলের কোনো ওষুধ বা পরামর্শ দেবে না এবং কৃষককে সঠিক ফসল নিশ্চিত করতে বা
  মিল থাকা পাতার ছবি দিতে বলবে।
"""

DEFAULT_GEMINI_MODEL = "gemini-3.5-flash-lite"
FALLBACK_GEMINI_MODELS: tuple[str, ...] = (
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
)
_RETRYABLE_GEMINI_CODES = frozenset({404, 429, 500, 502, 503, 504})
_GEMINI_MODEL_COOLDOWN_SECONDS = 180.0
_PROVIDER_COOLDOWN_SECONDS = 120.0


class _SpokenAdvice(BaseModel):
    spoken: str


def _candidate_gemini_models() -> list[str]:
    primary = settings.gemini_model.strip() or DEFAULT_GEMINI_MODEL
    candidates = [primary]
    for model in FALLBACK_GEMINI_MODELS:
        if model not in candidates:
            candidates.append(model)
    return candidates


def _allowed_chemicals(facts: dict[str, Any]) -> list[str]:
    rag_grounding = facts.get("rag_grounding") or {}
    if rag_grounding.get("cultural_only"):
        return []

    allowed: list[str] = []
    for chemical in facts.get("treatment", {}).get("chemical", []):
        bn_name = chemical.get("bnName") or chemical.get("bn_name")
        if isinstance(bn_name, str) and bn_name:
            allowed.append(bn_name)

    if not facts.get("uncertain", False):
        for passage in rag_grounding.get("passages", []):
            if passage.get("cultural_only"):
                continue
            for chemical in passage.get("chemicals", []):
                bn_name = chemical.get("bn_name") or chemical.get("bnName")
                if isinstance(bn_name, str) and bn_name and bn_name not in allowed:
                    allowed.append(bn_name)
    return allowed


def _build_user_prompt(facts: dict[str, Any], allowed: list[str], *, is_retry: bool) -> str:
    payload = json.dumps(facts, ensure_ascii=False, indent=2)
    if not is_retry:
        return f"তথ্য:\n{payload}"
    retry_note = (
        "আগের উত্তরটি নিয়ম মানেনি। এবার কেবল বাংলা অক্ষরে, "
        "২ থেকে ৪ বাক্যে, কোনো ইংরেজি অক্ষর ছাড়া লেখো।"
    )
    if not allowed:
        retry_note += " কোনো ওষুধ, ছত্রাকনাশক বা রাসায়নিকের নাম উল্লেখ করবে না।"
    if (facts.get("crop_mismatch") or {}).get("detected"):
        retry_note += (
            " ফসলের অমিল থাকায় কেবল ছবির ফসলের তথ্যই ব্যবহার করবে, "
            "প্রশ্নের অন্য ফসলের কোনো ওষুধ বা পরামর্শ দেবে না।"
        )
    return f"{retry_note}\n\nতথ্য:\n{payload}"


def _parse_and_validate_spoken(raw_text: str, allowed: list[str]) -> str:
    if not raw_text:
        raise InvalidNarrative("empty")

    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise InvalidNarrative("invalid_json") from exc

    if not isinstance(parsed, dict) or not isinstance(parsed.get("spoken"), str):
        raise InvalidNarrative("missing_spoken")

    return validate_narrative(parsed["spoken"], allowed)


class GeminiWriter:
    """Primary Bangla advisory writer backed by Google Gemini Flash."""

    def __init__(self) -> None:
        from google import genai
        from google.genai import types

        api_key = settings.effective_gemini_api_key
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY (or LLM_API_KEY) is not configured for GeminiWriter."
            )

        self._client: Any = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(timeout=settings.llm_timeout_ms),
        )
        self._semaphore = asyncio.Semaphore(settings.llm_max_concurrency)
        self._model_cooldowns: dict[str, float] = {}

    async def write(self, facts: dict[str, Any]) -> tuple[str, str]:
        allowed = _allowed_chemicals(facts)
        last_error: Exception | None = None
        preferred_model: str | None = None

        for attempt in (1, 2):
            try:
                raw_text, used_model = await self._generate_raw(
                    facts,
                    allowed,
                    is_retry=attempt == 2,
                    preferred_model=preferred_model,
                )
                # Stick to whichever fallback model answered so a retry does not
                # re-hit upstream 503s from earlier candidates.
                preferred_model = used_model
                return _parse_and_validate_spoken(raw_text, allowed), "llm"
            except InvalidNarrative as exc:
                last_error = exc
                log.warning("Gemini output rejected (attempt %d): %s", attempt, exc)
            except Exception as exc:
                log.warning("Gemini call failed: %s", exc.__class__.__name__)
                raise

        raise InvalidNarrative(f"unusable_after_retry:{last_error}")

    async def _generate_raw(
        self,
        facts: dict[str, Any],
        allowed: list[str],
        *,
        is_retry: bool,
        preferred_model: str | None = None,
    ) -> tuple[str, str]:
        from google.genai import types

        user_prompt = _build_user_prompt(facts, allowed, is_retry=is_retry)
        config = types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            response_mime_type="application/json",
            response_schema=_SpokenAdvice,
            temperature=0.35,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )

        async with self._semaphore:
            response, used_model = await asyncio.to_thread(
                self._generate_with_fallback,
                user_prompt,
                config,
                preferred_model=preferred_model,
            )

        return (getattr(response, "text", None) or "").strip(), used_model

    def _generate_with_fallback(
        self,
        user_prompt: str,
        config: Any,
        *,
        preferred_model: str | None = None,
    ) -> tuple[Any, str]:
        from google.genai import errors as genai_errors

        # Lazily create in case a test instantiates via GeminiWriter.__new__.
        cooldowns: dict[str, float] | None = getattr(self, "_model_cooldowns", None)
        if cooldowns is None:
            cooldowns = {}
            self._model_cooldowns = cooldowns

        candidates = _candidate_gemini_models()
        if preferred_model and preferred_model in candidates:
            candidates = [preferred_model, *[m for m in candidates if m != preferred_model]]

        now = time.monotonic()
        active = [m for m in candidates if cooldowns.get(m, 0.0) <= now]
        cooled = [m for m in candidates if cooldowns.get(m, 0.0) > now]

        last_exc: Exception | None = None
        for model_name in [*active, *cooled]:
            try:
                response = self._client.models.generate_content(
                    model=model_name,
                    contents=user_prompt,
                    config=config,
                )
                cooldowns.pop(model_name, None)
                return response, model_name
            except genai_errors.APIError as exc:
                last_exc = exc
                if exc.code in _RETRYABLE_GEMINI_CODES:
                    cooldowns[model_name] = time.monotonic() + _GEMINI_MODEL_COOLDOWN_SECONDS
                    log.warning(
                        "Gemini model %s returned HTTP %s; cooling down %.0fs and trying fallback.",
                        model_name,
                        exc.code,
                        _GEMINI_MODEL_COOLDOWN_SECONDS,
                    )
                    continue
                raise
            except (TimeoutError, httpx.TimeoutException) as exc:
                last_exc = exc
                cooldowns[model_name] = time.monotonic() + _GEMINI_MODEL_COOLDOWN_SECONDS
                log.warning(
                    "Gemini model %s timed out (%s); cooling down %.0fs and trying fallback.",
                    model_name,
                    exc.__class__.__name__,
                    _GEMINI_MODEL_COOLDOWN_SECONDS,
                )
                continue

        raise RuntimeError(f"All Gemini Flash models failed: {last_exc}") from last_exc

_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "speak_advice",
        "description": "কৃষকের জন্য মুখে বলার মতো সংক্ষিপ্ত বাংলা পরামর্শ",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {"spoken": {"type": "string", "description": "বাংলা ভাষায় ২-৪ বাক্যের পরামর্শ"}},
            "required": ["spoken"],
            "additionalProperties": False,
        },
    },
}


class LlmWriter:
    def __init__(self) -> None:
        # Lazy import keeps the package importable without the SDK installed.
        from openai import AsyncOpenAI

        self._client = AsyncOpenAI(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            timeout=settings.llm_timeout_ms / 1000,
            # The SDK's default retry would re-send on 429 window_exhausted,
            # which resets in hours. Retrying that burns quota for nothing.
            max_retries=0,
        )
        self._semaphore = asyncio.Semaphore(settings.llm_max_concurrency)

    async def write(self, facts: dict[str, Any]) -> tuple[str, str]:
        allowed = _allowed_chemicals(facts)
        last_error: Exception | None = None

        for attempt in (1, 2):
            try:
                text = await self._attempt(facts, allowed, is_retry=attempt == 2)
                return text, "llm"
            except InvalidNarrative as exc:
                last_error = exc
                log.warning("LLM output rejected (attempt %d): %s", attempt, exc)
            except Exception as exc:  # network, auth, rate limit, spend cap
                log.warning("LLM call failed: %s", exc.__class__.__name__)
                raise

        raise InvalidNarrative(f"unusable_after_retry:{last_error}")

    async def _attempt(self, facts: dict[str, Any], allowed: list[str], *, is_retry: bool) -> str:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _build_user_prompt(facts, allowed, is_retry=is_retry)},
        ]

        async with self._semaphore:
            response = await self._client.chat.completions.create(
                model=settings.llm_model,
                messages=messages,  # type: ignore[arg-type]
                tools=[_TOOL],  # type: ignore[list-item]
                tool_choice={"type": "function", "function": {"name": "speak_advice"}},
                temperature=0.4,
                # Bengali needs more tokens than English for the same content;
                # a limit tuned in English truncates the JSON.
                max_tokens=600,
            )

        choice = response.choices[0]
        if choice.finish_reason == "length":
            raise InvalidNarrative("truncated")

        calls = choice.message.tool_calls or []
        if not calls:
            raise InvalidNarrative("no_tool_call")

        arguments = calls[0].function.arguments
        spoken = json.loads(arguments)["spoken"]
        return validate_narrative(spoken, allowed)


def _bangla_only(text: str) -> str:
    """Strip stray Latin spans/empty parens so TemplateWriter never emits Latin for TTS."""
    cleaned = re.sub(r"[A-Za-z]+", " ", normalise(text))
    cleaned = re.sub(r"\(\s*\)", "", cleaned)
    cleaned = re.sub(r"\s*,\s*", ", ", cleaned).strip(" ,")
    return normalise(cleaned)


class TemplateWriter:
    """Composes the narrative from facts alone. No network, no model, no risk."""

    async def write(self, facts: dict[str, Any]) -> tuple[str, str]:
        return self.compose(facts), "template"

    def compose(self, facts: dict[str, Any]) -> str:
        diagnosis = facts["diagnosis"]
        treatment = facts["treatment"]
        weather = facts.get("weather")
        uncertain: bool = facts.get("uncertain", False)
        rag_grounding = facts.get("rag_grounding") or {}
        cultural_only = bool(rag_grounding.get("cultural_only"))
        is_healthy = bool(diagnosis.get("healthy")) or bool(rag_grounding.get("is_healthy"))
        refer_to_officer = bool(rag_grounding.get("refer_to_officer"))
        has_question = bool(str(facts.get("farmer_question") or "").strip())
        crop_mismatch = facts.get("crop_mismatch") or {}
        mismatch_detected = bool(crop_mismatch.get("detected"))
        resolved_place = facts.get("resolved_place")
        plot_hint_bn = facts.get("spoken_plot_hint_bn")
        extension_office_bn = facts.get("extension_office_bn")

        parts: list[str] = []
        if mismatch_detected:
            photo_crop_bn = _bangla_only(
                str(crop_mismatch.get("photo_crop_bn") or diagnosis["crop"])
            )
            question_crop_bn = _bangla_only(
                str(crop_mismatch.get("question_crop_bn") or "অন্য ফসল")
            )
            source_label_bn = (
                "প্রশ্ন বা নির্বাচনে"
                if crop_mismatch.get("selection_mismatch")
                else "আপনার প্রশ্নে"
            )
            parts.append(
                f"সতর্কতা: ছবিতে {photo_crop_bn} পাতা দেখা যাচ্ছে, কিন্তু {source_label_bn} "
                f"{question_crop_bn} ফসলের কথা বলা হয়েছে। বিভ্রান্তি এড়াতে কেবল ছবির "
                f"{photo_crop_bn} ফসলের যাচাইকৃত নির্দেশিকা থেকেই পরামর্শ দিচ্ছি।"
            )
        elif has_question:
            parts.append("আপনার প্রশ্নের প্রেক্ষিতে জানাচ্ছি,")

        if uncertain:
            parts.append(
                f"আপনার {diagnosis['crop']} গাছের ছবি ভালোভাবে শনাক্ত করা যায়নি।"
                " আন্দাজে কোনো ওষুধ স্প্রে করবেন না; আরও পরিষ্কার ছবি দিন, অথবা স্থানীয়"
                " উপ-সহকারী কৃষি কর্মকর্তাকে দেখান।"
            )
        elif is_healthy:
            # The model found nothing wrong; never invite an unnecessary spray.
            parts.append(
                f"আপনার {diagnosis['crop']} গাছটি সুস্থ দেখাচ্ছে, কোনো ছত্রাকনাশক বা"
                " রাসায়নিক স্প্রে করার প্রয়োজন নেই।"
            )
            organic = treatment.get("organic") or []
            if organic:
                parts.append(f"{organic[0]}।")
            elif facts.get("prevention"):
                parts.append(f"{facts['prevention'][0]}।")
        elif cultural_only:
            parts.append(
                f"আপনার {diagnosis['crop']} গাছে সম্ভবত {diagnosis['disease']} হয়েছে।"
                " এতে কোনো ছত্রাকনাশক স্প্রে কাজ করবে না।"
            )
            organic = treatment.get("organic") or []
            if organic:
                parts.append(f"{organic[0]}।")
            if refer_to_officer:
                parts.append("সঠিক পরামর্শের জন্য স্থানীয় উপ-সহকারী কৃষি কর্মকর্তাকে দেখান।")
        else:
            parts.append(f"আপনার {diagnosis['crop']} গাছে সম্ভবত {diagnosis['disease']} হয়েছে।")

            symptoms = facts.get("symptoms_observed") or []
            if symptoms:
                parts.append(f"লক্ষণ হিসেবে দেখা যাচ্ছে: {symptoms[0]}।")

            chemicals = treatment.get("chemical") or []
            if chemicals:
                first = chemicals[0]
                parts.append(
                    f"প্রাথমিকভাবে {first['bnName']} ব্যবহার করতে পারেন, "
                    f"মাত্রা {first['dose']}। প্রয়োগের আগে কৃষি কর্মকর্তার পরামর্শ নিন।"
                )
            elif treatment.get("organic"):
                parts.append(f"{treatment['organic'][0]}।")

        if weather and not is_healthy:
            if weather["spray_window_ok"] and not cultural_only and not uncertain:
                parts.append("আবহাওয়া এখন স্প্রে করার অনুকূল।")
            elif not weather["spray_window_ok"]:
                parts.append(f"তবে এখনই স্প্রে করবেন না, কারণ {weather['reason']}।")
                safest = weather.get("safest_window_summary_bn")
                if safest and not cultural_only and not uncertain:
                    parts.append(f"{safest}।")

        if plot_hint_bn:
            safe_hint = _bangla_only(str(plot_hint_bn))
            if safe_hint:
                if any(
                    kw in safe_hint
                    for kw in ("নিচু", "পুকুর", "জলাবদ্ধ", "পানি", "খাল", "বিল", "হাওর")
                ):
                    parts.append(
                        f"আপনার {safe_hint} হওয়ায় দ্রুত পানি নিষ্কাশন ও জলাবদ্ধতা এড়ানোর ব্যবস্থা রাখুন।"
                    )
                else:
                    parts.append(
                        f"আপনার {safe_hint} বিবেচনায় নিয়মিত ক্ষেত পর্যবেক্ষণ ও পরিচর্যা বজায় রাখুন।"
                    )

        if extension_office_bn or resolved_place:
            raw_office = str(
                extension_office_bn
                or f"{resolved_place} এলাকার উপজেলা কৃষি অফিস ও উপ-সহকারী কৃষি কর্মকর্তা"
            )
            safe_office = _bangla_only(raw_office)
            if safe_office:
                parts.append(f"মাঠ পর্যায়ের সহায়তার জন্য {safe_office}-এর সাথে যোগাযোগ করুন।")

        if mismatch_detected:
            parts.append("অনুগ্রহ করে সঠিক ফসল নিশ্চিত করুন অথবা মিল থাকা পাতার নতুন ছবি দিন।")

        return normalise(" ".join(parts))


class FallbackWriter:
    """Try configured LLM writers in priority order, cooling down failing providers."""

    def __init__(
        self,
        writers: list[Writer],
        *,
        cooldown_seconds: float = _PROVIDER_COOLDOWN_SECONDS,
    ) -> None:
        if not writers:
            raise ValueError("FallbackWriter requires at least one writer")
        self._writers = writers
        self._cooldown_seconds = cooldown_seconds
        self._cooldown_until: dict[int, float] = {}

    async def write(self, facts: dict[str, Any]) -> tuple[str, str]:
        now = time.monotonic()
        active = [
            idx for idx in range(len(self._writers)) if self._cooldown_until.get(idx, 0.0) <= now
        ]
        cooled = [
            idx for idx in range(len(self._writers)) if self._cooldown_until.get(idx, 0.0) > now
        ]
        order = [*active, *cooled]

        failed_indices: list[int] = []
        last_exc: Exception | None = None
        for pos, idx in enumerate(order):
            writer = self._writers[idx]
            try:
                result = await writer.write(facts)
                self._cooldown_until.pop(idx, None)
                cooldown_deadline = time.monotonic() + self._cooldown_seconds
                for failed_idx in failed_indices:
                    self._cooldown_until[failed_idx] = cooldown_deadline
                return result
            except Exception as exc:  # noqa: BLE001 - fall through to secondary LLM writer
                last_exc = exc
                failed_indices.append(idx)
                next_name = (
                    self._writers[order[pos + 1]].__class__.__name__
                    if pos + 1 < len(order)
                    else "TemplateWriter"
                )
                log.warning(
                    "Writer %s failed (%s); falling back to %s.",
                    writer.__class__.__name__,
                    exc.__class__.__name__,
                    next_name,
                )

        if last_exc is not None:
            raise last_exc
        raise RuntimeError("No LLM writers available")


def _try_build(factory: Callable[[], Writer], label: str) -> Writer | None:
    try:
        return factory()
    except Exception as exc:  # noqa: BLE001 - any construction failure degrades
        log.error("%s writer unavailable (%s); skipping", label, exc)
        return None


def build_writer() -> Writer:
    if not settings.llm_configured:
        return TemplateWriter()

    if settings.llm_provider == "openai":
        return _try_build(LlmWriter, "OpenAI-compatible") or TemplateWriter()

    has_gemini_key = bool(settings.gemini_api_key.strip())
    has_openai_key = bool(settings.llm_api_key.strip())
    if settings.llm_provider == "auto" and has_gemini_key and has_openai_key:
        writers = [
            writer
            for writer in (
                _try_build(GeminiWriter, "Gemini"),
                _try_build(LlmWriter, "OpenAI-compatible"),
            )
            if writer is not None
        ]
        if len(writers) >= 2:
            return FallbackWriter(writers)
        if len(writers) == 1:
            return writers[0]
        return TemplateWriter()

    return _try_build(GeminiWriter, "Gemini") or TemplateWriter()


_writer: Writer | None = None


def get_writer() -> Writer:
    global _writer
    if _writer is None:
        _writer = build_writer()
    return _writer


def reset_writer() -> None:
    """Drop the cached writer. Used by tests after changing settings."""
    global _writer
    _writer = None
