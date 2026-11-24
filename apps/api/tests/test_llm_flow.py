"""The LLM writer's control flow, without a network call.

The provider is never contacted here: the SDK client is replaced with a stub. That
keeps these tests runnable with no key, and keeps them fast and deterministic.
What is being tested is the part that matters - the retry, the tool-call parsing,
and the fallback when the model misbehaves.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from app.llm import writer as writer_module
from app.llm.writer import (
    FallbackWriter,
    GeminiWriter,
    InvalidNarrative,
    LlmWriter,
    TemplateWriter,
    get_writer,
    reset_writer,
)

FACTS = {
    "diagnosis": {"crop": "ধান", "disease": "ব্লাস্ট রোগ", "confidence": "মাঝারি"},
    "symptoms_observed": ["পাতায় ধূসর-সবুজ দিম্বাকৃতি দাগ"],
    "treatment": {
        "organic": ["আক্রান্ত পাতা সরান"],
        "chemical": [
            {
                "product": "Tricyclazole",
                "bnName": "ট্রাইসাইক্লাজল",
                "dose": "প্রতি লিটারে ০.৬ গ্রাম",
                "note": "৭ দিন পরপর",
            }
        ],
    },
    "prevention": ["অতিরিক্ত ইউরিয়া দেবেন না"],
    "weather": None,
    "uncertain": False,
}

GOOD_SPOKEN = (
    "আপনার ধান গাছে সম্ভবত ব্লাস্ট রোগ হয়েছে। "
    "পাতায় ধূসর-সবুজ দাগ দেখা যাচ্ছে। "
    "ট্রাইসাইক্লাজল প্রতি লিটারে ০.৬ গ্রাম হিসেবে ব্যবহার করতে পারেন। "
    "প্রয়োগের আগে স্থানীয় কৃষি কর্মকর্তার পরামর্শ নিন এবং গ্লাভস ও মাস্ক পরুন।"
)


def _response(spoken: str | None, *, finish_reason: str = "tool_calls") -> SimpleNamespace:
    if spoken is None:
        message = SimpleNamespace(tool_calls=None)
        return SimpleNamespace(
            choices=[SimpleNamespace(finish_reason=finish_reason, message=message)]
        )
    call = SimpleNamespace(function=SimpleNamespace(arguments=json.dumps({"spoken": spoken})))
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason=finish_reason,
                message=SimpleNamespace(tool_calls=[call]),
            )
        ]
    )


class StubClient:
    """Records calls and replays queued responses."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls: list[dict] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    async def _create(self, **kwargs):
        self.calls.append(kwargs)
        result = self._responses.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


def _writer_with(responses) -> tuple[LlmWriter, StubClient]:
    writer = LlmWriter.__new__(LlmWriter)  # skip the real SDK client
    import asyncio

    writer._semaphore = asyncio.Semaphore(3)
    client = StubClient(responses)
    writer._client = client
    return writer, client


async def test_good_output_is_accepted() -> None:
    writer, client = _writer_with([_response(GOOD_SPOKEN)])
    text, generated_by = await writer.write(FACTS)

    assert generated_by == "llm"
    assert "ট্রাইসাইক্লাজল" in text
    # A forced tool call, not response_format: the gateway drops the latter.
    forced = {"type": "function", "function": {"name": "speak_advice"}}
    assert client.calls[0]["tool_choice"] == forced
    assert client.calls[0]["max_tokens"] == 600
    assert client.calls[0]["temperature"] == 0.4


async def test_bad_then_good_retries_once() -> None:
    """A rejected answer gets exactly one more attempt, with a correction."""
    writer, client = _writer_with([_response("ধান"), _response(GOOD_SPOKEN)])
    text, generated_by = await writer.write(FACTS)

    assert generated_by == "llm"
    assert "ট্রাইসাইক্লাজল" in text
    assert len(client.calls) == 2

    retry_messages = client.calls[1]["messages"]
    assert "আগের উত্তরটি নিয়ম মানেনি" in retry_messages[-1]["content"]


async def test_two_bad_answers_raise_for_fallback() -> None:
    writer, _ = _writer_with([_response("ধান"), _response("ধান")])
    with pytest.raises(InvalidNarrative, match="unusable_after_retry"):
        await writer.write(FACTS)


async def test_missing_tool_call_raises() -> None:
    writer, _ = _writer_with([_response(None), _response(None)])
    with pytest.raises(InvalidNarrative, match="unusable_after_retry"):
        await writer.write(FACTS)


async def test_truncated_output_is_rejected() -> None:
    """Truncation is an invalid answer, so it gets the single retry like any other."""
    writer, client = _writer_with(
        [_response(GOOD_SPOKEN, finish_reason="length"), _response(GOOD_SPOKEN)]
    )
    text, generated_by = await writer.write(FACTS)

    assert generated_by == "llm"
    assert len(client.calls) == 2
    assert "ট্রাইসাইক্লাজল" in text


async def test_truncation_twice_raises_for_fallback() -> None:
    writer, _ = _writer_with(
        [
            _response(GOOD_SPOKEN, finish_reason="length"),
            _response(GOOD_SPOKEN, finish_reason="length"),
        ]
    )
    with pytest.raises(InvalidNarrative, match="unusable_after_retry"):
        await writer.write(FACTS)


async def test_network_error_propagates_for_the_caller_to_handle() -> None:
    """The pipeline catches this and falls back; the writer does not retry it.

    Retrying a 429 `window_exhausted` is pointless - it resets in hours.
    """
    writer, client = _writer_with([RuntimeError("429 window_exhausted")])
    with pytest.raises(RuntimeError):
        await writer.write(FACTS)
    assert len(client.calls) == 1


async def test_template_writer_needs_no_client() -> None:
    text, generated_by = await TemplateWriter().write(FACTS)
    assert generated_by == "template"
    assert "ট্রাইসাইক্লাজল" in text
    assert "আনুমানিক পরামর্শ" not in text


async def test_template_writer_mentions_weather_only_when_present() -> None:
    facts = dict(FACTS)
    facts["weather"] = {
        "district": "ঢাকা",
        "rain_probability_next_12h": 97,
        "wind_speed_max_next_12h": 9.6,
        "humidity": 95,
        "spray_window_ok": False,
        "reason": "আগামী ১২ ঘণ্টায় বৃষ্টির সম্ভাবনা ৯৭ শতাংশ",
    }
    text, _ = await TemplateWriter().write(facts)
    assert "স্প্রে করবেন না" in text
    assert "৯৭ শতাংশ" in text

    text_without, _ = await TemplateWriter().write(FACTS)
    assert "স্প্রে" not in text_without
    assert "আবহাওয়া" not in text_without


def test_writer_selection_falls_back_without_a_key(monkeypatch) -> None:
    monkeypatch.setattr(writer_module.settings, "llm_provider", "auto")
    monkeypatch.setattr(writer_module.settings, "gemini_api_key", "")
    monkeypatch.setattr(writer_module.settings, "llm_api_key", "")
    reset_writer()
    try:
        assert isinstance(get_writer(), TemplateWriter)
    finally:
        reset_writer()


def test_writer_selection_uses_gemini_when_gemini_or_llm_key_configured(monkeypatch) -> None:
    monkeypatch.setattr(writer_module.settings, "llm_provider", "auto")
    monkeypatch.setattr(writer_module.settings, "gemini_api_key", "test-key-not-real")
    monkeypatch.setattr(writer_module.settings, "llm_api_key", "")
    reset_writer()
    try:
        assert isinstance(get_writer(), GeminiWriter)
    finally:
        reset_writer()

    monkeypatch.setattr(writer_module.settings, "gemini_api_key", "")
    monkeypatch.setattr(writer_module.settings, "llm_api_key", "test-key-not-real")
    reset_writer()
    try:
        assert isinstance(get_writer(), GeminiWriter)
    finally:
        reset_writer()


def test_writer_selection_uses_openai_when_explicitly_selected(monkeypatch) -> None:
    monkeypatch.setattr(writer_module.settings, "llm_provider", "openai")
    monkeypatch.setattr(writer_module.settings, "gemini_api_key", "")
    monkeypatch.setattr(writer_module.settings, "llm_api_key", "test-key-not-real")
    reset_writer()
    try:
        assert isinstance(get_writer(), LlmWriter)
    finally:
        reset_writer()


def test_writer_selection_builds_fallback_writer_when_both_keys_configured(monkeypatch) -> None:
    monkeypatch.setattr(writer_module.settings, "llm_provider", "auto")
    monkeypatch.setattr(writer_module.settings, "gemini_api_key", "test-gemini-not-real")
    monkeypatch.setattr(writer_module.settings, "llm_api_key", "test-openai-not-real")
    reset_writer()
    try:
        assert isinstance(get_writer(), FallbackWriter)
    finally:
        reset_writer()

    monkeypatch.setattr(writer_module.settings, "llm_provider", "gemini")
    reset_writer()
    try:
        assert isinstance(get_writer(), GeminiWriter)
    finally:
        reset_writer()


class StubGeminiModels:
    """Records Gemini generate_content calls and replays queued responses."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls: list[dict] = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        result = self._responses.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


def _gemini_writer_with(responses) -> tuple[GeminiWriter, StubGeminiModels]:
    import asyncio

    writer = GeminiWriter.__new__(GeminiWriter)
    writer._semaphore = asyncio.Semaphore(3)
    models = StubGeminiModels(responses)
    writer._client = SimpleNamespace(models=models)
    return writer, models


async def test_gemini_good_output_is_accepted(monkeypatch) -> None:
    monkeypatch.setattr(writer_module.settings, "gemini_model", writer_module.DEFAULT_GEMINI_MODEL)
    writer, models = _gemini_writer_with(
        [SimpleNamespace(text=json.dumps({"spoken": GOOD_SPOKEN}, ensure_ascii=False))]
    )
    text, generated_by = await writer.write(FACTS)

    assert generated_by == "llm"
    assert "ট্রাইসাইক্লাজল" in text
    assert models.calls[0]["model"] == "gemini-3.5-flash-lite"


def test_settings_defaults_use_flash_lite_and_mp3() -> None:
    from app.config import Settings

    defaults = Settings(_env_file=None)
    assert defaults.gemini_model == "gemini-3.5-flash-lite"
    assert defaults.tts_format == "mp3"


async def test_gemini_fallback_models_on_retryable_status_codes(monkeypatch) -> None:
    from google.genai import errors as genai_errors

    monkeypatch.setattr(writer_module.settings, "gemini_model", "gemini-3.8-flash")
    writer, models = _gemini_writer_with(
        [
            genai_errors.APIError(404, {"error": {"message": "not found"}}),
            genai_errors.APIError(429, {"error": {"message": "rate limited"}}),
            genai_errors.APIError(503, {"error": {"message": "overloaded"}}),
            SimpleNamespace(text=json.dumps({"spoken": GOOD_SPOKEN}, ensure_ascii=False)),
        ]
    )
    text, generated_by = await writer.write(FACTS)

    assert generated_by == "llm"
    assert "ট্রাইসাইক্লাজল" in text
    assert [call["model"] for call in models.calls] == [
        "gemini-3.8-flash",
        "gemini-3.7-flash",
        "gemini-3.5-flash",
        "gemini-3.5-flash-lite",
    ]


async def test_gemini_non_retryable_api_error_propagates_immediately(monkeypatch) -> None:
    from google.genai import errors as genai_errors

    monkeypatch.setattr(writer_module.settings, "gemini_model", "gemini-3.8-flash")
    writer, models = _gemini_writer_with(
        [genai_errors.APIError(400, {"error": {"message": "bad request"}})]
    )
    with pytest.raises(genai_errors.APIError):
        await writer.write(FACTS)
    assert len(models.calls) == 1


async def test_gemini_invalid_narrative_retries_once_then_succeeds() -> None:
    writer, models = _gemini_writer_with(
        [
            SimpleNamespace(text=json.dumps({"spoken": "ধান"}, ensure_ascii=False)),
            SimpleNamespace(text=json.dumps({"spoken": GOOD_SPOKEN}, ensure_ascii=False)),
        ]
    )
    text, generated_by = await writer.write(FACTS)

    assert generated_by == "llm"
    assert "ট্রাইসাইক্লাজল" in text
    assert len(models.calls) == 2
    assert "আগের উত্তরটি নিয়ম মানেনি" in models.calls[1]["contents"]


async def test_gemini_two_bad_answers_raise_for_fallback() -> None:
    writer, _ = _gemini_writer_with(
        [
            SimpleNamespace(text=json.dumps({"spoken": "ধান"}, ensure_ascii=False)),
            SimpleNamespace(text=json.dumps({"spoken": "ধান Mancozeb"}, ensure_ascii=False)),
        ]
    )
    with pytest.raises(InvalidNarrative, match="unusable_after_retry"):
        await writer.write(FACTS)


async def test_gemini_retry_reuses_working_fallback_and_forbids_chemicals(monkeypatch) -> None:
    from google.genai import errors as genai_errors

    monkeypatch.setattr(writer_module.settings, "gemini_model", "gemini-3.8-flash")
    unconfirmed_facts = {
        "diagnosis": {"crop": "আলু", "disease": "আলুর নাবি ধসা রোগ", "confidence": None},
        "symptoms_observed": [],
        "treatment": {"organic": ["আক্রান্ত গাছ সরিয়ে ফেলুন"], "chemical": []},
        "prevention": ["সুষম সার ব্যবহার করুন"],
        "weather": None,
        "uncertain": True,
    }
    valid_unconfirmed_spoken = (
        "আপনার আলু গাছের ছবি থেকে রোগ নিশ্চিতভাবে বোঝা যায়নি। "
        "তাই আন্দাজে কোনো ছত্রাকনাশক বা ওষুধ স্প্রে করবেন না এবং স্থানীয় কৃষি কর্মকর্তার পরামর্শ নিন। "
        "আরও পরিষ্কার ছবি তুলে আবার চেষ্টা করুন।"
    )
    writer, models = _gemini_writer_with(
        [
            genai_errors.APIError(503, {"error": {"message": "overloaded"}}),
            genai_errors.APIError(503, {"error": {"message": "overloaded"}}),
            genai_errors.APIError(503, {"error": {"message": "overloaded"}}),
            SimpleNamespace(text=json.dumps({"spoken": "ধান"}, ensure_ascii=False)),
            SimpleNamespace(
                text=json.dumps({"spoken": valid_unconfirmed_spoken}, ensure_ascii=False)
            ),
        ]
    )
    text, generated_by = await writer.write(unconfirmed_facts)

    assert generated_by == "llm"
    assert "আন্দাজে" in text
    assert [call["model"] for call in models.calls] == [
        "gemini-3.8-flash",
        "gemini-3.7-flash",
        "gemini-3.5-flash",
        "gemini-3.5-flash-lite",
        "gemini-3.5-flash-lite",
    ]
    assert (
        "কোনো ওষুধ, ছত্রাকনাশক বা রাসায়নিকের নাম উল্লেখ করবে না"
        in models.calls[4]["contents"]
    )
    assert models.calls[0]["config"].automatic_function_calling.disable is True


async def test_gemini_504_and_timeout_fallback_cools_down_failing_models(monkeypatch) -> None:
    from google.genai import errors as genai_errors

    monkeypatch.setattr(writer_module.settings, "gemini_model", "gemini-3.8-flash")
    writer, models = _gemini_writer_with(
        [
            genai_errors.APIError(504, {"error": {"message": "deadline exceeded"}}),
            genai_errors.APIError(500, {"error": {"message": "internal error"}}),
            TimeoutError("client timeout"),
            SimpleNamespace(text=json.dumps({"spoken": GOOD_SPOKEN}, ensure_ascii=False)),
            SimpleNamespace(text=json.dumps({"spoken": GOOD_SPOKEN}, ensure_ascii=False)),
        ]
    )

    text1, gen1 = await writer.write(FACTS)
    assert gen1 == "llm"
    assert "ট্রাইসাইক্লাজল" in text1

    text2, gen2 = await writer.write(FACTS)
    assert gen2 == "llm"
    assert "ট্রাইসাইক্লাজল" in text2

    assert [call["model"] for call in models.calls] == [
        "gemini-3.8-flash",
        "gemini-3.7-flash",
        "gemini-3.5-flash",
        "gemini-3.5-flash-lite",
        "gemini-3.5-flash-lite",
    ]


async def test_fallback_writer_switches_to_secondary_on_failure_with_cooldown() -> None:
    calls: list[str] = []

    class StubPrimary:
        async def write(self, _facts):
            calls.append("primary")
            raise TimeoutError("gemini down")

    class StubSecondary:
        async def write(self, _facts):
            calls.append("secondary")
            return GOOD_SPOKEN, "llm"

    fallback = FallbackWriter([StubPrimary(), StubSecondary()], cooldown_seconds=60.0)
    text1, gen1 = await fallback.write(FACTS)
    assert (text1, gen1) == (GOOD_SPOKEN, "llm")

    # Primary is now on cooldown, so secondary is called first on the next request.
    text2, gen2 = await fallback.write(FACTS)
    assert (text2, gen2) == (GOOD_SPOKEN, "llm")
    assert calls == ["primary", "secondary", "secondary"]


async def test_fallback_writer_propagates_error_when_all_providers_fail() -> None:
    class StubPrimary:
        async def write(self, _facts):
            raise TimeoutError("gemini down")

    class StubSecondary:
        async def write(self, _facts):
            raise RuntimeError("openai down")

    fallback = FallbackWriter([StubPrimary(), StubSecondary()])
    with pytest.raises(RuntimeError, match="openai down"):
        await fallback.write(FACTS)
