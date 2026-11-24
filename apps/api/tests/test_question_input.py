"""Tests for optional Bangla voice and text question input."""

from __future__ import annotations

import io
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.config import settings
from app.contracts import AnalysisResult
from app.grounding import rag
from app.llm import audio_input
from app.llm.writer import TemplateWriter, reset_writer
from app.main import app


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    storage = tmp_path / "storage"
    monkeypatch.setattr(settings, "database_path", storage / "test.db")
    monkeypatch.setattr(settings, "storage_dir", storage)
    monkeypatch.setattr(settings, "tts_enabled", False)
    monkeypatch.setattr(settings, "vision_enabled", False)
    monkeypatch.setattr(settings, "llm_provider", "auto")
    monkeypatch.setattr(settings, "gemini_api_key", "")
    monkeypatch.setattr(settings, "llm_api_key", "")
    reset_writer()
    rag.reset_rag_cache()

    for directory in (settings.images_dir, settings.audio_dir):
        directory.mkdir(parents=True, exist_ok=True)

    with TestClient(app) as test_client:
        yield test_client

    reset_writer()


def _jpeg() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (64, 64), (40, 120, 60)).save(buffer, format="JPEG")
    return buffer.getvalue()


def _wait(client: TestClient, analysis_id: str, timeout: float = 20.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        summary = client.get(f"/api/v1/analyses/{analysis_id}").json()
        if summary["status"] in {"completed", "failed"}:
            return summary
        time.sleep(0.1)
    raise AssertionError("analysis did not finish")


def test_question_text_passthrough_to_rag_writer_and_result(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen_rag_queries: list[str | None] = []
    seen_writer_facts: list[dict] = []

    original_retrieve = rag.retrieve_grounding_bundle
    original_compose = TemplateWriter.compose

    def spy_retrieve(disease_id: str, crop_id: str, query_text: str | None = None, top_k: int = 3):
        seen_rag_queries.append(query_text)
        return original_retrieve(disease_id, crop_id, query_text, top_k)

    def spy_compose(self, facts: dict) -> str:
        seen_writer_facts.append(facts)
        return original_compose(self, facts)

    monkeypatch.setattr(rag, "retrieve_grounding_bundle", spy_retrieve)
    monkeypatch.setattr(TemplateWriter, "compose", spy_compose)

    question = "আজ বৃষ্টি হলে কি স্প্রে করব?"
    created = client.post(
        "/api/v1/analyses",
        files={"image": ("crop.jpg", _jpeg(), "image/jpeg")},
        data={"crop": "rice", "question_text": f"   {question}   "},
    )
    assert created.status_code == 202
    analysis_id = created.json()["id"]
    summary = _wait(client, analysis_id)
    assert summary["status"] == "completed"

    payload = client.get(f"/api/v1/analyses/{analysis_id}/result").json()
    result = AnalysisResult.model_validate(payload)
    assert result.farmerQuestion == question
    assert seen_rag_queries == [question]
    assert seen_writer_facts[0]["farmer_question"] == question
    assert "আপনার প্রশ্নের প্রেক্ষিতে" in result.narrative


def test_question_text_blank_and_max_length_trimming(client: TestClient) -> None:
    blank = client.post(
        "/api/v1/analyses",
        files={"image": ("crop.jpg", _jpeg(), "image/jpeg")},
        data={"crop": "rice", "question_text": "    \n\t  "},
    )
    assert blank.status_code == 202
    blank_id = blank.json()["id"]
    _wait(client, blank_id)
    blank_result = client.get(f"/api/v1/analyses/{blank_id}/result").json()
    assert blank_result["farmerQuestion"] is None

    long_q = "ক" * 450
    capped = client.post(
        "/api/v1/analyses",
        files={"image": ("crop.jpg", _jpeg(), "image/jpeg")},
        data={"crop": "rice", "question_text": long_q},
    )
    assert capped.status_code == 202
    capped_id = capped.json()["id"]
    _wait(client, capped_id)
    capped_result = client.get(f"/api/v1/analyses/{capped_id}/result").json()
    assert capped_result["farmerQuestion"] == "ক" * settings.max_question_chars


@pytest.mark.parametrize(
    "mime_type",
    [
        "audio/webm",
        "audio/webm;codecs=opus",
        "audio/ogg; codecs=opus",
        "audio/mp4",
        "audio/mpeg",
        "audio/wav",
        "audio/aac",
        "audio/x-m4a",
        "audio/x-wav",
        "video/mp4",
        "video/webm",
    ],
)
def test_supported_audio_mimes_accepted_and_degrade_without_key(
    client: TestClient, mime_type: str
) -> None:
    created = client.post(
        "/api/v1/analyses",
        files={
            "image": ("crop.jpg", _jpeg(), "image/jpeg"),
            "audio": ("question.webm", b"OggS\x00\x02dummy-audio-bytes", mime_type),
        },
        data={"crop": "rice"},
    )
    assert created.status_code == 202
    analysis_id = created.json()["id"]
    summary = _wait(client, analysis_id)
    assert summary["status"] == "completed"

    payload = client.get(f"/api/v1/analyses/{analysis_id}/result").json()
    result = AnalysisResult.model_validate(payload)
    assert result.farmerQuestion is None
    assert result.narrative


@pytest.mark.parametrize("mime_type", ["audio/flac", "application/octet-stream", "text/plain"])
def test_unsupported_audio_mime_is_rejected(client: TestClient, mime_type: str) -> None:
    resp = client.post(
        "/api/v1/analyses",
        files={
            "image": ("crop.jpg", _jpeg(), "image/jpeg"),
            "audio": ("bad.bin", b"1234", mime_type),
        },
        data={"crop": "rice"},
    )
    assert resp.status_code == 400
    assert resp.json()["detail"] == "unsupported_audio_type"


def test_oversized_audio_is_rejected_with_413(client: TestClient) -> None:
    too_big = b"a" * (settings.max_audio_upload_bytes + 1)
    resp = client.post(
        "/api/v1/analyses",
        files={
            "image": ("crop.jpg", _jpeg(), "image/jpeg"),
            "audio": ("big.webm", too_big, "audio/webm"),
        },
        data={"crop": "rice"},
    )
    assert resp.status_code == 413
    assert resp.json()["detail"] == "audio_too_large"


async def test_transcribe_bangla_question_uses_part_from_bytes_and_fallbacks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from google.genai import errors as genai_errors

    monkeypatch.setattr(settings, "gemini_api_key", "test-key-not-real")
    monkeypatch.setattr(settings, "gemini_model", "gemini-3.8-flash")

    calls: list[dict] = []

    class StubModels:
        def __init__(self, responses: list) -> None:
            self._responses = list(responses)

        def generate_content(self, **kwargs):
            calls.append(kwargs)
            item = self._responses.pop(0)
            if isinstance(item, Exception):
                raise item
            return item

    stub_client = SimpleNamespace(
        models=StubModels(
            [
                genai_errors.APIError(404, {"error": {"message": "not found"}}),
                SimpleNamespace(text='  **"পাতায় দাগ কেন হলো?"**  '),
            ]
        )
    )
    monkeypatch.setattr(audio_input, "_build_client", lambda *_args: stub_client)

    transcript = await audio_input.transcribe_bangla_question(
        b"webm-bytes", "audio/webm", settings
    )
    assert transcript == "পাতায় দাগ কেন হলো?"
    assert [c["model"] for c in calls] == ["gemini-3.8-flash", "gemini-3.7-flash"]

    audio_part = calls[0]["contents"][1]
    assert audio_part.inline_data.data == b"webm-bytes"
    assert audio_part.inline_data.mime_type == "audio/webm"


async def test_transcribe_bangla_question_returns_none_on_empty_or_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "gemini_api_key", "test-key-not-real")

    class EmptyModels:
        def generate_content(self, **_kwargs):
            return SimpleNamespace(text="   ")

    monkeypatch.setattr(
        audio_input, "_build_client", lambda *_args: SimpleNamespace(models=EmptyModels())
    )
    assert await audio_input.transcribe_bangla_question(b"123", "audio/webm", settings) is None

    class ExplodingModels:
        def generate_content(self, **_kwargs):
            raise TimeoutError("timed out")

    monkeypatch.setattr(
        audio_input, "_build_client", lambda *_args: SimpleNamespace(models=ExplodingModels())
    )
    assert await audio_input.transcribe_bangla_question(b"123", "audio/webm", settings) is None


def test_voice_and_text_combined_question_flows_into_pipeline(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def fake_transcribe(audio_bytes: bytes, mime_type: str, _settings) -> str | None:
        assert audio_bytes == b"voice-data"
        assert mime_type == "audio/webm"
        return "কোন সার দেব?"

    monkeypatch.setattr("app.jobs.pipeline.transcribe_bangla_question", fake_transcribe)

    created = client.post(
        "/api/v1/analyses",
        files={
            "image": ("crop.jpg", _jpeg(), "image/jpeg"),
            "audio": ("voice.webm", b"voice-data", "audio/webm;codecs=opus"),
        },
        data={"crop": "rice", "question_text": "পাতায় দাগ দেখা যাচ্ছে"},
    )
    assert created.status_code == 202
    analysis_id = created.json()["id"]
    summary = _wait(client, analysis_id)
    assert summary["status"] == "completed"

    payload = client.get(f"/api/v1/analyses/{analysis_id}/result").json()
    result = AnalysisResult.model_validate(payload)
    assert result.farmerQuestion == "পাতায় দাগ দেখা যাচ্ছে। কোন সার দেব?"


@pytest.mark.parametrize(
    ("uploaded_mime", "expected_gemini_mime"),
    [
        ("audio/x-m4a", "audio/mp4"),
        ("audio/x-m4a; codecs=mp4a.40.2", "audio/mp4"),
        ("audio/x-wav", "audio/wav"),
    ],
)
def test_audio_mime_aliases_normalize_before_gemini_part_from_bytes(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    uploaded_mime: str,
    expected_gemini_mime: str,
) -> None:
    calls: list[dict] = []

    class StubModels:
        def generate_content(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(text="আলুর পাতায় দাগ কেন?")

    monkeypatch.setattr(settings, "gemini_api_key", "test-key-not-real")
    monkeypatch.setattr(
        audio_input, "_build_client", lambda *_args: SimpleNamespace(models=StubModels())
    )
    monkeypatch.setattr("app.jobs.pipeline.get_writer", lambda: TemplateWriter())
    monkeypatch.setattr(rag, "_embed_query", lambda *_args: None)

    created = client.post(
        "/api/v1/analyses",
        files={
            "image": ("crop.jpg", _jpeg(), "image/jpeg"),
            "audio": ("voice.m4a", b"fake-audio-payload", uploaded_mime),
        },
        data={"crop": "potato"},
    )
    assert created.status_code == 202
    analysis_id = created.json()["id"]
    summary = _wait(client, analysis_id)
    assert summary["status"] == "completed"

    assert len(calls) == 1
    audio_part = calls[0]["contents"][1]
    assert audio_part.inline_data.mime_type == expected_gemini_mime
    assert audio_part.inline_data.data == b"fake-audio-payload"
