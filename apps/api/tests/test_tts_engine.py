"""TTS engine concurrency and caching.

The model is not loaded here: `_synthesise_blocking` is replaced with a stub that
records how many calls overlap. What is being tested is the scheduling rule, not
the voice.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from app.tts import engine as engine_module
from app.tts.engine import AudioClip, TtsEngine


@pytest.fixture()
def engine(tmp_path, monkeypatch) -> TtsEngine:
    monkeypatch.setattr(engine_module.settings, "tts_enabled", True)
    monkeypatch.setattr(engine_module.settings, "ffmpeg_path", "ffmpeg")
    instance = TtsEngine()
    # Pretend the model is loaded so ensure_loaded() is a no-op.
    instance._tts = object()
    return instance


async def test_synthesis_is_serialised(engine: TtsEngine, tmp_path: Path, monkeypatch) -> None:
    active = 0
    peak = 0

    def fake_blocking(prepared: str, out_dir: Path, target: Path) -> AudioClip:
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        try:
            # Stand in for the CPU-bound model call.
            import time

            time.sleep(0.12)
            target.write_bytes(b"fake")
            return AudioClip(target, 1000, 4, "audio/ogg")
        finally:
            active -= 1

    monkeypatch.setattr(engine, "_synthesise_blocking", fake_blocking)

    texts = [f"এটি পরামর্শ নম্বর {i} যা কিছুক্ষণ সময় নেবে" for i in range(4)]
    await asyncio.gather(*(engine.synthesise(text, tmp_path) for text in texts))

    assert peak == 1, f"expected serialised synthesis, saw {peak} concurrent calls"


async def test_identical_text_is_synthesised_once(
    engine: TtsEngine, tmp_path: Path, monkeypatch
) -> None:
    """Audio is content-addressed, so the same sentence never costs twice."""
    calls = 0

    def fake_blocking(prepared: str, out_dir: Path, target: Path) -> AudioClip:
        nonlocal calls
        calls += 1
        target.write_bytes(b"fake")
        return AudioClip(target, 1000, 4, "audio/ogg")

    monkeypatch.setattr(engine, "_synthesise_blocking", fake_blocking)

    text = "একই পরামর্শ দুইবার"
    await engine.synthesise(text, tmp_path)
    await engine.synthesise(text, tmp_path)

    assert calls == 1


async def test_concurrent_identical_requests_hit_cache(
    engine: TtsEngine, tmp_path: Path, monkeypatch
) -> None:
    """The re-check inside the lock means a waiter does not re-synthesise."""
    calls = 0

    def fake_blocking(prepared: str, out_dir: Path, target: Path) -> AudioClip:
        nonlocal calls
        calls += 1
        import time

        time.sleep(0.1)
        target.write_bytes(b"fake")
        return AudioClip(target, 1000, 4, "audio/ogg")

    monkeypatch.setattr(engine, "_synthesise_blocking", fake_blocking)

    text = "একসাথে ডাকা একই বাক্য"
    await asyncio.gather(*(engine.synthesise(text, tmp_path) for _ in range(4)))

    assert calls == 1, f"synthesised {calls} times for identical text"


async def test_prepare_runs_before_synthesis(
    engine: TtsEngine, tmp_path: Path, monkeypatch
) -> None:
    """The model must never see a raw slash or Latin text."""
    seen: list[str] = []

    def fake_blocking(prepared: str, out_dir: Path, target: Path) -> AudioClip:
        seen.append(prepared)
        target.write_bytes(b"fake")
        return AudioClip(target, 1000, 4, "audio/ogg")

    monkeypatch.setattr(engine, "_synthesise_blocking", fake_blocking)

    await engine.synthesise("মাত্রা ২ মিলি/লিটার", tmp_path)
    assert seen == ["মাত্রা ২ মিলি প্রতি লিটার"]


async def test_wav_fallback_when_ffmpeg_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import wave

    import numpy as np

    monkeypatch.setattr(engine_module.settings, "tts_enabled", True)
    monkeypatch.setattr(engine_module.settings, "tts_format", "opus")
    monkeypatch.setattr(engine_module.settings, "ffmpeg_path", "ffmpeg")
    monkeypatch.setattr(engine_module.shutil, "which", lambda _cmd: None)

    eng = TtsEngine()
    assert eng.available is True

    class StubTts:
        def synthesize(self, _text: str, voice: str = "female") -> np.ndarray:
            return np.zeros(24_000, dtype=np.float32)

    eng._tts = StubTts()
    clip = await eng.synthesise("আপনার ধান গাছ সুস্থ আছে।", tmp_path)

    assert clip.path.suffix == ".wav"
    assert clip.path.exists()
    assert clip.mime == "audio/wav"
    assert clip.duration_ms == 1000
    assert clip.byte_size > 44
    with wave.open(str(clip.path), "rb") as handle:
        assert handle.getnchannels() == 1
        assert handle.getsampwidth() == 2
        assert handle.getframerate() == 24_000


def test_get_audio_serves_wav_with_200_and_audio_wav_mime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import io
    import time

    import numpy as np
    from fastapi.testclient import TestClient
    from PIL import Image

    from app.config import settings
    from app.llm.writer import reset_writer
    from app.main import app

    storage = tmp_path / "storage"
    monkeypatch.setattr(settings, "database_path", storage / "test.db")
    monkeypatch.setattr(settings, "storage_dir", storage)
    monkeypatch.setattr(settings, "tts_enabled", True)
    monkeypatch.setattr(settings, "tts_format", "opus")
    monkeypatch.setattr(settings, "vision_enabled", False)
    monkeypatch.setattr(settings, "gemini_api_key", "")
    monkeypatch.setattr(settings, "llm_api_key", "")
    monkeypatch.setattr(engine_module.shutil, "which", lambda _cmd: None)
    for directory in (settings.images_dir, settings.audio_dir):
        directory.mkdir(parents=True, exist_ok=True)

    reset_writer()
    engine_module.get_engine.cache_clear()
    eng = engine_module.get_engine()

    class StubTts:
        def synthesize(self, _text: str, voice: str = "female") -> np.ndarray:
            return np.zeros(12_000, dtype=np.float32)

    eng._tts = StubTts()

    buf = io.BytesIO()
    Image.new("RGB", (64, 64), (40, 120, 60)).save(buf, format="JPEG")

    try:
        with TestClient(app) as client:
            created = client.post(
                "/api/v1/analyses",
                files={"image": ("crop.jpg", buf.getvalue(), "image/jpeg")},
                data={"crop": "rice"},
            )
            assert created.status_code == 202
            analysis_id = created.json()["id"]

            deadline = time.monotonic() + 15.0
            while time.monotonic() < deadline:
                summary = client.get(f"/api/v1/analyses/{analysis_id}").json()
                if summary["status"] == "completed" and summary["stage"] == "done":
                    break
                time.sleep(0.05)
            else:
                raise AssertionError("TTS background task did not finish")

            result = client.get(f"/api/v1/analyses/{analysis_id}/result").json()
            assert result["audio"]["status"] == "ready"
            assert result["audio"]["url"] == f"/api/v1/analyses/{analysis_id}/audio"
            assert list(settings.audio_dir.glob("*.wav"))

            audio_resp = client.get(f"/api/v1/analyses/{analysis_id}/audio")
            assert audio_resp.status_code == 200
            assert audio_resp.headers["content-type"] == "audio/wav"
            assert audio_resp.content.startswith(b"RIFF")
    finally:
        engine_module.get_engine.cache_clear()
        reset_writer()


def test_post_tts_validation_and_unavailable_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fastapi.testclient import TestClient

    from app.config import settings
    from app.main import app
    from app.routes import clear_tts_memory_cache

    storage = tmp_path / "storage"
    monkeypatch.setattr(settings, "database_path", storage / "test.db")
    monkeypatch.setattr(settings, "storage_dir", storage)
    monkeypatch.setattr(settings, "vision_enabled", False)
    for directory in (settings.images_dir, settings.audio_dir):
        directory.mkdir(parents=True, exist_ok=True)

    clear_tts_memory_cache()
    engine_module.get_engine.cache_clear()

    try:
        with TestClient(app) as client:
            assert client.post("/api/v1/tts", json={}).status_code == 422

            for bad_text in ("", "   ", "Hello PPE"):
                resp = client.post("/api/v1/tts", json={"text": bad_text})
                assert resp.status_code == 400
                assert resp.json()["detail"] == "empty_text"

            too_long = client.post("/api/v1/tts", json={"text": "ক" * 601})
            assert too_long.status_code == 400
            assert too_long.json()["detail"] == "text_too_long"

            monkeypatch.setattr(settings, "tts_enabled", False)
            unavailable = client.post("/api/v1/tts", json={"text": "এখন স্প্রে করার অনুকূল সময়।"})
            assert unavailable.status_code == 503
            assert unavailable.json()["detail"] == "tts_unavailable"
    finally:
        clear_tts_memory_cache()
        engine_module.get_engine.cache_clear()


def test_post_tts_synthesises_strips_latin_and_caches_in_memory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fastapi.testclient import TestClient

    from app.config import settings
    from app.main import app
    from app.routes import clear_tts_memory_cache

    storage = tmp_path / "storage"
    monkeypatch.setattr(settings, "database_path", storage / "test.db")
    monkeypatch.setattr(settings, "storage_dir", storage)
    monkeypatch.setattr(settings, "tts_enabled", True)
    monkeypatch.setattr(settings, "tts_format", "mp3")
    monkeypatch.setattr(settings, "vision_enabled", False)
    for directory in (settings.images_dir, settings.audio_dir):
        directory.mkdir(parents=True, exist_ok=True)

    clear_tts_memory_cache()
    engine_module.get_engine.cache_clear()
    eng = engine_module.get_engine()

    seen_texts: list[str] = []

    async def fake_synthesise(text: str, out_dir: Path, required_terms=None) -> AudioClip:
        seen_texts.append(text)
        out_dir.mkdir(parents=True, exist_ok=True)
        clip_path = out_dir / "card.mp3"
        clip_path.write_bytes(b"ID3fake-mp3-bytes")
        return AudioClip(clip_path, 900, 17, "audio/mpeg")

    monkeypatch.setattr(eng, "synthesise", fake_synthesise)

    try:
        with TestClient(app) as client:
            payload = {"text": "**পরামর্শ** (PPE): এখন স্প্রে করুন।"}
            first = client.post("/api/v1/tts", json=payload)
            assert first.status_code == 200
            assert first.headers["content-type"].startswith("audio/mpeg")
            assert first.content == b"ID3fake-mp3-bytes"
            assert seen_texts == ["পরামর্শ : এখন স্প্রে করুন।"]

            second = client.post("/api/v1/tts", json=payload)
            assert second.status_code == 200
            assert second.content == b"ID3fake-mp3-bytes"
            assert len(seen_texts) == 1
    finally:
        clear_tts_memory_cache()
        engine_module.get_engine.cache_clear()
