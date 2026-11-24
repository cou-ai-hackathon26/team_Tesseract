"""Concurrency behaviour of the job runner.

The box has ~3 GiB free and a CPU-only TTS model, so the point of these tests is
that a burst of uploads queues instead of loading work in parallel without limit.
"""

from __future__ import annotations

import asyncio
import io
import time

from fastapi.testclient import TestClient
from PIL import Image

from app.config import settings
from app.jobs import pipeline
from app.llm.writer import reset_writer
from app.main import app


def _jpeg() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (48, 48), (30, 110, 50)).save(buffer, format="JPEG")
    return buffer.getvalue()


def test_concurrency_is_capped(tmp_path, monkeypatch):
    """No more than CONCURRENCY pipelines run at once."""
    storage = tmp_path / "storage"
    monkeypatch.setattr(settings, "database_path", storage / "test.db")
    monkeypatch.setattr(settings, "storage_dir", storage)
    monkeypatch.setattr(settings, "tts_enabled", False)
    monkeypatch.setattr(settings, "gemini_api_key", "")
    monkeypatch.setattr(settings, "llm_api_key", "")
    reset_writer()

    for directory in (settings.images_dir, settings.audio_dir):
        directory.mkdir(parents=True, exist_ok=True)

    active = 0
    peak = 0
    original = pipeline._pipeline

    async def counting(*args, **kwargs):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        try:
            await asyncio.sleep(0.15)
            await original(*args, **kwargs)
        finally:
            active -= 1

    monkeypatch.setattr(pipeline, "_pipeline", counting)

    with TestClient(app) as client:
        ids = []
        for _ in range(6):
            response = client.post(
                "/api/v1/analyses",
                files={"image": ("crop.jpg", _jpeg(), "image/jpeg")},
                data={"crop": "rice"},
            )
            assert response.status_code == 202
            ids.append(response.json()["id"])

        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            states = [client.get(f"/api/v1/analyses/{i}").json()["status"] for i in ids]
            if all(state in {"completed", "failed"} for state in states):
                break
            time.sleep(0.1)

        assert all(state == "completed" for state in states), states
        assert peak <= pipeline.CONCURRENCY, f"peak {peak} exceeded {pipeline.CONCURRENCY}"
        assert peak > 1, "no parallelism at all; the semaphore is too tight"


def test_every_burst_upload_gets_its_own_result(tmp_path, monkeypatch):
    """Under contention, no analysis may inherit another's narrative."""
    storage = tmp_path / "storage"
    monkeypatch.setattr(settings, "database_path", storage / "test.db")
    monkeypatch.setattr(settings, "storage_dir", storage)
    monkeypatch.setattr(settings, "tts_enabled", False)
    monkeypatch.setattr(settings, "gemini_api_key", "")
    monkeypatch.setattr(settings, "llm_api_key", "")
    reset_writer()

    for directory in (settings.images_dir, settings.audio_dir):
        directory.mkdir(parents=True, exist_ok=True)

    crops = ["rice", "potato"]

    with TestClient(app) as client:
        ids = {}
        for crop in crops:
            response = client.post(
                "/api/v1/analyses",
                files={"image": ("crop.jpg", _jpeg(), "image/jpeg")},
                data={"crop": crop},
            )
            ids[response.json()["id"]] = crop

        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            states = [client.get(f"/api/v1/analyses/{i}").json()["status"] for i in ids]
            if all(state in {"completed", "failed"} for state in states):
                break
            time.sleep(0.1)

        for analysis_id, crop in ids.items():
            result = client.get(f"/api/v1/analyses/{analysis_id}/result").json()
            summary = client.get(f"/api/v1/analyses/{analysis_id}").json()

            expected = {
                "rice": "ধান",
                "potato": "আলু",
            }[crop]
            assert summary["cropName"] == expected
            assert result["narrative"], analysis_id
            # The narrative must name the crop that was asked for.
            assert summary["cropName"] in result["narrative"]
