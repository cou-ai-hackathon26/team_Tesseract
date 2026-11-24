"""End-to-end pipeline behaviour over the real HTTP surface.

Uses `TestClient`, so this exercises routing, storage, the job runner, and the
contract models together. Weather is stubbed because a test must not depend on a
third party being reachable.
"""

from __future__ import annotations

import io
import time

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.config import settings
from app.contracts import AnalysisResult
from app.grounding.weather import WeatherFacts
from app.llm.writer import GeminiWriter, reset_writer
from app.main import app


@pytest.fixture()
def client(tmp_path, monkeypatch):
    """A client with an isolated database and storage, and no TTS.

    Vision is disabled here so the suite is deterministic and offline: the
    classifier downloads weights on first use, and a unit test must not depend
    on a model host. `test_vision.py` covers the decision logic directly.
    """
    storage = tmp_path / "storage"
    monkeypatch.setattr(settings, "database_path", storage / "test.db")
    monkeypatch.setattr(settings, "storage_dir", storage)
    monkeypatch.setattr(settings, "tts_enabled", False)
    monkeypatch.setattr(settings, "vision_enabled", False)
    monkeypatch.setattr(settings, "llm_provider", "auto")
    monkeypatch.setattr(settings, "gemini_api_key", "")
    monkeypatch.setattr(settings, "llm_api_key", "")
    reset_writer()

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


def test_health_reports_template_fallback(client: TestClient) -> None:
    body = client.get("/api/v1/health").json()
    assert body["llm"] == "unconfigured"
    assert body["tts"] in {"ready", "unavailable"}
    # The fixture disables vision, so "unavailable" is the honest answer; a
    # "ready" here would mean the field is reporting the env flag, not reality.
    assert body["vision"] == "unavailable"
    assert body["db"] == "ok"

    root_health = client.get("/health")
    assert root_health.status_code == 200
    assert root_health.json() == body


def test_districts_covers_all_64(client: TestClient) -> None:
    districts = client.get("/api/v1/districts").json()
    assert len(districts) == 64
    assert all(set(d) >= {"id", "name", "division", "latitude", "longitude"} for d in districts)


def test_full_pipeline_without_weather(client: TestClient) -> None:
    created = client.post(
        "/api/v1/analyses",
        files={"image": ("crop.jpg", _jpeg(), "image/jpeg")},
        data={"crop": "rice"},
    )
    assert created.status_code == 202
    analysis_id = created.json()["id"]

    summary = _wait(client, analysis_id)
    assert summary["status"] == "completed"
    assert summary["stage"] == "done"
    assert summary["diseaseLabel"] == "ধানের ব্লাস্ট রোগ"
    assert summary["cropName"] == "ধান"

    result = client.get(f"/api/v1/analyses/{analysis_id}/result").json()
    assert result["stage"] == "done"
    assert result["generatedBy"] == "template"
    assert result["weather"] is None
    assert result["audio"]["status"] == "none"
    assert result["diagnosis"]["diseaseLabel"] == "ধানের ব্লাস্ট রোগ"
    assert result["treatment"]["source"]
    assert result["prevention"]
    # A farmer must never get a confident-sounding machine answer.
    assert result["needsProfessionalVerification"] is True
    assert result["narrative"]
    assert "আনুমানিক পরামর্শ" not in result["narrative"]


def test_without_vision_the_result_says_so(client: TestClient) -> None:
    """No classifier means no diagnosis, and no invented confidence.

    This is the honest path: real curated treatment for the crop, but nothing
    claiming the photo was recognised.
    """
    created = client.post(
        "/api/v1/analyses",
        files={"image": ("crop.jpg", _jpeg(), "image/jpeg")},
        data={"crop": "rice"},
    )
    analysis_id = created.json()["id"]
    _wait(client, analysis_id)

    result = client.get(f"/api/v1/analyses/{analysis_id}/result").json()
    assert result["uncertainty"]["isUncertain"] is True
    assert result["diagnosis"]["confidence"] is None
    assert result["diagnosis"]["confidenceBand"] == "low"
    # Symptoms describe a disease; nothing saw one.
    assert result["symptoms"] == []
    # The treatment is still real, curated advice for the crop.
    assert result["treatment"]["steps"]


def test_identified_leaf_uses_the_classifier_result(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When the classifier *does* identify a leaf, its verdict drives the result.

    The model is stubbed rather than downloaded: this asserts the wiring between
    vision, the knowledge base, and the contract, which is the part that can
    break. Accuracy of the model itself is not something a unit test can assert.
    """
    from app.jobs import pipeline
    from app.vision import VisionFacts

    async def fake_classify(self, image_path, crop_id):
        return VisionFacts(
            status="identified",
            label="Rice___Leaf_Blast",
            confidence=0.91,
            crop_id="rice",
            disease_id="rice_blast",
            is_healthy=False,
        )

    monkeypatch.setattr(settings, "vision_enabled", True)
    monkeypatch.setattr(
        pipeline.get_classifier().__class__, "classify", fake_classify, raising=False
    )

    created = client.post(
        "/api/v1/analyses",
        files={"image": ("crop.jpg", _jpeg(), "image/jpeg")},
        data={"crop": "rice"},
    )
    analysis_id = created.json()["id"]
    _wait(client, analysis_id)

    result = client.get(f"/api/v1/analyses/{analysis_id}/result").json()
    assert result["uncertainty"]["isUncertain"] is False
    assert result["diagnosis"]["confidence"] == pytest.approx(0.91)
    assert result["diagnosis"]["confidenceBand"] == "high"
    assert result["diagnosis"]["diseaseLabel"] == "ধানের ব্লাস্ট রোগ"
    assert result["symptoms"]


def test_history_row_carries_display_fields(client: TestClient) -> None:
    created = client.post(
        "/api/v1/analyses",
        files={"image": ("crop.jpg", _jpeg(), "image/jpeg")},
        data={"crop": "potato"},
    )
    analysis_id = created.json()["id"]
    _wait(client, analysis_id)

    rows = client.get("/api/v1/analyses").json()
    row = next(r for r in rows if r["id"] == analysis_id)
    assert row["cropName"] == "আলু"
    assert row["diseaseLabel"] == "আলুর নাবি ধসা রোগ"
    assert row["thumbnailUrl"].endswith("/image")


def test_weather_flows_into_result(client: TestClient, monkeypatch) -> None:
    async def fake_weather(**_kwargs):
        return WeatherFacts(
            district_id="dhaka",
            district_name="ঢাকা",
            rain_probability=97,
            wind_speed_max=9.6,
            humidity=95,
            spray_window_ok=False,
            reason="আগামী ১২ ঘণ্টায় বৃষ্টির সম্ভাবনা ৯৭ শতাংশ",
        )

    monkeypatch.setattr("app.jobs.pipeline.fetch_weather", fake_weather)

    created = client.post(
        "/api/v1/analyses",
        files={"image": ("crop.jpg", _jpeg(), "image/jpeg")},
        data={"crop": "rice", "district": "dhaka"},
    )
    analysis_id = created.json()["id"]
    _wait(client, analysis_id)

    result = client.get(f"/api/v1/analyses/{analysis_id}/result").json()
    assert result["weather"]["sprayWindowOk"] is False
    assert result["weather"]["rainProbabilityNext12h"] == 97
    assert result["district"] == "ঢাকা"
    # The spoken advice must agree with the code-computed verdict.
    assert "স্প্রে করবেন না" in result["narrative"]

    summary = client.get(f"/api/v1/analyses/{analysis_id}").json()
    assert summary["district"] == "ঢাকা"


def test_image_is_served_and_reencoded(client: TestClient) -> None:
    created = client.post(
        "/api/v1/analyses",
        files={"image": ("crop.jpg", _jpeg(), "image/jpeg")},
        data={"crop": "rice"},
    )
    analysis_id = created.json()["id"]
    _wait(client, analysis_id)

    response = client.get(f"/api/v1/analyses/{analysis_id}/image")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"

    with Image.open(io.BytesIO(response.content)) as image:
        # EXIF is dropped: phones write GPS into photos.
        assert not image.getexif()


def test_result_is_409_until_ready(client: TestClient, monkeypatch) -> None:
    """The client polls; a premature read must not look like a broken analysis."""

    async def never_finishes(*_args, **_kwargs):
        import asyncio

        await asyncio.sleep(30)

    monkeypatch.setattr("app.jobs.pipeline._pipeline", never_finishes)

    created = client.post(
        "/api/v1/analyses",
        files={"image": ("crop.jpg", _jpeg(), "image/jpeg")},
        data={"crop": "rice"},
    )
    analysis_id = created.json()["id"]

    response = client.get(f"/api/v1/analyses/{analysis_id}/result")
    assert response.status_code == 409


def test_unknown_district_is_rejected(client: TestClient) -> None:
    response = client.post(
        "/api/v1/analyses",
        files={"image": ("crop.jpg", _jpeg(), "image/jpeg")},
        data={"crop": "rice", "district": "atlantis"},
    )
    assert response.status_code == 400


@pytest.mark.parametrize("blank_district", ["", "   "])
def test_empty_district_is_normalized_to_none(client: TestClient, blank_district: str) -> None:
    response = client.post(
        "/api/v1/analyses",
        files={"image": ("crop.jpg", _jpeg(), "image/jpeg")},
        data={"crop": "rice", "district": blank_district},
    )
    assert response.status_code == 202
    analysis_id = response.json()["id"]
    summary = _wait(client, analysis_id)
    assert summary["status"] == "completed"
    assert summary["district"] is None

    result = client.get(f"/api/v1/analyses/{analysis_id}/result").json()
    assert result["district"] is None
    assert result["weather"] is None


def test_unreadable_image_is_rejected(client: TestClient) -> None:
    response = client.post(
        "/api/v1/analyses",
        files={"image": ("crop.jpg", b"not an image", "image/jpeg")},
        data={"crop": "rice"},
    )
    assert response.status_code == 400


def test_missing_analysis_is_404(client: TestClient) -> None:
    assert client.get("/api/v1/analyses/00000000-0000-0000-0000-000000000000").status_code == 404


def test_delete_removes_row_and_image(client: TestClient) -> None:
    created = client.post(
        "/api/v1/analyses",
        files={"image": ("crop.jpg", _jpeg(), "image/jpeg")},
        data={"crop": "rice"},
    )
    analysis_id = created.json()["id"]
    _wait(client, analysis_id)

    assert client.delete(f"/api/v1/analyses/{analysis_id}").status_code == 204
    assert client.get(f"/api/v1/analyses/{analysis_id}").status_code == 404
    assert client.get(f"/api/v1/analyses/{analysis_id}/image").status_code == 404


def test_stage_is_mirrored_into_stored_result(client: TestClient) -> None:
    """A result still claiming `speaking` would keep the client polling forever."""
    created = client.post(
        "/api/v1/analyses",
        files={"image": ("crop.jpg", _jpeg(), "image/jpeg")},
        data={"crop": "rice"},
    )
    analysis_id = created.json()["id"]
    _wait(client, analysis_id)

    result = client.get(f"/api/v1/analyses/{analysis_id}/result").json()
    assert result["stage"] == "done"


def test_gemini_advisory_pipeline_conforms_to_contract(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When Gemini Flash is configured and succeeds, the result uses generatedBy='llm'."""
    import asyncio
    import json
    from types import SimpleNamespace

    from app.jobs import pipeline
    from app.vision import VisionFacts

    gemini_spoken = (
        "আপনার ধান গাছে ব্লাস্ট রোগের লক্ষণ দেখা যাচ্ছে। "
        "প্রাথমিকভাবে ট্রাইসাইক্লাজল স্প্রে করার আগে স্থানীয় কৃষি কর্মকর্তার পরামর্শ নিন। "
        "স্প্রে করার সময় গ্লাভস ও মাস্ক পরুন এবং কাজ শেষে হাত ধুয়ে নিন।"
    )

    async def fake_classify(self, image_path, crop_id):
        return VisionFacts(
            status="identified",
            label="rice_blast",
            confidence=0.91,
            crop_id="rice",
            disease_id="rice_blast",
            is_healthy=False,
            model="efficientnet",
        )

    class StubModels:
        def generate_content(self, **_kwargs):
            return SimpleNamespace(text=json.dumps({"spoken": gemini_spoken}, ensure_ascii=False))

    stub_writer = GeminiWriter.__new__(GeminiWriter)
    stub_writer._semaphore = asyncio.Semaphore(3)
    stub_writer._client = SimpleNamespace(models=StubModels())

    monkeypatch.setattr(settings, "vision_enabled", True)
    monkeypatch.setattr(
        pipeline.get_classifier().__class__, "classify", fake_classify, raising=False
    )
    monkeypatch.setattr(settings, "gemini_api_key", "test-key-not-real")
    monkeypatch.setattr("app.llm.writer.GeminiWriter", lambda: stub_writer)
    reset_writer()

    health = client.get("/api/v1/health").json()
    assert health["llm"] == "ready"
    assert health["status"] == "ok"

    created = client.post(
        "/api/v1/analyses",
        files={"image": ("crop.jpg", _jpeg(), "image/jpeg")},
        data={"crop": "rice"},
    )
    assert created.status_code == 202
    analysis_id = created.json()["id"]
    summary = _wait(client, analysis_id)
    assert summary["status"] == "completed"

    payload = client.get(f"/api/v1/analyses/{analysis_id}/result").json()
    validated = AnalysisResult.model_validate(payload)
    assert validated.generatedBy == "llm"
    assert "ট্রাইসাইক্লাজল" in validated.narrative
    assert "কৃষি কর্মকর্তার পরামর্শ" in validated.narrative


def test_gemini_failure_falls_back_to_template_conforming_to_contract(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If Gemini fails at runtime, the pipeline degrades cleanly to template fallback."""
    from types import SimpleNamespace

    class FailingModels:
        def generate_content(self, **_kwargs):
            raise RuntimeError("upstream unavailable")

    stub_writer = GeminiWriter.__new__(GeminiWriter)
    import asyncio

    stub_writer._semaphore = asyncio.Semaphore(3)
    stub_writer._client = SimpleNamespace(models=FailingModels())

    monkeypatch.setattr(settings, "gemini_api_key", "test-key-not-real")
    monkeypatch.setattr("app.llm.writer.GeminiWriter", lambda: stub_writer)
    reset_writer()

    created = client.post(
        "/api/v1/analyses",
        files={"image": ("crop.jpg", _jpeg(), "image/jpeg")},
        data={"crop": "rice"},
    )
    assert created.status_code == 202
    analysis_id = created.json()["id"]
    summary = _wait(client, analysis_id)
    assert summary["status"] == "completed"

    payload = client.get(f"/api/v1/analyses/{analysis_id}/result").json()
    validated = AnalysisResult.model_validate(payload)
    assert validated.generatedBy == "template"
    assert validated.narrative
    assert "আনুমানিক পরামর্শ" not in validated.narrative
