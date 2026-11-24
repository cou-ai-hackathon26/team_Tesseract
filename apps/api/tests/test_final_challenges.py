"""Tests for CoU AI Hackathon 2026 Final Day challenges:

1. Voice-named village/plot -> 3-layer geocoding with duplicate disambiguation ->
   Open-Meteo forecast grid + location-aware extension office advice + quick-switch.
2. Strict BARI/BRRI RAG grounding on leaf-photo vs spoken-question crop mismatch.
"""

from __future__ import annotations

import asyncio
import io
import json
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.config import settings
from app.contracts import AnalysisResult, Weather
from app.grounding import geocoder, rag
from app.grounding import weather as weather_module
from app.jobs import pipeline
from app.llm import audio_input
from app.llm.writer import GeminiWriter, reset_writer
from app.main import app
from app.tts.engine import prepare_text
from app.vision import VisionFacts

_DHAKA_TZ = timezone(timedelta(hours=6))
_LATIN_RE = re.compile(r"[A-Za-z]")


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
        time.sleep(0.05)
    raise AssertionError("analysis did not finish")


def _make_48h_payload(*, lat: float = 23.5, lon: float = 91.125) -> dict:
    start = datetime.now(_DHAKA_TZ).replace(minute=0, second=0, microsecond=0)
    times = [(start + timedelta(hours=i)).isoformat(timespec="minutes") for i in range(48)]
    return {
        "latitude": lat,
        "longitude": lon,
        "hourly": {
            "time": times,
            "precipitation_probability": [10] * 48,
            "precipitation": [0.0] * 48,
            "wind_speed_10m": [5.0] * 48,
            "relative_humidity_2m": [72] * 48,
        },
    }


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
    weather_module._cache.clear()
    reset_writer()
    rag.reset_rag_cache()

    for directory in (settings.images_dir, settings.audio_dir):
        directory.mkdir(parents=True, exist_ok=True)

    with TestClient(app) as test_client:
        yield test_client

    weather_module._cache.clear()
    reset_writer()


# ---------------------------------------------------------------------------
# 1. Structured voice & text extraction
# ---------------------------------------------------------------------------


async def test_extract_question_context_gemini_structured_json_and_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from google.genai import errors as genai_errors

    monkeypatch.setattr(settings, "gemini_api_key", "test-key-not-real")
    monkeypatch.setattr(settings, "gemini_model", "gemini-3.8-flash")

    calls: list[dict] = []
    structured_payload = {
        "transcript_bn": "আমি কুমিল্লার বুড়িচং থেকে বলছি, নিচু জমিতে ধানের পাতায় দাগ।",
        "spoken_place_bn": "বুড়িচং",
        "spoken_place_en": "Burichang",
        "spoken_district_en": "Cumilla",
        "spoken_plot_hint_bn": "নিচু জমি",
        "spoken_crop": "rice",
        "spoken_crop_bn": "ধান",
    }

    class StubModels:
        def __init__(self) -> None:
            self.step = 0

        def generate_content(self, **kwargs):
            calls.append(kwargs)
            self.step += 1
            if self.step == 1:
                raise genai_errors.APIError(503, {"error": {"message": "busy"}})
            return SimpleNamespace(text=json.dumps(structured_payload, ensure_ascii=False))

    monkeypatch.setattr(
        audio_input, "_build_client", lambda *_args: SimpleNamespace(models=StubModels())
    )

    ctx = await audio_input.extract_question_context(
        b"webm-audio", "audio/webm", None, settings
    )
    assert [c["model"] for c in calls] == ["gemini-3.8-flash", "gemini-3.7-flash"]
    assert calls[0]["config"].response_mime_type == "application/json"
    assert calls[0]["contents"][1].inline_data.mime_type == "audio/webm"
    assert ctx.transcript_bn == structured_payload["transcript_bn"]
    assert ctx.spoken_place_bn == "বুড়িচং"
    assert ctx.spoken_place_en == "Burichang"
    assert ctx.spoken_district_en == "Cumilla"
    assert ctx.spoken_plot_hint_bn == "নিচু জমি"
    assert ctx.spoken_crop == "rice"
    assert ctx.spoken_crop_bn == "ধান"


async def test_extract_question_context_offline_bangla_keywords() -> None:
    ctx1 = audio_input.extract_question_context_offline(
        "আমি কুমিল্লার বুড়িচং থেকে বলছি, আমার পুকুরের পাশের জমিতে ধানের পাতায় দাগ হয়েছে।"
    )
    assert ctx1.spoken_place_bn == "বুড়িচং"
    assert ctx1.spoken_place_en == "Burichang"
    assert ctx1.spoken_district_en == "Cumilla"
    assert ctx1.spoken_plot_hint_bn == "পুকুরের পাশের জমি"
    assert ctx1.spoken_crop == "rice"
    assert ctx1.spoken_crop_bn == "ধান"

    # Ambiguous place name "Sherpur" without parent district leaves spoken_district_en None.
    ctx2 = audio_input.extract_question_context_offline("শেরপুরে আমার আলু ক্ষেতে মড়ক ধরেছে।")
    assert ctx2.spoken_place_bn == "শেরপুর"
    assert ctx2.spoken_place_en == "Sherpur"
    assert ctx2.spoken_district_en is None
    assert ctx2.spoken_crop == "potato"
    assert ctx2.spoken_crop_bn == "আলু"

    # "আমার" must NOT trigger mango ("আম").
    ctx3 = audio_input.extract_question_context_offline("আমার নিচু জমিতে গাছের পাতা হলুদ হয়ে যাচ্ছে।")
    assert ctx3.spoken_crop == "none"
    assert ctx3.spoken_plot_hint_bn == "নিচু জমি"

    # Other crop ("টমেটো") triggers "other".
    ctx4 = audio_input.extract_question_context_offline("আমার টমেটো গাছে পোকা ধরেছে।")
    assert ctx4.spoken_crop == "other"
    assert ctx4.spoken_crop_bn == "টমেটো"


# ---------------------------------------------------------------------------
# 2. Precise location geocoder with 3-step disambiguation
# ---------------------------------------------------------------------------


def _sherpur_open_meteo_payload() -> dict:
    return {
        "results": [
            {
                "name": "Sherpur",
                "latitude": 25.0205,
                "longitude": 90.0153,
                "country_code": "BD",
                "admin1": "Mymensingh Division",
                "admin2": "Sherpur",
            },
            {
                "name": "Sherpur",
                "latitude": 24.6749,
                "longitude": 89.4170,
                "country_code": "BD",
                "admin1": "Rajshahi Division",
                "admin2": "Bogra",
            },
            {
                "name": "Sherpur",
                "latitude": 24.6272,
                "longitude": 91.6817,
                "country_code": "BD",
                "admin1": "Sylhet Division",
                "admin2": "Moulvibazar",
            },
        ]
    }


async def test_geocoder_disambiguates_duplicate_sherpur_by_spoken_district_and_gps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _sherpur_open_meteo_payload()

    class FakeResp:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return payload

    class FakeClient:
        def __init__(self, **_kwargs) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args) -> None:
            return None

        async def get(self, url: str, params=None) -> FakeResp:
            assert "geocoding-api.open-meteo.com" in url
            return FakeResp()

    monkeypatch.setattr(geocoder.httpx, "AsyncClient", FakeClient)

    # 1. Explicit spoken_district_en="Bogura" selects Bogura Sherpur + returns 3 candidates.
    res_bogura = await geocoder.resolve_precise_location(
        "Sherpur", "শেরপুর", spoken_district_en="Bogura"
    )
    assert res_bogura is not None
    assert res_bogura.district_id == "bogura"
    assert res_bogura.resolved_place == "শেরপুর, বগুড়া"
    assert len(res_bogura.candidates) == 3
    assert {c.district for c in res_bogura.candidates} == {"বগুড়া", "ময়মনসিংহ", "মৌলভীবাজার"}

    # 2. No spoken district, GPS near Mymensingh -> tie-breaks to Sherpur (Mymensingh).
    res_gps_mym = await geocoder.resolve_precise_location(
        "Sherpur", "শেরপুর", None, gps_lat=24.85, gps_lon=90.20
    )
    assert res_gps_mym is not None
    assert res_gps_mym.district_id == "sherpur"
    assert len(res_gps_mym.candidates) == 3

    # 3. No spoken district and no GPS -> tie-breaks by fallback_district_id="bogura".
    res_fallback_bogura = await geocoder.resolve_precise_location(
        "Sherpur", "শেরপুর", None, fallback_district_id="bogura"
    )
    assert res_fallback_bogura is not None
    assert res_fallback_bogura.district_id == "bogura"


async def test_geocoder_falls_back_to_nominatim_then_offline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen_calls: list[tuple[str, dict]] = []

    class FakeResp:
        def __init__(self, data) -> None:
            self._data = data

        def raise_for_status(self) -> None:
            return None

        def json(self):
            return self._data

    class FakeClientNominatim:
        def __init__(self, **kwargs) -> None:
            self._headers = kwargs.get("headers") or {}

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args) -> None:
            return None

        async def get(self, url: str, params=None) -> FakeResp:
            seen_calls.append((url, {"params": params, "headers": self._headers}))
            if "open-meteo" in url:
                return FakeResp({"results": []})
            return FakeResp(
                [
                    {
                        "lat": "23.5494",
                        "lon": "91.1242",
                        "name": "Burichang",
                        "address": {"county": "Cumilla", "state": "Chattogram Division"},
                    }
                ]
            )

    monkeypatch.setattr(geocoder.httpx, "AsyncClient", FakeClientNominatim)
    res = await geocoder.resolve_precise_location("Burichang", "বুড়িচং", "Cumilla")
    assert res is not None
    assert res.source_layer == "nominatim"
    assert res.district_id == "comilla"
    assert res.resolved_place == "বুড়িচং, কুমিল্লা"
    assert seen_calls[1][1]["headers"]["User-Agent"] == "FosholLens-CoU-Hackathon/1.0"
    assert seen_calls[1][1]["params"]["countrycodes"] == "bd"

    # Offline fallback when both HTTP layers fail
    class FailingClient:
        def __init__(self, **_kwargs) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args) -> None:
            return None

        async def get(self, _url: str, params=None):
            raise httpx.ConnectError("offline")

    monkeypatch.setattr(geocoder.httpx, "AsyncClient", FailingClient)
    offline_res = await geocoder.resolve_precise_location("Sherpur", "শেরপুর", "Bogura")
    assert offline_res is not None
    assert offline_res.source_layer == "offline"
    assert offline_res.district_id == "bogura"
    assert len(offline_res.candidates) == 3


# ---------------------------------------------------------------------------
# 3. End-to-end voice location priority, snapped grid, extension office, & /weather
# ---------------------------------------------------------------------------


def test_voice_location_overrides_gps_and_district_with_grid_and_extension_advice(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def fake_resolve(*_args, **_kwargs):
        return geocoder.ResolvedLocation(
            resolved_place="বুড়িচং, কুমিল্লা",
            district_id="comilla",
            district_name="কুমিল্লা",
            latitude=23.5494,
            longitude=91.1242,
            candidates=(
                geocoder.LocationCandidateInfo("বুড়িচং, কুমিল্লা", "কুমিল্লা", 23.5494, 91.1242),
            ),
            source_layer="open-meteo",
        )

    payload = _make_48h_payload(lat=23.5, lon=91.125)

    class FakeResp:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return payload

    class FakeWeatherClient:
        def __init__(self, **_kwargs) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args) -> None:
            return None

        async def get(self, _url: str, params=None) -> FakeResp:
            return FakeResp()

    monkeypatch.setattr("app.jobs.pipeline.resolve_precise_location", fake_resolve)
    monkeypatch.setattr(weather_module.httpx, "AsyncClient", FakeWeatherClient)

    created = client.post(
        "/api/v1/analyses",
        files={"image": ("crop.jpg", _jpeg(), "image/jpeg")},
        data={
            "crop": "rice",
            "district": "dhaka",
            "latitude": "23.8103",
            "longitude": "90.4125",
            "question_text": "আমি কুমিল্লার বুড়িচং থেকে বলছি, পুকুরের পাশের জমিতে ধানের পাতায় দাগ।",
        },
    )
    assert created.status_code == 202
    analysis_id = created.json()["id"]
    summary = _wait(client, analysis_id)
    assert summary["status"] == "completed"

    result = AnalysisResult.model_validate(
        client.get(f"/api/v1/analyses/{analysis_id}/result").json()
    )
    assert result.weather is not None
    assert result.weather.locationSource == "voice"
    assert result.weather.resolvedPlace == "বুড়িচং, কুমিল্লা"
    assert result.weather.gridLatitude == pytest.approx(23.5)
    assert result.weather.gridLongitude == pytest.approx(91.125)
    assert result.weather.extensionOfficeBn is not None
    assert "বুড়িচং, কুমিল্লা" in result.weather.extensionOfficeBn
    assert "উপজেলা কৃষি অফিস" in result.weather.extensionOfficeBn

    assert result.cropMismatch is None
    assert "বুড়িচং" in result.narrative
    assert "পুকুরের পাশের জমি" in result.narrative
    assert "উপজেলা কৃষি অফিস" in result.narrative
    assert not _LATIN_RE.search(result.narrative)
    assert prepare_text(result.narrative)

    # Quick-switch GET /api/v1/weather endpoint
    weather_resp = client.get(
        "/api/v1/weather",
        params={"lat": 23.5494, "lon": 91.1242, "label": "বুড়িচং, কুমিল্লা"},
    )
    assert weather_resp.status_code == 200
    switched = Weather.model_validate(weather_resp.json())
    assert switched.locationSource == "voice"
    assert switched.resolvedPlace == "বুড়িচং, কুমিল্লা"
    assert switched.gridLatitude == pytest.approx(23.5)
    assert switched.gridLongitude == pytest.approx(91.125)
    assert len(switched.hourlyWindows) == 48

    bad_weather = client.get("/api/v1/weather", params={"lat": 95.0, "lon": 91.0})
    assert bad_weather.status_code == 400


# ---------------------------------------------------------------------------
# 4. Strict BARI/BRRI grounding on leaf-photo vs spoken-question crop mismatch
# ---------------------------------------------------------------------------


def test_crop_mismatch_rice_photo_potato_question_strictly_grounded_to_rice(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def fake_classify(self, _path, _crop):
        return VisionFacts(
            status="identified",
            label="rice_blast",
            confidence=0.92,
            crop_id="rice",
            disease_id="rice_blast",
            is_healthy=False,
            model="efficientnet",
        )

    monkeypatch.setattr(settings, "vision_enabled", True)
    monkeypatch.setattr(
        pipeline.get_classifier().__class__, "classify", fake_classify, raising=False
    )

    created = client.post(
        "/api/v1/analyses",
        files={"image": ("rice.jpg", _jpeg(), "image/jpeg")},
        data={
            "crop": "rice",
            "question_text": "আমার আলুর পাতায় মড়ক লেগেছে, কোন ছত্রাকনাশক স্প্রে করব?",
        },
    )
    assert created.status_code == 202
    analysis_id = created.json()["id"]
    _wait(client, analysis_id)

    result = AnalysisResult.model_validate(
        client.get(f"/api/v1/analyses/{analysis_id}/result").json()
    )
    assert result.cropMismatch is not None
    assert result.cropMismatch.detected is True
    assert result.cropMismatch.photoCropBn == "ধান"
    assert result.cropMismatch.questionCropBn == "আলু"
    assert "ধান" in result.cropMismatch.messageBn
    assert "আলু" in result.cropMismatch.messageBn
    assert result.referToOfficer is True

    # RAG sources strictly for photographed rice_blast, never potato.
    assert len(result.ragSources) >= 1
    assert all(src.diseaseId == "rice_blast" for src in result.ragSources)
    assert all("আলু" not in src.sourceTitle for src in result.ragSources)

    # Zero potato-only chemicals leaked into treatment steps or narrative.
    steps_blob = " ".join(result.treatment.steps) if result.treatment else ""
    for potato_chem in ("ম্যানকোজেব", "মেটালাক্সিল", "ক্লোরোথ্যালোনিল"):
        assert potato_chem not in steps_blob
        assert potato_chem not in result.narrative

    assert "ধান" in result.narrative
    assert "আলু" in result.narrative
    assert not _LATIN_RE.search(result.narrative)


def test_crop_mismatch_potato_photo_rice_question_strictly_grounded_to_potato(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def fake_classify(self, _path, _crop):
        return VisionFacts(
            status="identified",
            label="potato_late_blight",
            confidence=0.90,
            crop_id="potato",
            disease_id="potato_late_blight",
            is_healthy=False,
            model="efficientnet",
        )

    monkeypatch.setattr(settings, "vision_enabled", True)
    monkeypatch.setattr(
        pipeline.get_classifier().__class__, "classify", fake_classify, raising=False
    )

    created = client.post(
        "/api/v1/analyses",
        files={"image": ("potato.jpg", _jpeg(), "image/jpeg")},
        data={
            "crop": "potato",
            "question_text": "আমার ধানের জমিতে ব্লাস্ট হয়েছে, ট্রাইসাইক্লাজল দেব কি?",
        },
    )
    assert created.status_code == 202
    analysis_id = created.json()["id"]
    _wait(client, analysis_id)

    result = AnalysisResult.model_validate(
        client.get(f"/api/v1/analyses/{analysis_id}/result").json()
    )
    assert result.cropMismatch is not None
    assert result.cropMismatch.detected is True
    assert result.cropMismatch.photoCropBn == "আলু"
    assert result.cropMismatch.questionCropBn == "ধান"
    assert result.referToOfficer is True
    assert len(result.ragSources) >= 1
    assert all(src.diseaseId == "potato_late_blight" for src in result.ragSources)

    steps_blob = " ".join(result.treatment.steps) if result.treatment else ""
    for rice_chem in ("ট্রাইসাইক্লাজল", "আইসোপ্রোথায়োলেন"):
        assert rice_chem not in steps_blob
        assert rice_chem not in result.narrative


def test_unconfirmed_rice_with_other_crop_question_uses_safe_scouting_only(
    client: TestClient,
) -> None:
    created = client.post(
        "/api/v1/analyses",
        files={"image": ("leaf.jpg", _jpeg(), "image/jpeg")},
        data={"crop": "rice", "question_text": "আমার টমেটো গাছে পাতা কুঁকড়ে যাচ্ছে, কী স্প্রে করব?"},
    )
    assert created.status_code == 202
    analysis_id = created.json()["id"]
    _wait(client, analysis_id)

    result = AnalysisResult.model_validate(
        client.get(f"/api/v1/analyses/{analysis_id}/result").json()
    )
    assert result.cropMismatch is not None
    assert result.cropMismatch.detected is True
    assert result.cropMismatch.photoCropBn == "ধান"
    assert result.cropMismatch.questionCropBn == "টমেটো"
    assert result.referToOfficer is True
    assert [s.chunkId for s in result.ragSources] == ["brri_rice_healthy_02"]
    assert not _LATIN_RE.search(result.narrative)


async def test_gemini_writer_rejects_cross_crop_chemical_during_mismatch() -> None:
    from app.grounding import knowledge_base

    disease = knowledge_base.get_disease("rice_blast")
    assert disease is not None
    bundle = rag.retrieve_grounding_bundle("rice_blast", "rice")

    facts = pipeline._build_facts(
        disease,
        weather=None,
        vision=VisionFacts(
            status="identified",
            label="rice_blast",
            confidence=0.91,
            crop_id="rice",
            disease_id="rice_blast",
            is_healthy=False,
            model="efficientnet",
        ),
        confirmed=True,
        grounding=bundle,
        farmer_question="আমার আলু গাছে কোন ওষুধ দেব?",
        crop_mismatch={
            "detected": True,
            "photoCropBn": "ধান",
            "questionCropBn": "আলু",
            "messageBn": "অমিল",
        },
    )

    bad_cross_crop = (
        "ছবিতে ধান হলেও আপনার আলু ক্ষেতের জন্য প্রতি লিটার পানিতে ম্যানকোজেব স্প্রে করতে পারেন। "
        "তবে প্রয়োগের আগে গ্লাভস ও মাস্ক পরুন এবং স্থানীয় কৃষি কর্মকর্তার পরামর্শ নিন।"
    )
    good_grounded = (
        "ছবিতে ধান পাতা দেখা যাচ্ছে কিন্তু প্রশ্নে আলু ফসলের কথা বলেছেন। "
        "তাই কেবল ছবির ধানের ব্লাস্টের জন্য ট্রাইসাইক্লাজল ব্যবহার করুন এবং গ্লাভস ও মাস্ক পরুন। "
        "সঠিক ফসল নিশ্চিত করতে স্থানীয় কৃষি কর্মকর্তার পরামর্শ নিন।"
    )

    prompts: list[str] = []

    class StubModels:
        def __init__(self) -> None:
            self.payloads = [bad_cross_crop, good_grounded]

        def generate_content(self, **kwargs):
            prompts.append(kwargs["contents"])
            return SimpleNamespace(
                text=json.dumps({"spoken": self.payloads.pop(0)}, ensure_ascii=False)
            )

    writer = GeminiWriter.__new__(GeminiWriter)
    writer._semaphore = asyncio.Semaphore(3)
    writer._client = SimpleNamespace(models=StubModels())

    text, generated_by = await writer.write(facts)
    assert generated_by == "llm"
    assert "ট্রাইসাইক্লাজল" in text
    assert "ম্যানকোজেব" not in text
    assert len(prompts) == 2
    assert "ফসলের অমিল" in prompts[1]


# ---------------------------------------------------------------------------
# 5. Cross-crop unmasked photo detection (e.g. Potato photo + Rice selection)
# ---------------------------------------------------------------------------


def _probs(**weights: float):
    import numpy as np

    from app.vision.models import EFFICIENTNET

    vector = np.zeros(len(EFFICIENTNET["labels"]), dtype=np.float32)
    for label, weight in weights.items():
        vector[EFFICIENTNET["labels"].index(label)] = weight
    return vector


def test_classifier_decide_exposes_unmasked_crop_override_without_breaking_scoped_status() -> None:
    import numpy as np

    from app.vision.classifier import _decide
    from app.vision.models import EFFICIENTNET

    # Potato leaf uploaded while crop="rice" is selected.
    potato_under_rice = _decide(
        _probs(potato_late_blight=0.92, potato_early_blight=0.05, rice_blast=0.01),
        "rice",
        EFFICIENTNET,
    )
    assert potato_under_rice["status"] == "undetermined"
    assert potato_under_rice["raw_top_label"] == "potato_late_blight"
    assert potato_under_rice["raw_top_confidence"] == pytest.approx(0.92)
    assert potato_under_rice["inferred_photo_crop"] == "potato"
    assert potato_under_rice["effective_crop"] == "potato"
    assert potato_under_rice["crop_override_detected"] is True
    assert potato_under_rice["override_disease_id"] == "potato_late_blight"
    assert potato_under_rice["override_is_healthy"] is False
    assert potato_under_rice["override_confirmed"] is True

    # Rice leaf uploaded while crop="potato" is selected.
    rice_under_potato = _decide(
        _probs(rice_blast=0.88, rice_healthy=0.05, potato_healthy=0.02),
        "potato",
        EFFICIENTNET,
    )
    assert rice_under_potato["status"] == "undetermined"
    assert rice_under_potato["inferred_photo_crop"] == "rice"
    assert rice_under_potato["crop_override_detected"] is True
    assert rice_under_potato["override_disease_id"] == "rice_blast"
    assert rice_under_potato["override_confirmed"] is True

    # Uniform noise (0.125 across 8 classes -> 5 rice classes sum to 0.625) must NOT infer rice.
    uniform = np.full(len(EFFICIENTNET["labels"]), 1.0 / len(EFFICIENTNET["labels"]))
    noise = _decide(uniform, "potato", EFFICIENTNET)
    assert noise["inferred_photo_crop"] is None
    assert noise["crop_override_detected"] is False


def test_cross_crop_potato_photo_under_rice_selection_and_rice_question(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app.vision.classifier import _decide
    from app.vision.models import EFFICIENTNET

    potato_under_rice = _decide(
        _probs(potato_late_blight=0.92, potato_early_blight=0.04, rice_blast=0.01),
        "rice",
        EFFICIENTNET,
    )

    async def fake_classify(self, _path, _crop):
        return potato_under_rice

    monkeypatch.setattr(settings, "vision_enabled", True)
    monkeypatch.setattr(
        pipeline.get_classifier().__class__, "classify", fake_classify, raising=False
    )

    created = client.post(
        "/api/v1/analyses",
        files={"image": ("potato.jpg", _jpeg(), "image/jpeg")},
        data={
            "crop": "rice",
            "question_text": "আমার ধানের পাতায় দাগ হয়েছে, কোন ওষুধ স্প্রে করব?",
        },
    )
    assert created.status_code == 202
    analysis_id = created.json()["id"]
    _wait(client, analysis_id)

    result = AnalysisResult.model_validate(
        client.get(f"/api/v1/analyses/{analysis_id}/result").json()
    )
    assert result.uncertainty.isUncertain is False
    assert result.cropName == "আলু"
    assert result.diagnosis is not None
    assert result.diagnosis.diseaseLabel == "আলুর নাবি ধসা রোগ"
    assert result.diagnosis.confidence == pytest.approx(0.92)
    assert result.cropMismatch is not None
    assert result.cropMismatch.detected is True
    assert result.cropMismatch.photoCropBn == "আলু"
    assert result.cropMismatch.questionCropBn == "ধান"
    assert "আলু" in result.cropMismatch.messageBn
    assert "ধান" in result.cropMismatch.messageBn
    assert result.referToOfficer is True

    # Strictly BARI Potato RAG chunks (zero BRRI Rice chunks).
    assert len(result.ragSources) >= 1
    assert all(src.diseaseId == "potato_late_blight" for src in result.ragSources)
    assert all("BARI" in src.sourceTitle for src in result.ragSources)
    assert all("BRRI" not in src.sourceTitle for src in result.ragSources)

    # Zero rice chemicals in treatment steps or narrative, and TTS-safe pure Bangla.
    steps_blob = " ".join(result.treatment.steps) if result.treatment else ""
    for rice_chem in ("ট্রাইসাইক্লাজল", "আইসোপ্রোথায়োলেন", "প্রোপিকোনাজল"):
        assert rice_chem not in steps_blob
        assert rice_chem not in result.narrative
    assert not _LATIN_RE.search(result.narrative)
    assert prepare_text(result.narrative)


def test_cross_crop_potato_photo_under_rice_dropdown_without_question(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app.vision.classifier import _decide
    from app.vision.models import EFFICIENTNET

    potato_under_rice = _decide(
        _probs(potato_early_blight=0.89, potato_late_blight=0.05, rice_healthy=0.01),
        "rice",
        EFFICIENTNET,
    )

    async def fake_classify(self, _path, _crop):
        return potato_under_rice

    monkeypatch.setattr(settings, "vision_enabled", True)
    monkeypatch.setattr(
        pipeline.get_classifier().__class__, "classify", fake_classify, raising=False
    )

    created = client.post(
        "/api/v1/analyses",
        files={"image": ("potato_early.jpg", _jpeg(), "image/jpeg")},
        data={"crop": "rice"},
    )
    assert created.status_code == 202
    analysis_id = created.json()["id"]
    _wait(client, analysis_id)

    result = AnalysisResult.model_validate(
        client.get(f"/api/v1/analyses/{analysis_id}/result").json()
    )
    assert result.uncertainty.isUncertain is False
    assert result.cropName == "আলু"
    assert result.diagnosis is not None
    assert result.diagnosis.diseaseLabel == "আলুর আগাম ধসা রোগ"
    assert result.cropMismatch is not None
    assert result.cropMismatch.detected is True
    assert result.cropMismatch.photoCropBn == "আলু"
    assert result.cropMismatch.questionCropBn == "ধান"
    assert result.referToOfficer is True
    assert len(result.ragSources) >= 1
    assert all(src.diseaseId == "potato_early_blight" for src in result.ragSources)
    assert not _LATIN_RE.search(result.narrative)

