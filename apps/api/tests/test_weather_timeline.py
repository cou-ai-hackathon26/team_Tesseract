"""Tests for the 48-hour color-coded spray timeline and safe-window summary."""

from __future__ import annotations

import io
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.config import settings
from app.contracts import AnalysisResult, Weather
from app.grounding import knowledge_base, rag
from app.grounding.weather import classify
from app.jobs import pipeline
from app.llm.writer import TemplateWriter, reset_writer
from app.main import app
from app.vision import VisionFacts

_DHAKA_TZ = timezone(timedelta(hours=6))
_LATIN_RE = re.compile(r"[A-Za-z]")


def _make_48h_payload(
    *,
    start: datetime | None = None,
    hours: int = 48,
    rain_prob: list[int] | None = None,
    rain_mm: list[float] | None = None,
    wind: list[float] | None = None,
    humidity: list[int] | None = None,
) -> dict:
    start_dt = start or datetime(2026, 11, 22, 6, 0, tzinfo=_DHAKA_TZ)
    times = [(start_dt + timedelta(hours=i)).isoformat(timespec="minutes") for i in range(hours)]
    return {
        "hourly": {
            "time": times,
            "precipitation_probability": rain_prob if rain_prob is not None else [10] * hours,
            "precipitation": rain_mm if rain_mm is not None else [0.0] * hours,
            "wind_speed_10m": wind if wind is not None else [5.0] * hours,
            "relative_humidity_2m": humidity if humidity is not None else [70] * hours,
        }
    }


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


def test_classify_produces_48_windows_validating_contract() -> None:
    start = datetime(2026, 11, 22, 6, 0, tzinfo=_DHAKA_TZ)
    facts = classify(_make_48h_payload(start=start), "dhaka", "ঢাকা", now=start)
    assert len(facts.hourly_windows) == 48
    assert all(w.safety == "green" for w in facts.hourly_windows)
    assert facts.safest_window_summary_bn
    assert not _LATIN_RE.search(facts.safest_window_summary_bn)

    contract = Weather(
        district=facts.district_name,
        rainProbabilityNext12h=facts.rain_probability,
        windSpeedMax12h=facts.wind_speed_max,
        humidity=facts.humidity,
        sprayWindowOk=facts.spray_window_ok,
        reason=facts.reason,
        hourlyWindows=[w.as_contract_dict() for w in facts.hourly_windows],
        safestWindowSummaryBn=facts.safest_window_summary_bn,
    )
    assert len(contract.hourlyWindows) == 48


def test_deterministic_green_amber_red_thresholds() -> None:
    probs = [10] * 48
    mms = [0.0] * 48
    winds = [5.0] * 48

    # Isolated amber triggers well apart from any rain red hours.
    probs[10] = 30
    mms[16] = 0.2
    winds[22] = 10.0

    # Isolated high wind triggers red without wash-off lookahead bleeding backward.
    winds[30] = 15.0

    start = datetime(2026, 11, 22, 6, 0, tzinfo=_DHAKA_TZ)
    facts = classify(
        _make_48h_payload(start=start, rain_prob=probs, rain_mm=mms, wind=winds),
        "bogura",
        "বগুড়া",
        now=start,
    )
    windows = facts.hourly_windows

    assert windows[0].safety == "green"
    assert windows[10].safety == "amber"
    assert windows[16].safety == "amber"
    assert windows[22].safety == "amber"
    assert windows[29].safety == "green"  # wind does NOT trigger wash-off lookahead
    assert windows[30].safety == "red"


def test_three_hour_washoff_lookahead_marks_prior_hours_red() -> None:
    probs = [5] * 48
    mms = [0.0] * 48

    # Heavy rain at hour 10 (prob >= 60) and hour 25 (mm >= 1.0).
    probs[10] = 65
    mms[25] = 1.2

    start = datetime(2026, 11, 22, 6, 0, tzinfo=_DHAKA_TZ)
    facts = classify(
        _make_48h_payload(start=start, rain_prob=probs, rain_mm=mms),
        "sylhet",
        "সিলেট",
        now=start,
    )
    windows = facts.hourly_windows

    # Hour 6 is 4h before rain at hour 10 -> stays green.
    assert windows[6].safety == "green"
    # Hours 7, 8, 9 are within 3h before hour 10 -> marked red due to wash-off risk.
    assert windows[7].safety == "red"
    assert windows[8].safety == "red"
    assert windows[9].safety == "red"
    assert windows[10].safety == "red"

    # Hour 21 is 4h before rain_mm=1.2 at hour 25 -> green; 22..24 -> red.
    assert windows[21].safety == "green"
    assert windows[22].safety == "red"
    assert windows[23].safety == "red"
    assert windows[24].safety == "red"
    assert windows[25].safety == "red"


def test_safest_window_prefers_daylight_block_and_contains_no_latin() -> None:
    # Start at 00:00: make 00:00-05:59 green (night), 06:00-14:59 rainy (red),
    # and 15:00-20:59 dry so 15:00-17:59 (hours 15, 16, 17) are genuinely green
    # (18:00-20:59 are within 3h of 21:00 rain, so lookahead marks 18..20 red).
    start = datetime(2026, 11, 22, 0, 0, tzinfo=_DHAKA_TZ)
    probs = [80] * 48
    for h in range(0, 6):
        probs[h] = 5
    for h in range(15, 21):
        probs[h] = 10

    facts = classify(_make_48h_payload(start=start, rain_prob=probs), "dhaka", "ঢাকা", now=start)
    summary = facts.safest_window_summary_bn
    assert summary is not None
    assert not _LATIN_RE.search(summary)
    assert "আজ" in summary
    assert "বিকেল ৩টা থেকে ৬টা" in summary

    # When today is all rainy and tomorrow 06:00-09:59 (indices 30..33) are green
    # (with 34..36 dry buffer before rain resumes at 37):
    probs_tomorrow = [80] * 48
    for h in range(30, 37):
        probs_tomorrow[h] = 10
    facts_tomorrow = classify(
        _make_48h_payload(start=start, rain_prob=probs_tomorrow), "dhaka", "ঢাকা", now=start
    )
    assert facts_tomorrow.safest_window_summary_bn is not None
    assert not _LATIN_RE.search(facts_tomorrow.safest_window_summary_bn)
    assert "আগামীকাল সকাল ৬টা থেকে ১০টা" in facts_tomorrow.safest_window_summary_bn

    # When no daylight green exists across 48h, returns pure Bangla wait note:
    facts_none = classify(
        _make_48h_payload(start=start, rain_prob=[80] * 48), "dhaka", "ঢাকা", now=start
    )
    assert facts_none.safest_window_summary_bn is not None
    assert not _LATIN_RE.search(facts_none.safest_window_summary_bn)
    assert "নিরাপদ স্প্রে সময় নেই" in facts_none.safest_window_summary_bn


def test_pipeline_passes_safest_window_only_when_spraying_is_applicable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    start = datetime(2026, 11, 22, 6, 0, tzinfo=_DHAKA_TZ)
    # First 12h rainy (blocks sprayWindowOk), tomorrow 06:00-09:59 (indices 24..27) green.
    probs = [85] * 48
    for idx in range(24, 28):
        probs[idx] = 5
    weather_facts = classify(
        _make_48h_payload(start=start, rain_prob=probs), "dhaka", "ঢাকা", now=start
    )
    assert weather_facts.spray_window_ok is False
    assert weather_facts.safest_window_summary_bn

    # Check _build_facts gating directly:
    blast = knowledge_base.get_disease("rice_blast")
    blb = knowledge_base.get_disease("rice_bacterial_blight")
    healthy = knowledge_base.get_disease("potato_healthy")
    assert blast and blb and healthy

    blast_bundle = rag.retrieve_grounding_bundle("rice_blast", "rice")
    blb_bundle = rag.retrieve_grounding_bundle("rice_bacterial_leaf_blight", "rice")
    healthy_bundle = rag.retrieve_grounding_bundle("potato_healthy", "potato")

    blast_facts = pipeline._build_facts(
        blast,
        weather_facts,
        VisionFacts(
            status="identified",
            label="rice_blast",
            confidence=0.9,
            crop_id="rice",
            disease_id="rice_blast",
            is_healthy=False,
            model="efficientnet",
        ),
        confirmed=True,
        grounding=blast_bundle,
    )
    assert (
        blast_facts["weather"]["safest_window_summary_bn"]
        == weather_facts.safest_window_summary_bn
    )

    blb_facts = pipeline._build_facts(
        blb,
        weather_facts,
        VisionFacts(
            status="identified",
            label="rice_bacterial_leaf_blight",
            confidence=0.9,
            crop_id="rice",
            disease_id="rice_bacterial_blight",
            is_healthy=False,
            model="efficientnet",
        ),
        confirmed=True,
        grounding=blb_bundle,
    )
    assert blb_facts["weather"]["safest_window_summary_bn"] is None

    healthy_facts = pipeline._build_facts(
        healthy,
        weather_facts,
        VisionFacts(
            status="identified",
            label="potato_healthy",
            confidence=0.9,
            crop_id="potato",
            disease_id="potato_healthy",
            is_healthy=True,
            model="efficientnet",
        ),
        confirmed=True,
        grounding=healthy_bundle,
    )
    assert healthy_facts["weather"]["safest_window_summary_bn"] is None

    # End-to-end API test for rice_blast surfaces 48h timeline & safest window in narrative.
    storage = tmp_path / "storage"
    monkeypatch.setattr(settings, "database_path", storage / "test.db")
    monkeypatch.setattr(settings, "storage_dir", storage)
    monkeypatch.setattr(settings, "tts_enabled", False)
    monkeypatch.setattr(settings, "vision_enabled", True)
    monkeypatch.setattr(settings, "gemini_api_key", "")
    monkeypatch.setattr(settings, "llm_api_key", "")
    reset_writer()
    for directory in (settings.images_dir, settings.audio_dir):
        directory.mkdir(parents=True, exist_ok=True)

    async def fake_classify(self, _path, _crop):
        return VisionFacts(
            status="identified",
            label="rice_blast",
            confidence=0.9,
            crop_id="rice",
            disease_id="rice_blast",
            is_healthy=False,
            model="efficientnet",
        )

    async def fake_weather(**_kwargs):
        return weather_facts

    monkeypatch.setattr(
        pipeline.get_classifier().__class__, "classify", fake_classify, raising=False
    )
    monkeypatch.setattr("app.jobs.pipeline.fetch_weather", fake_weather)

    with TestClient(app) as client:
        created = client.post(
            "/api/v1/analyses",
            files={"image": ("crop.jpg", _jpeg(), "image/jpeg")},
            data={"crop": "rice", "district": "dhaka"},
        )
        assert created.status_code == 202
        analysis_id = created.json()["id"]
        _wait(client, analysis_id)

        payload = client.get(f"/api/v1/analyses/{analysis_id}/result").json()
        result = AnalysisResult.model_validate(payload)
        assert result.weather is not None
        assert len(result.weather.hourlyWindows) == 48
        assert result.weather.safestWindowSummaryBn == weather_facts.safest_window_summary_bn
        assert weather_facts.safest_window_summary_bn in result.narrative

    # TemplateWriter must never suggest spraying for healthy or cultural_only.
    healthy_narrative = TemplateWriter().compose(healthy_facts)
    assert "সবচেয়ে নিরাপদ স্প্রে সময়" not in healthy_narrative


def test_classify_excludes_past_hours_and_never_recommends_past_window() -> None:
    """With forecast_days=3 at 22:00 Dhaka time, past hours 00..21 are excluded."""
    day_start = datetime(2026, 11, 22, 0, 0)  # naive ISO as returned by Open-Meteo
    hours = 72
    probs = [80] * hours
    # Make today morning 06:00-11:59 green (in the past once now=22:00).
    for h in range(6, 12):
        probs[h] = 5
    # Make tomorrow 07:00-10:59 green (indices 31..34) with 35..37 dry lookahead buffer.
    for h in range(31, 38):
        probs[h] = 5

    times = [(day_start + timedelta(hours=i)).strftime("%Y-%m-%dT%H:%M") for i in range(hours)]
    payload = {
        "hourly": {
            "time": times,
            "precipitation_probability": probs,
            "precipitation": [0.0] * hours,
            "wind_speed_10m": [5.0] * hours,
            "relative_humidity_2m": [70] * hours,
        }
    }

    now = datetime(2026, 11, 22, 22, 15, tzinfo=_DHAKA_TZ)
    facts = classify(payload, "dhaka", "ঢাকা", now=now)

    assert len(facts.hourly_windows) == 48
    assert facts.hourly_windows[0].time_iso == "2026-11-22T22:00"
    assert facts.hourly_windows[-1].time_iso == "2026-11-24T21:00"
    # First 12 future hours (22:00..09:00) include rain probability 80, NOT morning 5.
    assert facts.rain_probability == 80
    assert facts.spray_window_ok is False
    # Safest window must pick tomorrow morning (07:00-11:00), never today's past 06:00.
    assert facts.safest_window_summary_bn is not None
    assert "আগামীকাল সকাল ৭টা থেকে ১১টা" in facts.safest_window_summary_bn
    assert "আজ সকাল" not in facts.safest_window_summary_bn


async def test_fetch_weather_requests_three_forecast_days(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.grounding import weather as weather_module

    weather_module._cache.clear()
    seen_params: list[dict] = []
    start = datetime.now(_DHAKA_TZ).replace(minute=0, second=0, microsecond=0) - timedelta(hours=3)
    payload = _make_48h_payload(start=start, hours=72)

    class FakeResponse:
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

        async def get(self, _url: str, params: dict) -> FakeResponse:
            seen_params.append(params)
            return FakeResponse()

    monkeypatch.setattr(weather_module.httpx, "AsyncClient", FakeClient)

    facts = await weather_module.fetch_weather(23.8103, 90.4125, "dhaka", "ঢাকা")
    assert facts is not None
    assert seen_params[0]["forecast_days"] == 3
    assert seen_params[0]["timezone"] == "Asia/Dhaka"
    assert len(facts.hourly_windows) == 48
    first_dt = weather_module._parse_dhaka_dt(facts.hourly_windows[0].time_iso)
    assert first_dt >= weather_module._current_dhaka_hour()


def test_classify_returns_none_for_missing_empty_or_expired_hourly() -> None:
    now = datetime(2026, 11, 23, 12, 0, tzinfo=_DHAKA_TZ)
    assert classify({}, "dhaka", "ঢাকা", now=now) is None
    assert classify({"hourly": None}, "dhaka", "ঢাকা", now=now) is None
    assert classify({"hourly": {}}, "dhaka", "ঢাকা", now=now) is None
    assert (
        classify(
            {
                "hourly": {
                    "time": [],
                    "precipitation_probability": [],
                    "wind_speed_10m": [],
                    "relative_humidity_2m": [],
                }
            },
            "dhaka",
            "ঢাকা",
            now=now,
        )
        is None
    )

    expired_payload = _make_48h_payload(start=now - timedelta(hours=48), hours=24)
    assert classify(expired_payload, "dhaka", "ঢাকা", now=now) is None


async def test_fetch_weather_returns_none_on_empty_hourly_and_does_not_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.grounding import weather as weather_module

    weather_module._cache.clear()
    payloads = [
        {"hourly": {"time": [], "precipitation_probability": [], "wind_speed_10m": []}},
        _make_48h_payload(
            start=datetime.now(_DHAKA_TZ).replace(minute=0, second=0, microsecond=0),
            hours=48,
        ),
    ]

    class FakeResponse:
        def __init__(self, payload: dict) -> None:
            self._payload = payload

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return self._payload

    class FakeClient:
        def __init__(self, **_kwargs) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args) -> None:
            return None

        async def get(self, _url: str, params: dict) -> FakeResponse:
            return FakeResponse(payloads.pop(0))

    monkeypatch.setattr(weather_module.httpx, "AsyncClient", FakeClient)

    first = await weather_module.fetch_weather(23.8103, 90.4125, "dhaka", "ঢাকা")
    assert first is None

    second = await weather_module.fetch_weather(23.8103, 90.4125, "dhaka", "ঢাকা")
    assert second is not None
    assert len(second.hourly_windows) == 48


def test_synthetic_payload_without_time_never_uses_2026_01_01() -> None:
    now = datetime(2026, 11, 23, 14, 0, tzinfo=_DHAKA_TZ)
    payload = {
        "hourly": {
            "precipitation_probability": [10] * 12,
            "precipitation": [0.0] * 12,
            "wind_speed_10m": [5.0] * 12,
            "relative_humidity_2m": [70] * 12,
        }
    }
    facts = classify(payload, "dhaka", "ঢাকা", now=now)
    assert facts is not None
    assert len(facts.hourly_windows) == 12
    assert not any(w.time_iso.startswith("2026-01-01") for w in facts.hourly_windows)
    assert facts.hourly_windows[0].time_iso.startswith("2026-11-23T14:00")


def test_exact_gps_coordinates_query_open_meteo_and_share_rounded_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app.grounding import weather as weather_module

    weather_module._cache.clear()
    seen_params: list[dict] = []
    start = datetime.now(_DHAKA_TZ).replace(minute=0, second=0, microsecond=0)
    payload = _make_48h_payload(start=start, hours=72)

    class FakeResponse:
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

        async def get(self, _url: str, params: dict) -> FakeResponse:
            seen_params.append(params)
            return FakeResponse()

    storage = tmp_path / "storage"
    monkeypatch.setattr(settings, "database_path", storage / "test.db")
    monkeypatch.setattr(settings, "storage_dir", storage)
    monkeypatch.setattr(settings, "tts_enabled", False)
    monkeypatch.setattr(settings, "vision_enabled", False)
    monkeypatch.setattr(settings, "gemini_api_key", "")
    monkeypatch.setattr(settings, "llm_api_key", "")
    monkeypatch.setattr(weather_module.httpx, "AsyncClient", FakeClient)
    reset_writer()
    for directory in (settings.images_dir, settings.audio_dir):
        directory.mkdir(parents=True, exist_ok=True)

    with TestClient(app) as client:
        # 1. Exact GPS coordinates near Comilla (centroid is 23.4607, 91.1809).
        first = client.post(
            "/api/v1/analyses",
            files={"image": ("crop.jpg", _jpeg(), "image/jpeg")},
            data={
                "crop": "rice",
                "district": "comilla",
                "latitude": "23.4712",
                "longitude": "91.1754",
            },
        )
        assert first.status_code == 202
        first_id = first.json()["id"]
        _wait(client, first_id)

        first_result = AnalysisResult.model_validate(
            client.get(f"/api/v1/analyses/{first_id}/result").json()
        )
        assert first_result.weather is not None
        assert first_result.weather.locationSource == "gps"
        assert first_result.weather.district == "কুমিল্লা (সঠিক অবস্থান)"
        assert len(seen_params) == 1
        assert seen_params[0]["latitude"] == pytest.approx(23.4712)
        assert seen_params[0]["longitude"] == pytest.approx(91.1754)

        # 2. Nearby GPS point (~500m away) shares the rounded gps:23.47,91.18 cache.
        second = client.post(
            "/api/v1/analyses",
            files={"image": ("crop.jpg", _jpeg(), "image/jpeg")},
            data={"crop": "rice", "latitude": "23.474", "longitude": "91.176"},
        )
        assert second.status_code == 202
        second_id = second.json()["id"]
        _wait(client, second_id)

        second_result = AnalysisResult.model_validate(
            client.get(f"/api/v1/analyses/{second_id}/result").json()
        )
        assert second_result.weather is not None
        assert second_result.weather.locationSource == "gps"
        assert second_result.weather.district == "কুমিল্লা (সঠিক অবস্থান)"
        assert len(seen_params) == 1

        # 3. Manual district without GPS falls back to district centroid.
        manual = client.post(
            "/api/v1/analyses",
            files={"image": ("crop.jpg", _jpeg(), "image/jpeg")},
            data={"crop": "rice", "district": "comilla"},
        )
        assert manual.status_code == 202
        manual_id = manual.json()["id"]
        _wait(client, manual_id)

        manual_result = AnalysisResult.model_validate(
            client.get(f"/api/v1/analyses/{manual_id}/result").json()
        )
        assert manual_result.weather is not None
        assert manual_result.weather.locationSource == "district"
        assert manual_result.weather.district == "কুমিল্লা"
        assert len(seen_params) == 2
        assert seen_params[1]["latitude"] == pytest.approx(23.4607)
        assert seen_params[1]["longitude"] == pytest.approx(91.1809)


@pytest.mark.parametrize(
    "coords",
    [
        {"latitude": "23.5"},
        {"longitude": "90.4"},
        {"latitude": "91.0", "longitude": "90.4"},
        {"latitude": "-90.5", "longitude": "90.4"},
        {"latitude": "23.5", "longitude": "181.0"},
        {"latitude": "23.5", "longitude": "-180.5"},
    ],
)
def test_invalid_gps_coordinates_return_400(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, coords: dict[str, str]
) -> None:
    storage = tmp_path / "storage"
    monkeypatch.setattr(settings, "database_path", storage / "test.db")
    monkeypatch.setattr(settings, "storage_dir", storage)
    monkeypatch.setattr(settings, "tts_enabled", False)
    monkeypatch.setattr(settings, "vision_enabled", False)
    for directory in (settings.images_dir, settings.audio_dir):
        directory.mkdir(parents=True, exist_ok=True)

    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/analyses",
            files={"image": ("crop.jpg", _jpeg(), "image/jpeg")},
            data={"crop": "rice", **coords},
        )
        assert resp.status_code == 400
        assert resp.json()["detail"] == "invalid_coordinates"
