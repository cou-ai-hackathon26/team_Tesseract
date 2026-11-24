"""The spray verdict.

This is the one place the product decides something about a farmer's money:
spraying in the rain wastes a dose and a day's work. It is computed in code,
never by the language model, so it is tested directly.
"""

from __future__ import annotations

from app.grounding.knowledge_base import KB, crops, diseases_for_crop, get_disease
from app.grounding.weather import RAIN_THRESHOLD_PCT, WIND_THRESHOLD_KMH, _decide, classify

PAYLOAD = {
    "hourly": {
        "precipitation_probability": [10] * 12,
        "precipitation": [0.0] * 12,
        "wind_speed_10m": [5.0] * 12,
        "relative_humidity_2m": [70] * 12,
    }
}


def test_dry_and_calm_is_ok() -> None:
    ok, reason = _decide(10, 5.0)
    assert ok is True
    assert "অনুকূল" in reason


def test_rain_blocks_spraying() -> None:
    ok, reason = _decide(RAIN_THRESHOLD_PCT, 0.0)
    assert ok is False
    assert "বৃষ্টি" in reason


def test_wind_blocks_spraying() -> None:
    ok, reason = _decide(0, WIND_THRESHOLD_KMH)
    assert ok is False
    assert "বাতাস" in reason


def test_rain_is_named_when_both_block() -> None:
    """The reason names the binding constraint, not a list of complaints."""
    ok, reason = _decide(95, 40.0)
    assert ok is False
    assert "বৃষ্টি" in reason
    assert "বাতাস" not in reason


def test_reason_uses_bengali_digits() -> None:
    _, reason = _decide(97, 0.0)
    assert "৯৭" in reason
    assert "97" not in reason
    assert "১২" in reason


def test_classify_uses_the_maximum_over_the_horizon() -> None:
    """A single rainy hour in the next 12 blocks the window."""
    payload = {
        "hourly": {
            "precipitation_probability": [5, 5, 90] + [5] * 9,
            "precipitation": [0.0] * 12,
            "wind_speed_10m": [5.0] * 12,
            "relative_humidity_2m": [70] * 12,
        }
    }
    facts = classify(payload, "dhaka", "ঢাকা")
    assert facts.rain_probability == 90
    assert facts.spray_window_ok is False


def test_classify_ignores_missing_values() -> None:
    payload = {
        "hourly": {
            "precipitation_probability": [None, 20] + [None] * 10,
            "precipitation": [None] * 12,
            "wind_speed_10m": [None, 8.0] + [None] * 10,
            "relative_humidity_2m": [60] * 12,
        }
    }
    facts = classify(payload, "dhaka", "ঢাকা")
    assert facts.rain_probability == 20
    assert facts.wind_speed_max == 8.0
    assert facts.spray_window_ok is True


def test_every_crop_has_a_disease() -> None:
    for crop in crops():
        assert diseases_for_crop(crop["id"]), crop["id"]


def test_knowledge_base_entries_are_complete() -> None:
    for entry in KB:
        assert entry["bnName"]
        assert entry["symptoms"], entry["id"]
        assert entry["prevention"], entry["id"]
        assert entry["source"], entry["id"]
        # Every chemical needs a Bangla name: TTS deletes Latin ones.
        for chemical in entry["chemical"]:
            assert chemical["bnName"]
            assert not any("a" <= ch.lower() <= "z" for ch in chemical["bnName"]), (
                f"{entry['id']} has Latin in bnName: {chemical['bnName']}"
            )


def test_get_disease_roundtrip() -> None:
    assert get_disease("rice_blast") is not None
    assert get_disease("does_not_exist") is None
