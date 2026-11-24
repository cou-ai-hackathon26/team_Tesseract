"""The narrative validator: the guard between a model and a farmer.

Each test here corresponds to a way the model has actually produced something
unusable, or to a check that keeps an invented pesticide out of the advice.
"""

from __future__ import annotations

import pytest

from app.llm.writer import InvalidNarrative, normalise, validate_narrative

CHEMICALS = ["ট্রাইসাইক্লাজল", "ম্যানকোজেব"]

GOOD = (
    "আপনার ধান গাছে সম্ভবত ব্লাস্ট রোগ হয়েছে। "
    "পাতায় ধূসর-সবুজ ডিম্বাকৃতি দাগ দেখা যাচ্ছে। "
    "প্রাথমিকভাবে ট্রাইসাইক্লাজল ব্যবহার করতে পারেন, প্রয়োগের আগে কৃষি কর্মকর্তার পরামর্শ নিন। "
    "স্প্রে করার সময় গ্লাভস ও মাস্ক পরুন এবং কাজ শেষে হাত ধুয়ে নিন।"
)


def test_accepts_a_clean_narrative() -> None:
    assert validate_narrative(GOOD, CHEMICALS) == normalise(GOOD)


def test_rejects_empty() -> None:
    with pytest.raises(InvalidNarrative, match="empty"):
        validate_narrative("   ", CHEMICALS)


def test_rejects_non_bangla() -> None:
    with pytest.raises(InvalidNarrative, match="no_bangla"):
        validate_narrative("spray the crop now please", CHEMICALS)


def test_rejects_latin_script() -> None:
    """Latin must never reach TTS, which silently deletes it."""
    with pytest.raises(InvalidNarrative, match="latin_script"):
        validate_narrative(GOOD + " Mancozeb", CHEMICALS)


def test_rejects_markdown() -> None:
    with pytest.raises(InvalidNarrative, match="markdown"):
        validate_narrative(GOOD + " **জরুরি**", CHEMICALS)


def test_rejects_emoji() -> None:
    with pytest.raises(InvalidNarrative, match="emoji"):
        validate_narrative(GOOD + " 🌾", CHEMICALS)


def test_rejects_too_short() -> None:
    with pytest.raises(InvalidNarrative, match="word_count"):
        validate_narrative("আপনার ধানে ব্লাস্ট হয়েছে।", CHEMICALS)


def test_rejects_too_long() -> None:
    with pytest.raises(InvalidNarrative, match="word_count"):
        validate_narrative(" ".join(["ধান"] * 200), CHEMICALS)


def test_rejects_invented_medicine() -> None:
    """The most dangerous failure: a pesticide the knowledge base never supplied.

    Latin leaks first, so the check is that *something* rejects it. The Bangla
    branch is covered by the chemical-token rule below.
    """
    text = GOOD + " অথবা carbendazim ব্যবহার করুন।"
    with pytest.raises(InvalidNarrative, match=r"unknown_medicine|latin_script"):
        validate_narrative(text, CHEMICALS)


def test_rejects_unknown_bangla_medicine() -> None:
    """A Bangla name the KB never supplied must be rejected too."""
    text = GOOD + " অথবা কার্বেন্ডাজিম ব্যবহার করুন।"
    with pytest.raises(InvalidNarrative, match="unknown_medicine"):
        validate_narrative(text, CHEMICALS)


def test_normalise_collapses_whitespace() -> None:
    assert normalise("ধান   গাছ\n\nরোগ") == "ধান গাছ রোগ"
