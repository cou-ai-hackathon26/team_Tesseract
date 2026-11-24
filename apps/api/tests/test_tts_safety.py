"""The TTS normaliser's two dangerous behaviours, captured as tests.

Both were found by running the model, not by reading its docs. They are the
reason `prepare_text` exists and the reason the medicine-name assertion is not
optional.
"""

from __future__ import annotations

import pytest

from app.tts.engine import TtsSafetyError, prepare_text


def test_slash_becomes_per_not_or() -> None:
    """The model reads `/` as "বা" (or), turning a dose ratio into nonsense."""
    assert prepare_text("২ মিলি/লিটার") == "২ মিলি প্রতি লিটার"


def test_per_litre_dose_survives() -> None:
    assert prepare_text("প্রতি লিটারে ০.৬ গ্রাম") == "প্রতি লিটারে ০.৬ গ্রাম"


def test_percent_is_spelled_out() -> None:
    """Aloud, "97%" is "আশি শতাংশ"-shaped, and the model handles the digits."""
    assert prepare_text("বৃষ্টির সম্ভাবনা ৯৭%") == "বৃষ্টির সম্ভাবনা ৯৭ শতাংশ"


def test_markdown_noise_is_stripped() -> None:
    assert prepare_text("**জরুরি** `পরামর্শ`") == "জরুরি পরামর্শ"


def test_latin_is_rejected() -> None:
    """The model deletes Latin spans, so a Latin medicine name would vanish."""
    with pytest.raises(TtsSafetyError, match="latin"):
        prepare_text("Mancozeb ব্যবহার করুন")


def test_medicine_lost_during_cleaning_is_rejected() -> None:
    """If a named medicine is stripped out of the text, refuse the audio.

    This is the dangerous shape: the sentence stays fluent and complete-sounding
    while the actual product name has been deleted.
    """
    with pytest.raises(TtsSafetyError, match="ট্রাইসাইক্লাজল"):
        prepare_text("*ট্রাইসাইক্লাজল* ব্যবহার করুন", ["*ট্রাইসাইক্লাজল*"])


def test_medicine_absent_from_text_is_not_required() -> None:
    """Only terms the narrative actually names are required.

    A narrative recommends one medicine, not every candidate the knowledge base
    holds; demanding all of them refused valid audio.
    """
    prepared = prepare_text("ট্রাইসাইক্লাজল ব্যবহার করুন", ["ট্রাইসাইক্লাজল", "ম্যানকোজেব"])
    assert "ট্রাইসাইক্লাজল" in prepared
