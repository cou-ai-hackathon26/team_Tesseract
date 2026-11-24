"""The classifier's decision logic.

These tests exercise `_decide` with injected probabilities instead of running
the model: the weights are a 16 MB artifact, and the interesting decisions are
all about what to do with a probability vector, not about computing one.

The rule under test is the one in `predict_final.py`: choose the best class
inside the selected crop, but judge it on its raw, unmasked probability. Masking
is for ranking only — thresholding the renormalised value would let a
wrong-crop photo clear the bar, because renormalisation throws away the
evidence that the photo was mostly another crop.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app.config import settings
from app.vision.classifier import LeafClassifier, _decide, _softmax, missing_weights
from app.vision.models import EFFICIENTNET, ModelSpec


def _probs(spec: ModelSpec, **weights: float) -> np.ndarray:
    """Build a probability vector by label name, zero elsewhere."""
    vector = np.zeros(len(spec["labels"]), dtype=np.float32)
    for label, weight in weights.items():
        vector[spec["labels"].index(label)] = weight
    return vector


# ------------------------------------------------------------ checkpoint -----
def test_labels_match_the_shipped_checkpoint() -> None:
    assert len(EFFICIENTNET["labels"]) == 8
    assert EFFICIENTNET["input_name"] == "input"
    assert EFFICIENTNET["input_size"] == 320
    # ImageNet statistics.
    assert EFFICIENTNET["mean"] == (0.485, 0.456, 0.406)
    # Order is part of the model contract, so a silent reorder is a bug.
    assert EFFICIENTNET["labels"][3] == "rice_bacterial_leaf_blight"


def test_confident_rice_blast_resolves_to_the_kb_entry() -> None:
    facts = _decide(_probs(EFFICIENTNET, rice_blast=0.9), "rice", EFFICIENTNET)
    assert facts["status"] == "identified"
    assert facts["disease_id"] == "rice_blast"
    assert facts["confidence"] == pytest.approx(0.9)
    assert facts["model"] == "efficientnet"


def test_below_threshold_is_undetermined() -> None:
    facts = _decide(_probs(EFFICIENTNET, rice_blast=0.4), "rice", EFFICIENTNET)
    assert facts["status"] == "undetermined"
    assert facts["disease_id"] is None
    # The raw number is kept for the UI, but never promoted to a diagnosis.
    assert facts["confidence"] == pytest.approx(0.4)


def test_margin_accepts_a_clear_winner_below_the_main_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A clear winner just under the threshold is not a refusal.

    Raw 0.50 is below the 0.60 primary threshold, but it beats the runner-up by
    0.40 and the photo is 70% rice, so the margin path accepts it.
    """
    monkeypatch.setattr(settings, "vision_confidence_threshold", 0.6)
    monkeypatch.setattr(settings, "vision_margin_floor", 0.45)
    monkeypatch.setattr(settings, "vision_margin", 0.30)
    monkeypatch.setattr(settings, "vision_margin_min_crop_mass", 0.70)

    facts = _decide(
        _probs(
            EFFICIENTNET,
            rice_blast=0.50,
            rice_brown_spot=0.10,
            rice_healthy=0.10,
            potato_late_blight=0.05,
        ),
        "rice",
        EFFICIENTNET,
    )
    assert facts["status"] == "identified"
    assert facts["disease_id"] == "rice_blast"
    assert facts["confidence"] == pytest.approx(0.50)


def test_margin_requires_a_clear_gap(monkeypatch: pytest.MonkeyPatch) -> None:
    """Two close classes are ambiguous, even if the winner clears the floor."""
    monkeypatch.setattr(settings, "vision_confidence_threshold", 0.6)
    monkeypatch.setattr(settings, "vision_margin_floor", 0.45)
    monkeypatch.setattr(settings, "vision_margin", 0.30)
    monkeypatch.setattr(settings, "vision_margin_min_crop_mass", 0.70)

    facts = _decide(
        _probs(EFFICIENTNET, rice_blast=0.50, rice_brown_spot=0.40, potato_late_blight=0.05),
        "rice",
        EFFICIENTNET,
    )
    assert facts["status"] == "undetermined"


def test_margin_requires_most_of_the_belief_in_the_selected_crop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A clear potato winner must not be accepted as a rice answer."""
    monkeypatch.setattr(settings, "vision_confidence_threshold", 0.6)
    monkeypatch.setattr(settings, "vision_margin_floor", 0.45)
    monkeypatch.setattr(settings, "vision_margin", 0.30)
    monkeypatch.setattr(settings, "vision_margin_min_crop_mass", 0.70)

    facts = _decide(
        _probs(EFFICIENTNET, rice_blast=0.50, potato_late_blight=0.40),
        "rice",
        EFFICIENTNET,
    )
    assert facts["status"] == "undetermined"


def test_margin_does_not_go_below_its_floor(monkeypatch: pytest.MonkeyPatch) -> None:
    """A large gap at very low confidence is still not enough evidence."""
    monkeypatch.setattr(settings, "vision_confidence_threshold", 0.6)
    monkeypatch.setattr(settings, "vision_margin_floor", 0.45)
    monkeypatch.setattr(settings, "vision_margin", 0.30)
    monkeypatch.setattr(settings, "vision_margin_min_crop_mass", 0.70)

    facts = _decide(
        _probs(
            EFFICIENTNET,
            rice_blast=0.44,
            rice_healthy=0.16,
            rice_brown_spot=0.05,
            rice_bacterial_leaf_blight=0.05,
            potato_late_blight=0.10,
        ),
        "rice",
        EFFICIENTNET,
    )
    assert facts["status"] == "undetermined"


def test_confidence_is_raw_not_renormalised() -> None:
    """Masking ranks classes; it must not inflate the reported confidence.

    `rice_blast` is 0.7 of the full distribution and 0.875 of the rice-only
    mass. The farmer sees 0.7, because that is the model's actual evidence.
    """
    facts = _decide(
        _probs(EFFICIENTNET, rice_blast=0.7, rice_brown_spot=0.1, potato_late_blight=0.2),
        "rice",
        EFFICIENTNET,
    )
    assert facts["status"] == "identified"
    assert facts["disease_id"] == "rice_blast"
    assert facts["confidence"] == pytest.approx(0.7)


def test_threshold_is_applied_to_raw_mass_not_renormalised() -> None:
    """A potato leaf must not become a confident rice diagnosis.

    `rice_blast` is the highest *rice* class here, but its raw probability is
    0.015. Renormalising onto rice first — then thresholding — would report
    ~0.75 confidence; the raw probability is 0.015 and must be judged as such.
    """
    facts = _decide(
        _probs(
            EFFICIENTNET,
            potato_late_blight=0.5,
            potato_early_blight=0.48,
            rice_blast=0.015,
        ),
        "rice",
        EFFICIENTNET,
    )
    assert facts["status"] == "undetermined"
    assert facts["disease_id"] is None
    assert facts["confidence"] == pytest.approx(0.015)


def test_winning_label_from_another_crop_is_undetermined() -> None:
    facts = _decide(
        _probs(EFFICIENTNET, potato_late_blight=0.9, rice_blast=0.02),
        "rice",
        EFFICIENTNET,
    )
    assert facts["status"] == "undetermined"
    assert facts["confidence"] == pytest.approx(0.02)


def test_covers_bacterial_blight() -> None:
    facts = _decide(_probs(EFFICIENTNET, rice_bacterial_leaf_blight=0.88), "rice", EFFICIENTNET)
    assert facts["status"] == "identified"
    # Checkpoint spelling differs from the KB id; the mapping is explicit.
    assert facts["disease_id"] == "rice_bacterial_blight"


def test_tungro_resolves_to_its_kb_entry() -> None:
    facts = _decide(_probs(EFFICIENTNET, rice_tungro=0.9), "rice", EFFICIENTNET)
    assert facts["status"] == "identified"
    assert facts["disease_id"] == "rice_tungro"
    assert facts["is_healthy"] is False


def test_brown_spot_resolves_to_its_kb_entry() -> None:
    """The regression that started this: brown spot was recognised and dropped."""
    facts = _decide(_probs(EFFICIENTNET, rice_brown_spot=0.96), "rice", EFFICIENTNET)
    assert facts["status"] == "identified"
    assert facts["disease_id"] == "rice_brown_spot"
    assert facts["confidence"] == pytest.approx(0.96)


@pytest.mark.parametrize(
    ("label", "crop", "disease_id"),
    [
        ("rice_blast", "rice", "rice_blast"),
        ("rice_bacterial_leaf_blight", "rice", "rice_bacterial_blight"),
        ("rice_brown_spot", "rice", "rice_brown_spot"),
        ("rice_tungro", "rice", "rice_tungro"),
        ("potato_early_blight", "potato", "potato_early_blight"),
        ("potato_late_blight", "potato", "potato_late_blight"),
    ],
)
def test_every_disease_label_is_surfaced(label: str, crop: str, disease_id: str) -> None:
    """Every disease class the shipped model can emit must reach the farmer."""
    facts = _decide(_probs(EFFICIENTNET, **{label: 0.9}), crop, EFFICIENTNET)
    assert facts["status"] == "identified"
    assert facts["disease_id"] == disease_id


def test_respects_the_requested_crop() -> None:
    """A potato label must not answer a rice question."""
    facts = _decide(_probs(EFFICIENTNET, potato_late_blight=0.95), "rice", EFFICIENTNET)
    assert facts["status"] == "undetermined"


@pytest.mark.parametrize(
    ("label", "crop", "disease_id"),
    [("rice_healthy", "rice", "rice_healthy"), ("potato_healthy", "potato", "potato_healthy")],
)
def test_healthy_is_surfaced_as_a_healthy_answer(
    label: str, crop: str, disease_id: str
) -> None:
    """Healthy is a real result, not a miss.

    It resolves to the crop's healthy entry, which carries no chemical, so the
    farmer is told the plant looks fine instead of being left with "could not
    identify" or — worse — a spray.
    """
    facts = _decide(_probs(EFFICIENTNET, **{label: 0.95}), crop, EFFICIENTNET)
    assert facts["status"] == "identified"
    assert facts["is_healthy"] is True
    assert facts["disease_id"] == disease_id

    from app.grounding import knowledge_base

    entry = knowledge_base.get_disease(disease_id)
    assert entry is not None
    assert entry["chemical"] == []


def test_crops_outside_rice_and_potato_are_undetermined() -> None:
    """Tomato and chili are out of scope: the checkpoint cannot name them."""
    for crop in ("tomato", "chili"):
        label = EFFICIENTNET["labels"][0]
        facts = _decide(_probs(EFFICIENTNET, **{label: 0.95}), crop, EFFICIENTNET)
        assert facts["status"] == "undetermined", crop


# ------------------------------------------------------------- shared ------
def test_threshold_comes_from_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "vision_confidence_threshold", 0.95)
    # Two close classes: the margin path cannot rescue this, so the raised
    # threshold is what keeps it undetermined.
    assert (
        _decide(
            _probs(EFFICIENTNET, rice_blast=0.9, rice_brown_spot=0.85),
            "rice",
            EFFICIENTNET,
        )["status"]
        == "undetermined"
    )

    monkeypatch.setattr(settings, "vision_confidence_threshold", 0.2)
    assert (
        _decide(_probs(EFFICIENTNET, rice_blast=0.3), "rice", EFFICIENTNET)["status"]
        == "identified"
    )


def test_softmax_is_numerically_stable() -> None:
    """Large logits must not overflow to nan."""
    logits = np.array([1000.0, 1001.0, 999.0], dtype=np.float32)
    probabilities = _softmax(logits)
    assert not np.isnan(probabilities).any()
    assert probabilities.sum() == pytest.approx(1.0, abs=1e-6)


def test_spec_only_maps_labels_onto_real_kb_entries() -> None:
    """A typo in a label_to_disease table would otherwise fail silently."""
    from app.grounding import knowledge_base

    for label, disease_id in EFFICIENTNET["label_disease"].items():
        assert label in EFFICIENTNET["labels"], f"{label} not in labels"
        assert knowledge_base.get_disease(disease_id) is not None, (
            f"{label} -> {disease_id} has no KB entry"
        )


# ------------------------------------------------------- health truthfulness
def test_status_is_ready_only_when_the_local_checkpoint_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`VISION_ENABLED=true` must not mean `vision: ready` without the file.

    A fresh clone has the flag set and no weights, so `/health` would claim the
    stage was fine while every upload silently returned `undetermined`.
    """
    monkeypatch.setattr(settings, "vision_enabled", True)

    monkeypatch.setattr(settings, "vision_model_dir", tmp_path)
    assert LeafClassifier(EFFICIENTNET).status == "unavailable"
    assert missing_weights(EFFICIENTNET) == tmp_path / EFFICIENTNET["file"]

    (tmp_path / EFFICIENTNET["file"]).write_bytes(b"not really a model")
    assert LeafClassifier(EFFICIENTNET).status == "ready"
    assert missing_weights(EFFICIENTNET) is None


def test_status_is_unavailable_when_vision_is_disabled(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(settings, "vision_enabled", False)
    monkeypatch.setattr(settings, "vision_model_dir", tmp_path)
    (tmp_path / EFFICIENTNET["file"]).write_bytes(b"not really a model")
    assert LeafClassifier(EFFICIENTNET).status == "unavailable"


def test_relative_model_dir_is_anchored_to_the_api_package(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`.env.example` documents `./models`; that must not be CWD-relative."""
    from app.config import API_ROOT, Settings

    env = tmp_path / ".env"
    env.write_text("VISION_MODEL_DIR=./models\n", encoding="utf-8")
    parsed = Settings(_env_file=env)

    assert parsed.vision_model_dir.is_absolute()
    assert parsed.vision_model_dir == (API_ROOT / "models").resolve()


def test_absolute_model_dir_is_left_alone(tmp_path: Path) -> None:
    from app.config import Settings

    env = tmp_path / ".env"
    env.write_text(f"VISION_MODEL_DIR={tmp_path / 'weights'}\n", encoding="utf-8")
    assert Settings(_env_file=env).vision_model_dir == (tmp_path / "weights").resolve()


# ------------------------------------------------------------ coverage ------
def test_every_label_is_mapped() -> None:
    """No label may silently fall through unmapped.

    This is the guard for the brown-spot bug: a class the model is confident
    about but the app has no treatment for is a dead end for the farmer. Adding
    a class to the checkpoint must force a mapping decision here.
    """
    for label in EFFICIENTNET["labels"]:
        assert label in EFFICIENTNET["label_disease"], label


def test_has_no_missing_kb_entries() -> None:
    from app.grounding import knowledge_base

    for label, disease_id in EFFICIENTNET["label_disease"].items():
        assert knowledge_base.get_disease(disease_id) is not None, (label, disease_id)


def test_supported_crops_match_the_model_classes() -> None:
    """The crops the app offers must be exactly the crops the model can name."""
    from app.grounding import knowledge_base

    model_crops = set(EFFICIENTNET["label_crop"].values())
    assert model_crops == set(knowledge_base.SUPPORTED_CROPS) == {"rice", "potato"}


def test_knowledge_base_has_no_entries_outside_supported_crops() -> None:
    from app.grounding import knowledge_base

    for entry in knowledge_base.KB:
        assert entry["cropId"] in knowledge_base.SUPPORTED_CROPS, entry["id"]


def test_logs_top3_probabilities_and_crop_mass(
    caplog: pytest.LogCaptureFixture,
) -> None:
    import logging

    caplog.set_level(logging.INFO, logger="app.vision.classifier")

    split_probs = _probs(
        EFFICIENTNET,
        potato_early_blight=0.354,
        potato_late_blight=0.123,
        potato_healthy=0.050,
        rice_blast=0.166,
        rice_brown_spot=0.168,
    )
    potato_facts = _decide(split_probs, "potato", EFFICIENTNET)
    # Raw 0.354 is below the 0.6 threshold, so this stays undetermined even
    # though potato classes hold most of the mass.
    assert potato_facts["status"] == "undetermined"
    assert potato_facts["confidence"] == pytest.approx(0.354)
    assert any("vision top3 (efficientnet, crop=potato)" in r.message for r in caplog.records)
    assert any("potato_early_blight=0.354" in r.message for r in caplog.records)
    assert any("crop_mass=0.527" in r.message for r in caplog.records)

    uniform_probs = np.full(len(EFFICIENTNET["labels"]), 1.0 / len(EFFICIENTNET["labels"]))
    assert _decide(uniform_probs, "potato", EFFICIENTNET)["status"] == "undetermined"
