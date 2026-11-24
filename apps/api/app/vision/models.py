"""Model spec for the leaf classifier.

A spec is everything about the checkpoint except its bytes: the label order,
how to map a label onto a crop and a knowledge-base entry, and how to
preprocess an image. The decision rule in `classifier.py` is shared.

The shipped model is the final EfficientNet-B0 export. It covers rice and
potato with 8 classes, and every one of those classes has a curated
knowledge-base entry: four rice diseases, two potato diseases, and the two
`*_healthy` classes. It has no catch-all class, so an unrecognised leaf is
refused by the confidence threshold rather than named as "other".
"""

from __future__ import annotations

from typing import TypedDict


class ModelSpec(TypedDict):
    """Everything about a checkpoint except its bytes."""

    id: str
    labels: tuple[str, ...]
    #: Model label -> crop. A label naming a different crop than the one the
    #: farmer selected is treated as "no answer", not as a diagnosis.
    label_crop: dict[str, str]
    #: Model label -> curated knowledge-base entry. Labels absent here are
    #: recognised by the model but have no sourced treatment, so they must not
    #: become a diagnosis.
    label_disease: dict[str, str]
    #: Healthy classes. Correct, and a real answer — they map to a KB entry that
    #: says the plant looks healthy and gives no chemical.
    healthy: frozenset[str]
    input_size: int
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    #: ONNX input node name. Part of the graph, not a convention.
    input_name: str
    #: Filename to look for in `settings.vision_model_dir`.
    file: str


# --------------------------------------------------------------------------
# EfficientNet-B0, 8 classes: rice and potato only. THE shipped model.
# Order is part of the ONNX graph contract — a silent reorder is a bug.
# --------------------------------------------------------------------------
EFFICIENTNET_LABELS: tuple[str, ...] = (
    "potato_early_blight",
    "potato_healthy",
    "potato_late_blight",
    "rice_bacterial_leaf_blight",
    "rice_blast",
    "rice_brown_spot",
    "rice_healthy",
    "rice_tungro",
)

_EFFICIENTNET_LABEL_CROP: dict[str, str] = {
    "potato_early_blight": "potato",
    "potato_healthy": "potato",
    "potato_late_blight": "potato",
    "rice_bacterial_leaf_blight": "rice",
    "rice_blast": "rice",
    "rice_brown_spot": "rice",
    "rice_healthy": "rice",
    "rice_tungro": "rice",
}

_EFFICIENTNET_LABEL_DISEASE: dict[str, str] = {
    "potato_early_blight": "potato_early_blight",
    "potato_late_blight": "potato_late_blight",
    "potato_healthy": "potato_healthy",
    "rice_healthy": "rice_healthy",
    "rice_blast": "rice_blast",
    "rice_brown_spot": "rice_brown_spot",
    "rice_tungro": "rice_tungro",
    # The checkpoint spells this `_leaf_blight`; the knowledge base calls the
    # entry `rice_bacterial_blight`. Mapped explicitly rather than by string
    # surgery, so a rename on either side fails a test instead of going quiet.
    "rice_bacterial_leaf_blight": "rice_bacterial_blight",
}

EFFICIENTNET: ModelSpec = {
    "id": "efficientnet",
    "labels": EFFICIENTNET_LABELS,
    "label_crop": _EFFICIENTNET_LABEL_CROP,
    "label_disease": _EFFICIENTNET_LABEL_DISEASE,
    "healthy": frozenset({"potato_healthy", "rice_healthy"}),
    "input_size": 320,
    "mean": (0.485, 0.456, 0.406),
    "std": (0.229, 0.224, 0.225),
    "input_name": "input",
    "file": "foshollens_leaf.onnx",
}
