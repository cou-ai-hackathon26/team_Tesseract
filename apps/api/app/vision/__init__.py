"""Vision: leaf disease classification.

Optional. Without the checkpoint, or when the classifier cannot place a photo,
the pipeline falls back to an unconfirmed knowledge-base entry and says so — it
never invents a confidence.

The shipped model is EfficientNet-B0: rice and potato, 8 classes, with a curated
entry for every one of them. It has no catch-all class.
"""

from .classifier import (
    LeafClassifier,
    VisionFacts,
    get_classifier,
    reset_classifier,
)
from .models import (
    EFFICIENTNET,
    ModelSpec,
)

__all__ = [
    "EFFICIENTNET",
    "LeafClassifier",
    "ModelSpec",
    "VisionFacts",
    "get_classifier",
    "reset_classifier",
]
